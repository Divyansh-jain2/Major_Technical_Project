# ============================================================================
# STEP 11: GENERATE SYNTHETIC EXTREMES WITH SEVERAL GENERATORS (GAN, VAE, COPULA)
# Paste into ONE Colab cell. Runtime > Change runtime type > T4 GPU. ~45-70 min.
# Re-run the same cell after a disconnect: finished generators are skipped.
#
# Research question (Steps 11-12): with ONE fixed classifier (LightGBM, next-day heat extreme
# from the last days and weeks), which way of generating extra extreme days helps most?
#
# Generators trained here, all on the same 29,604 real training extremes (same features as v3):
#   gaussian_copula  statistical baseline (SDV)                     unconditional
#   tvae             variational autoencoder (CTGAN package)        unconditional
#   ctgan            conditional tabular GAN (CTGAN package)        unconditional (re-trained here,
#                    so every generator is trained the same way)
#   copulagan        CTGAN on copula-transformed columns (SDV)      unconditional
#   cwgan            our conditional Wasserstein GAN with gradient  CONDITIONAL on season + severity,
#                    penalty, same inputs as our diffusion model    like the diffusion model
# Already made in Step 7 (not retrained): diffusion (ddpm_emp, ddpm_evt_tail).
# SMOTE / Borderline-SMOTE / ADASYN are made on the fly in Step 12.
#
# Two sets per generator, so EVT's effect can be measured for every generator:
#   <gen>_plain  the generator as it is
#   <gen>_tail   EVT tail focus: 60% of rows from the top 20% of the severity tail, like
#                ddpm_evt_tail. Unconditional generators: a large pool is generated, every row's
#                severity is scored by a "judge" (LightGBM trained on real extremes) and turned
#                into a GPD tail probability, and rows are picked to match the target. A generator
#                that rarely makes severe days has to REPEAT its few severe rows: we report that.
#                cwgan: the severities are requested directly from the GPD, as for diffusion.
# Every row goes through the same physics clean-up as Step 7 (real grid cell, Tmin<=Tmean<=Tmax,
# ranges, exact distance features). Raw rule-breaking BEFORE clean-up is measured and reported.
#
# Reads (never changes): data/model_ready_v3/*, results/step3b/*, results/step7/synthetic_ddpm_*.parquet
# Writes ONLY to:        results/step11/
# ============================================================================
!pip install -q lightgbm sdv==2.2.0

import os, json, time, shutil, warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import KFold
import lightgbm as lgb
import torch
import torch.nn as nn
from google.colab import drive

warnings.filterwarnings("ignore")
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/himachal_project"
V3, S7, OUT = f"{BASE}/data/model_ready_v3", f"{BASE}/results/step7", f"{BASE}/results/step11"
LOCAL = "/content/local_v3"

# ---------------------------------------------------------------- settings
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GPU = DEV == "cuda"
GENERATORS = ["gaussian_copula", "tvae", "ctgan", "copulagan", "cwgan"]
EPOCHS = 300 if GPU else 60            # TVAE, CTGAN, CopulaGAN (package defaults: 300)
WGAN_STEPS = 6000 if GPU else 1000     # generator updates for cwgan (5 critic updates each)
POOL_FACTOR = 10                       # unconditional generators: pool = 10 x real extremes
OUT_FACTOR = 3                         # each saved set = 3 x real extremes (Step 12 draws 1x per seed)
TAIL_FROM, TAIL_SHARE = 0.8, 0.6       # EVT tail focus: 60% of rows from the top 20% of the tail
SEED = 42
# ----------------------------------------------------------------
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


def drive_ok():
    for _ in range(3):
        try:
            os.listdir(BASE)
            return
        except OSError:
            print("   Drive connection lost, re-mounting ...", flush=True)
            time.sleep(5)
            drive.mount("/content/drive", force_remount=True)
    os.listdir(BASE)


def copy_local(src):
    dst = f"{LOCAL}/{os.path.basename(src)}"
    for _ in range(4):
        try:
            drive_ok()
            if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
                shutil.copy(src, dst + ".part")
                os.replace(dst + ".part", dst)
            return dst
        except OSError as e:
            print(f"   copy failed ({e}); retrying", flush=True)
            time.sleep(10)
    raise RuntimeError(f"could not copy {src}; re-run the cell")


os.makedirs(LOCAL, exist_ok=True)
for f in ("train.parquet", "features.json", "thresholds.nc", "cells.csv"):
    copy_local(f"{V3}/{f}")
drive_ok()
os.makedirs(OUT, exist_ok=True)
LINES, T0 = [], time.time()
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def progress(msg):
    print(f"   ... [{(time.time() - T0) / 60:4.1f} min] {msg}", flush=True)


def save_summary():
    drive_ok()
    txt = "\n".join(LINES) + "\n"
    for p in (f"{OUT}/step11_summary.txt", f"{BASE}/step11_summary.txt"):
        open(p, "w").write(txt)


def section(t):
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


out(f"STEP 11  {time.strftime('%Y-%m-%d %H:%M')}  device={DEV}  epochs={EPOCHS}  cwgan steps={WGAN_STEPS}")

# ================================================================ 1. real extremes, EVT fits (as Step 7)
section("1. REAL EXTREMES AND EVT FITS")
meta = json.load(open(f"{LOCAL}/features.json"))
BASE_FEATS, FEATS, SNAMES = meta["BASE_FEATS"], meta["FEATS"], meta["SEASONS"]
tr = pd.read_parquet(f"{LOCAL}/train.parquet", columns=BASE_FEATS + [c for c in meta["DERIVED"]] + ["y", "z", "season"])
pos = tr[tr.y == 1].reset_index(drop=True)
del tr
n_pos = len(pos)
N_POOL, N_OUT = POOL_FACTOR * n_pos, OUT_FACTOR * n_pos
season_share = pos.season.value_counts(normalize=True).sort_index()
out(f"{n_pos:,} real extremes -> pool {N_POOL:,} rows per unconditional generator, {N_OUT:,} rows per saved set")

cells = pd.read_csv(f"{LOCAL}/cells.csv")
thr_sea = xr.open_dataset(f"{LOCAL}/thresholds.nc")["thr_heat_season"].values
drive_ok()
sc_full = pd.read_csv(f"{BASE}/results/step3b/cell_scales.csv").pivot(
    index="cell", columns="season_next", values="heat_scale").reindex(cells.cell).values
tree = cKDTree(np.c_[cells.lat.values, cells.lon.values])
fits = pd.read_csv(f"{BASE}/results/step3b/evt_fits_std.csv").set_index("group")
GPD = {}
for k, s in enumerate(SNAMES):
    xi, sig = float(fits.loc[f"heat/{s}", "xi_std"]), float(fits.loc[f"heat/{s}", "sigma_std"])
    GPD[k] = (xi, sig, sig / -xi if xi < 0 else np.inf)
Z_CAP = {k: 1.5 * float(pos.z[pos.season == k].max()) for k in range(len(SNAMES))}
LAT, LON = (cells.lat.min(), cells.lat.max()), (cells.lon.min(), cells.lon.max())


def construct(X, season, z):
    """Same as Step 7: snap to a real cell, repair physics, compute distance features exactly."""
    df = pd.DataFrame(np.asarray(X, dtype="float64"), columns=BASE_FEATS)
    _, idx = tree.query(np.c_[df.lat.values, df.lon.values])
    c = cells.iloc[idx]
    df["lat"], df["lon"], df["elev_m"], df["slope_deg"] = c.lat.values, c.lon.values, c.elev_m.values, c.slope_deg.values
    df["cell"] = c.cell.values
    df["season"] = np.asarray(season).astype(int)
    df["z"] = z
    for l in (0, 1, 2):
        cols = [f"t2m_min_l{l}", f"t2m_mean_l{l}", f"t2m_max_l{l}"]
        df[cols] = np.sort(df[cols].values, axis=1)
        df[f"rh_l{l}"] = df[f"rh_l{l}"].clip(0, 100)
        df[f"tp_l{l}"] = df[f"tp_l{l}"].clip(lower=0)
        df[f"ssrd_l{l}"] = df[f"ssrd_l{l}"].clip(lower=0)
        df[f"swvl1_l{l}"] = df[f"swvl1_l{l}"].clip(0, 0.8)
        df[f"d2m_l{l}"] = np.minimum(df[f"d2m_l{l}"], df[f"t2m_mean_l{l}"])
    for k_, lo, hi in (("warm_spell_days", 0, 60), ("days_since_extreme", 0, 365), ("extremes_last30", 0, 30)):
        df[k_] = df[k_].clip(lo, hi).round()
    for k_ in ("tp_sum7", "tp_sum30"):
        df[k_] = df[k_].clip(lower=0)
    for k_ in ("swvl1_m14", "swvl1_m30"):
        df[k_] = df[k_].clip(0, 0.8)
    s_ = df.season.values
    thr, sc = thr_sea[s_, df.cell.values], sc_full[df.cell.values, s_]
    df["thr"], df["scale"] = thr, sc
    for l in (0, 1, 2):
        df[f"dist_l{l}"] = df[f"t2m_max_l{l}"] - thr
        df[f"dist_std_l{l}"] = df[f"dist_l{l}"] / sc
    df["dist_trend"] = df["dist_l0"] - df["dist_l1"]
    df["tmax_change"] = df["t2m_max_l0"] - df["t2m_max_l1"]
    return df


def raw_invalid(df):
    """Physics rules broken by a RAW row, before any clean-up."""
    bad = np.zeros(len(df), bool)
    for l in (0, 1, 2):
        mn, me, mx = (df[f"t2m_{v}_l{l}"].values for v in ("min", "mean", "max"))
        bad |= (mn > me + 1e-6) | (me > mx + 1e-6)
        bad |= (df[f"rh_l{l}"].values < 0) | (df[f"rh_l{l}"].values > 100)
        bad |= (df[f"tp_l{l}"].values < 0) | (df[f"ssrd_l{l}"].values < 0)
        bad |= (df[f"swvl1_l{l}"].values < 0) | (df[f"swvl1_l{l}"].values > 0.8)
        bad |= df[f"d2m_l{l}"].values > me + 0.5
    bad |= (df.lat.values < LAT[0] - 0.05) | (df.lat.values > LAT[1] + 0.05)
    bad |= (df.lon.values < LON[0] - 0.05) | (df.lon.values > LON[1] + 0.05)
    return bad


def gpd_ppf(k, u):
    xi, sig, ub = GPD[k]
    z = stats.genpareto.ppf(np.clip(u, 0, 1 - 1e-9), xi, scale=sig)
    return np.minimum(z, min(ub * 0.999, Z_CAP[k]))


out(f"real rows breaking the same rules (should be ~0): {100 * raw_invalid(pos).mean():.2f}%")
save_summary()

# ================================================================ 2. severity judge
section("2. SEVERITY JUDGE (LightGBM: features -> severity z, trained on real extremes)")
JP = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=SEED)
oof = np.zeros(n_pos)
for a, b in KFold(5, shuffle=True, random_state=SEED).split(pos):
    oof[b] = lgb.LGBMRegressor(**JP).fit(pos[FEATS].values[a], pos.z.values[a]).predict(pos[FEATS].values[b])
judge = lgb.LGBMRegressor(**JP).fit(pos[FEATS].values, pos.z.values)
REF = {k: np.sort(oof[pos.season.values == k]) for k in range(len(SNAMES))}
out(f"judge on unseen real extremes: Spearman {stats.spearmanr(pos.z, oof).correlation:.3f}")


def severity(df):
    """Judged severity of each row -> tail probability u in [0,1] (its rank among real extremes of its
    season) -> EVT severity z = GPD quantile at u."""
    pj = judge.predict(df[FEATS].values)
    u = np.zeros(len(df))
    for k in range(len(SNAMES)):
        m = df.season.values == k
        u[m] = np.searchsorted(REF[k], pj[m], side="right") / (len(REF[k]) + 1)
    z = np.zeros(len(df))
    for k in range(len(SNAMES)):
        m = df.season.values == k
        z[m] = gpd_ppf(k, u[m])
    return u, z, pj


EDGES = np.linspace(0, 1, 21)
MASS = np.where(EDGES[:-1] >= TAIL_FROM - 1e-9, TAIL_SHARE / 4, (1 - TAIL_SHARE) / 16)   # 20 bins


def select_tail(df, u, n):
    """Pick n rows whose severity follows the EVT tail-focus target (60% from u >= 0.8), season shares as in
    the real extremes. Returns row indices and the share of tail rows that had to be repeated."""
    chosen, need_tail, rep_tail = [], 0, 0
    seasons = df.season.values
    b_all = np.clip(np.digitize(u, EDGES) - 1, 0, 19)
    for k, share in season_share.items():
        nk = int(round(n * share))
        idx_k = np.where(seasons == k)[0]
        if len(idx_k) == 0:
            idx_k = np.arange(len(df))
        b = b_all[idx_k]
        for j in range(20):
            m = int(round(nk * MASS[j]))
            if m == 0:
                continue
            jj, cand = j, idx_k[b == j]
            while len(cand) == 0 and jj > 0:
                jj -= 1
                cand = idx_k[b == jj]
            if len(cand) == 0:
                cand = idx_k
            if len(cand) >= m:
                take = rng.choice(cand, m, replace=False)
            else:
                take = np.concatenate([cand, rng.choice(cand, m - len(cand), replace=True)])
            if j >= 16:
                need_tail += m
                rep_tail += max(0, m - int((b == j).sum()))
            chosen.append(take)
    return np.concatenate(chosen), rep_tail / max(need_tail, 1)


def finish(name, raw_df, season, req_z=None):
    """Clean up a raw pool, score severities, save <name>_plain and <name>_tail. Returns a stats row."""
    raw_bad = float(raw_invalid(raw_df).mean())
    df = construct(raw_df[BASE_FEATS].values, season, np.zeros(len(raw_df)))
    u, zev, pj = severity(df)
    df["u"], df["z"] = u, zev
    r = dict(generator=name, raw_rule_breaking_pct=round(100 * raw_bad, 2),
             pool_rows=len(df), pool_top20_pct=round(100 * float(np.mean(u >= 0.8)), 1),
             pool_top5_pct=round(100 * float(np.mean(u >= 0.95)), 1))
    if req_z is None:                                   # unconditional: plain = random rows, tail = EVT selection
        plain = df.iloc[rng.choice(len(df), N_OUT, replace=len(df) < N_OUT)].reset_index(drop=True)
        idx, rep = select_tail(df, u, N_OUT)
        tail = df.iloc[idx].reset_index(drop=True)
        r["tail_rows_repeated_pct"] = round(100 * rep, 1)
    else:                                               # conditional: two separately requested halves
        df["z_requested"] = req_z
        h = len(df) // 2
        plain, tail = df.iloc[:h].reset_index(drop=True), df.iloc[h:].reset_index(drop=True)
        r["tail_rows_repeated_pct"] = 0.0
        r["severity_control_spearman"] = round(stats.spearmanr(tail.z_requested, judge.predict(tail[FEATS].values)).correlation, 3)
    r["plain_top20_pct"] = round(100 * float(np.mean(plain.u >= 0.8)), 1)
    r["tail_top20_pct"] = round(100 * float(np.mean(tail.u >= 0.8)), 1)
    drive_ok()
    plain.to_parquet(f"{OUT}/synthetic_{name}_plain.parquet", index=False)
    tail.to_parquet(f"{OUT}/synthetic_{name}_tail.parquet", index=False)
    return r


stats_path = f"{OUT}/generator_stats.csv"
drive_ok()
ST = pd.read_csv(stats_path) if os.path.exists(stats_path) else pd.DataFrame(columns=["generator"])


def done(name):
    drive_ok()
    return name in set(ST.generator) and all(os.path.exists(f"{OUT}/synthetic_{name}_{v}.parquet") for v in ("plain", "tail"))


def record(r, minutes):
    global ST
    r["train_minutes"] = round(minutes, 1)
    ST = pd.concat([ST[ST.generator != r["generator"]], pd.DataFrame([r])], ignore_index=True)
    drive_ok()
    ST.to_csv(stats_path, index=False)
    out("   " + ", ".join(f"{k}={v}" for k, v in r.items() if k != "generator"))
    save_summary()


# ---------------------------------------------------------------- self-check (seconds)
_d = construct(pos[BASE_FEATS].values[:300], pos.season.values[:300], pos.z.values[:300])
_u, _z, _ = severity(_d)
_i, _ = select_tail(_d.assign(u=_u), _u, 200)
assert len(_d) == 300 and np.isfinite(_d[FEATS].values).all() and 0 <= _u.min() and _u.max() <= 1 and len(_i) > 150
out("self-check OK")

# ================================================================ 3. unconditional generators
section("3. TRAIN GENERATORS AND MAKE <gen>_plain / <gen>_tail")
REAL = pos[BASE_FEATS].astype("float64").copy()
REAL["season"] = "s" + pos.season.astype(int).astype(str)


def to_season(col):
    return col.astype(str).str.lstrip("s").astype(int).clip(0, len(SNAMES) - 1).values


def sdv_metadata():
    from sdv.metadata import Metadata
    md = Metadata.detect_from_dataframes({"extremes": REAL}, infer_keys=None)
    for c in BASE_FEATS:
        md.update_column(column_name=c, table_name="extremes", sdtype="numerical")
    md.update_column(column_name="season", table_name="extremes", sdtype="categorical")
    return md


def make_unconditional(name):
    if name == "gaussian_copula":
        from sdv.single_table import GaussianCopulaSynthesizer
        g = GaussianCopulaSynthesizer(sdv_metadata())
        g.fit({"extremes": REAL})
    elif name == "copulagan":
        from sdv.single_table import CopulaGANSynthesizer
        g = CopulaGANSynthesizer(sdv_metadata(), epochs=EPOCHS, batch_size=500, enable_gpu=GPU)
        g.fit({"extremes": REAL})
    elif name == "tvae":
        from ctgan import TVAE
        g = TVAE(epochs=EPOCHS, batch_size=500, enable_gpu=GPU)
        g.fit(REAL, discrete_columns=["season"])
    elif name == "ctgan":
        from ctgan import CTGAN
        g = CTGAN(epochs=EPOCHS, batch_size=500, enable_gpu=GPU)
        g.fit(REAL, discrete_columns=["season"])
    progress(f"{name}: trained, sampling {N_POOL:,} rows")
    s_ = g.sample(N_POOL) if name not in ("gaussian_copula", "copulagan") else g.sample(num_rows=N_POOL)
    if isinstance(s_, dict):                          # SDV 2.x returns {table name: DataFrame}
        s_ = next(iter(s_.values()))
    return s_


for name in [g_ for g_ in GENERATORS if g_ != "cwgan"]:
    if done(name):
        out(f"{name}: already done, skipped")
        continue
    progress(f"training {name}")
    t1 = time.time()
    raw = make_unconditional(name).reset_index(drop=True)
    record(finish(name, raw, to_season(raw["season"])), (time.time() - t1) / 60)
    del raw

# ================================================================ 4. conditional WGAN-GP
if "cwgan" in GENERATORS and not done("cwgan"):
    progress("training cwgan (conditional on season + severity, like the diffusion model)")
    t1 = time.time()
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=SEED, subsample=10**9)
    Xn = qt.fit_transform(pos[BASE_FEATS].values.astype("float64")).astype("float32")
    D_IN, NZ, NS = Xn.shape[1], 64, len(SNAMES)

    def cond(season, z):
        c = np.zeros((len(season), NS + 2), "float32")
        c[np.arange(len(season)), season] = 1
        c[:, NS] = np.asarray(z) / 0.5
        c[:, NS + 1] = 1
        return c

    def mlp(i, o, act):
        return nn.Sequential(nn.Linear(i, 512), act(), nn.Linear(512, 512), act(), nn.Linear(512, 512), act(), nn.Linear(512, o))

    G = mlp(NZ + NS + 2, D_IN, nn.SiLU).to(DEV)
    Dn = mlp(D_IN + NS + 2, 1, lambda: nn.LeakyReLU(0.2)).to(DEV)
    oG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    oD = torch.optim.Adam(Dn.parameters(), lr=1e-4, betas=(0.5, 0.9))
    Xg = torch.tensor(Xn, device=DEV)
    Cg = torch.tensor(cond(pos.season.values.astype(int), pos.z.values), device=DEV)
    B = 1024
    for step in range(WGAN_STEPS):
        for _ in range(5):
            i = torch.randint(0, n_pos, (B,), device=DEV)
            x, c = Xg[i], Cg[i]
            fake = G(torch.cat([torch.randn(B, NZ, device=DEV), c], 1)).detach()
            e = torch.rand(B, 1, device=DEV)
            xh = (e * x + (1 - e) * fake).requires_grad_(True)
            gr = torch.autograd.grad(Dn(torch.cat([xh, c], 1)).sum(), xh, create_graph=True)[0]
            lossD = Dn(torch.cat([fake, c], 1)).mean() - Dn(torch.cat([x, c], 1)).mean() + 10 * ((gr.norm(2, dim=1) - 1) ** 2).mean()
            oD.zero_grad(set_to_none=True)
            lossD.backward()
            oD.step()
        c = Cg[torch.randint(0, n_pos, (B,), device=DEV)]
        lossG = -Dn(torch.cat([G(torch.cat([torch.randn(B, NZ, device=DEV), c], 1)), c], 1)).mean()
        oG.zero_grad(set_to_none=True)
        lossG.backward()
        oG.step()
        if (step + 1) % 1000 == 0:
            progress(f"cwgan step {step + 1}/{WGAN_STEPS}  critic loss {lossD.item():.3f}")

    def request(kind, n):
        """Severities to ask for: 'emp' = copied from real extremes (no EVT), 'tail' = EVT tail focus."""
        if kind == "emp":
            i = rng.integers(0, n_pos, n)
            return pos.season.values[i].astype(int), pos.z.values[i].astype("float32")
        s_ = rng.choice(season_share.index.values, n, p=season_share.values)
        z = np.zeros(n, "float32")
        for k in np.unique(s_):
            m = s_ == k
            u = rng.uniform(0, 1, m.sum())
            hi = rng.random(m.sum()) < 0.5
            u[hi] = rng.uniform(TAIL_FROM, 1, hi.sum())
            z[m] = gpd_ppf(k, u)
        return s_, z

    @torch.no_grad()
    def gen(season, z):
        G.eval()
        outs = []
        for a in range(0, len(season), 50000):
            c = torch.tensor(cond(season[a:a + 50000], z[a:a + 50000]), device=DEV)
            outs.append(G(torch.cat([torch.randn(len(c), NZ, device=DEV), c], 1)).clamp(-6, 6).cpu().numpy())
        return qt.inverse_transform(np.concatenate(outs).astype("float64"))

    s1, z1 = request("emp", N_OUT)
    s2, z2 = request("tail", N_OUT)
    season, req = np.r_[s1, s2], np.r_[z1, z2]
    raw = pd.DataFrame(gen(season, req), columns=BASE_FEATS)
    record(finish("cwgan", raw, season, req_z=req), (time.time() - t1) / 60)
elif "cwgan" in GENERATORS:
    out("cwgan: already done, skipped")

# ================================================================ 5. summary, with Step 7 diffusion for comparison
section("4. HOW THE GENERATORS COMPARE (before any classifier)")
rows = []
for nm in ("ddpm_emp", "ddpm_evt_tail"):
    p = f"{S7}/synthetic_{nm}.parquet"
    drive_ok()
    if os.path.exists(p):
        d = pd.read_parquet(p).sample(50000, random_state=SEED).reset_index(drop=True)
        u, _, pj = severity(d)
        r = dict(generator=f"diffusion ({nm}, Step 7)", raw_rule_breaking_pct=np.nan,
                 top20_pct=round(100 * float(np.mean(u >= 0.8)), 1),
                 severity_control_spearman=round(stats.spearmanr(d.z, pj).correlation, 3))
        rows.append(r)
tab = ST.copy()
out(tab.to_string(index=False))
out("\nStep 7 diffusion sets scored the same way (top20_pct = share of rows judged in the top 20% of the tail):")
out(pd.DataFrame(rows).to_string(index=False))
out("""
How to read this:
  raw_rule_breaking_pct    rows that broke physics BEFORE clean-up (real data ~0)
  pool_top20_pct           share of a generator's own output in the top 20% of the tail (real data: 20)
  pool_top5_pct            same for the top 5% (real data: 5): can it reach the far tail at all?
  tail_rows_repeated_pct   EVT tail set: share of tail rows that had to be repeated because the
                           generator made too few severe days (0 = enough unique severe days)
  plain_top20_pct / tail_top20_pct   share in the top 20% for the two saved sets (tail target: 60)
  severity_control_spearman  conditional generators only: does the weather match the requested severity
                           (real data about 0.78, Step 7 diffusion about 0.67)""")
out(f"\nfiles in {OUT}")
out(f"END  (took {(time.time() - T0) / 60:.1f} min)")
save_summary()
print(f"\nSummary saved to {BASE}/step11_summary.txt")
