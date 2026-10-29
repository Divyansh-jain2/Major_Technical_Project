# ============================================================================
# STEP 7: EVT-GUIDED TABULAR DIFFUSION (+ CTGAN baseline) ON DATASET v3
# Paste into ONE Colab cell. Runtime > Change runtime type > T4 GPU. ~25-35 min.
# Re-run the same cell after a disconnect: finished parts are skipped.
#
# The generator learns what the days before an extreme look like, CONDITIONED ON
#   season and severity z  (z = how far past the threshold tomorrow goes, in units of
#   the cell's normal variability). Because z is an INPUT, we choose it, so EVT can
#   decide exactly which severities are generated (the step CTGAN could not do).
#
# Model: a compact denoising diffusion model for tables (TabDDPM idea):
#   features -> quantile-normal transform -> Gaussian diffusion (1000 steps, cosine)
#   denoiser = residual MLP, input + time + condition embeddings; condition dropped 10%
#   of the time so classifier-free guidance can sharpen adherence to the requested z.
# Synthetic sets (same trained model, only the requested severities differ):
#   ddpm_emp       (season, z) resampled from the real extremes        <- no EVT
#   ddpm_evt       season as observed, z ~ fitted GPD of that season   <- EVT (can go beyond
#                  the largest observed extreme, up to the tail's physical bound)
#   ddpm_evt_tail  half the z from the GPD, half from its upper 20%    <- EVT + tail emphasis
#   ctgan          CTGAN on the same data (GAN baseline), severity not controlled
# Every row: snapped to a real grid cell, physics repaired, distance-to-threshold features
# computed exactly from that cell's threshold. A "control check" tests whether generated
# rows really look as severe as requested.
#
# Reads (never changes): data/model_ready_v3/*, results/step3b/evt_fits_std.csv
# Writes ONLY to:        results/step7/
# ============================================================================
!pip install -q lightgbm ctgan

import os, json, math, time, shutil, warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.preprocessing import QuantileTransformer
import lightgbm as lgb
import torch
import torch.nn as nn
from google.colab import drive

warnings.filterwarnings("ignore")
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/himachal_project"
V3 = f"{BASE}/data/model_ready_v3"
OUT = f"{BASE}/results/step7"
LOCAL = "/content/local_v3"

# ---------------------------------------------------------------- settings
SYNTH_RATIO = 0.10            # each generator adds extremes up to 10% of normal days
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 15000 if DEV == "cuda" else 2500      # diffusion training steps
BATCH = 1024
HIDDEN = 512
T_DIFF = 1000
SAMPLE_STEPS = 50             # DDIM steps when sampling
GUIDANCE = 2.0                # classifier-free guidance weight
P_DROP = 0.10                 # condition dropout during training
RUN_CTGAN = True
CTGAN_EPOCHS = 300 if DEV == "cuda" else 60
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


def copy_local(name):
    src, dst = f"{V3}/{name}", f"{LOCAL}/{name}"
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
    raise RuntimeError(f"could not copy {name}; re-run the cell")


os.makedirs(LOCAL, exist_ok=True)
for f in ("train.parquet", "features.json", "thresholds.nc", "cells.csv"):
    copy_local(f)
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
    for p in (f"{OUT}/step7_summary.txt", f"{BASE}/step7_summary.txt"):
        open(p, "w").write(txt)


def section(t):
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


out(f"STEP 7  {time.strftime('%Y-%m-%d %H:%M')}  device={DEV}  diffusion steps={STEPS}")

# ================================================================ 1. data
section("1. REAL EXTREMES (training years, random half)")
meta = json.load(open(f"{LOCAL}/features.json"))
BASE_FEATS, DERIVED, FEATS, SNAMES = meta["BASE_FEATS"], meta["DERIVED"], meta["FEATS"], meta["SEASONS"]
tr = pd.read_parquet(f"{LOCAL}/train.parquet")
pos = tr[tr.y == 1].reset_index(drop=True)
n_pos, n_neg = len(pos), int((tr.y == 0).sum())
N_SYN = int(SYNTH_RATIO * n_neg) - n_pos
season_share = pos.season.value_counts(normalize=True).sort_index()
out(f"{n_pos:,} real extremes, {n_neg:,} normal days -> each synthetic set has {N_SYN:,} rows")
out("extremes per season: " + ", ".join(f"{SNAMES[k]} {int(v * n_pos):,}" for k, v in season_share.items()))
del tr

cells = pd.read_csv(f"{LOCAL}/cells.csv")
thr_sea = xr.open_dataset(f"{LOCAL}/thresholds.nc")["thr_heat_season"].values
scale_tab = pos.groupby(["cell", "season"])["scale"].first()
sc_full = pd.read_csv(f"{BASE}/results/step3b/cell_scales.csv").pivot(
    index="cell", columns="season_next", values="heat_scale").reindex(cells.cell).values      # (cell, season)
tree = cKDTree(np.c_[cells.lat.values, cells.lon.values])

fits = pd.read_csv(f"{BASE}/results/step3b/evt_fits_std.csv").set_index("group")
GPD = {}
for k, s in enumerate(SNAMES):
    xi, sig = float(fits.loc[f"heat/{s}", "xi_std"]), float(fits.loc[f"heat/{s}", "sigma_std"])
    GPD[k] = (xi, sig, sig / -xi if xi < 0 else np.inf)
    zr = pos.z[pos.season == k]
    out(f"   {s:<13} GPD xi {xi:+.3f} sigma {sig:.3f} bound {GPD[k][2]:.2f} | real z max {zr.max():.2f}")
save_summary()

# ================================================================ 2. diffusion model
section("2. TRAIN THE CONDITIONAL DIFFUSION MODEL")
qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=SEED, subsample=10**9)
Xn = qt.fit_transform(pos[BASE_FEATS].values.astype("float64")).astype("float32")
D_IN = Xn.shape[1]
Z_SCALE = 0.5


def cond_matrix(season, z, keep):
    c = np.zeros((len(season), len(SNAMES) + 2), "float32")
    c[np.arange(len(season)), season] = 1
    c[:, len(SNAMES)] = z / Z_SCALE
    c[:, len(SNAMES) + 1] = 1
    return c * keep[:, None]


def tstep_emb(t, dim=128):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class Block(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.n = nn.LayerNorm(h)
        self.f = nn.Sequential(nn.Linear(h, h * 2), nn.SiLU(), nn.Linear(h * 2, h))

    def forward(self, x):
        return x + self.f(self.n(x))


class Denoiser(nn.Module):
    def __init__(self, d, c, h=HIDDEN, nblocks=4):
        super().__init__()
        self.inp = nn.Linear(d, h)
        self.temb = nn.Sequential(nn.Linear(128, h), nn.SiLU(), nn.Linear(h, h))
        self.cemb = nn.Sequential(nn.Linear(c, h), nn.SiLU(), nn.Linear(h, h))
        self.blocks = nn.Sequential(*[Block(h) for _ in range(nblocks)])
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, d))

    def forward(self, x, t, c):
        return self.out(self.blocks(self.inp(x) + self.temb(tstep_emb(t)) + self.cemb(c)))


s_ = torch.linspace(0, 1, T_DIFF + 1)
abar = torch.cos((s_ + 0.008) / 1.008 * math.pi / 2) ** 2
abar = (abar / abar[0]).clamp(1e-5, 1)[1:].to(DEV)                 # alpha-bar for t = 1..T
model_path = f"{OUT}/diffusion.pt"
net = Denoiser(D_IN, len(SNAMES) + 2).to(DEV)
ema = Denoiser(D_IN, len(SNAMES) + 2).to(DEV)
if os.path.exists(model_path):
    st = torch.load(model_path, map_location=DEV)
    ema.load_state_dict(st["ema"])
    out(f"loaded trained diffusion model ({st['steps']} steps, final loss {st['loss']:.4f})")
else:
    ema.load_state_dict(net.state_dict())
    X_t = torch.tensor(Xn, device=DEV)
    S_t = torch.tensor(pos.season.values.astype("int64"), device=DEV)
    Z_t = torch.tensor(pos.z.values.astype("float32"), device=DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    losses = []
    for step in range(1, STEPS + 1):
        i = torch.randint(0, len(X_t), (BATCH,), device=DEV)
        x0 = X_t[i]
        keep = (torch.rand(BATCH, device=DEV) > P_DROP).float()
        c = torch.zeros(BATCH, len(SNAMES) + 2, device=DEV)
        c[torch.arange(BATCH), S_t[i]] = 1
        c[:, len(SNAMES)] = Z_t[i] / Z_SCALE
        c[:, len(SNAMES) + 1] = 1
        c = c * keep[:, None]
        t = torch.randint(0, T_DIFF, (BATCH,), device=DEV)
        eps = torch.randn_like(x0)
        ab = abar[t][:, None]
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        loss = ((net(xt, t, c) - eps) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(0.999).add_(pn, alpha=0.001)
        losses.append(loss.item())
        if step % 2500 == 0:
            progress(f"step {step}/{STEPS}  loss {np.mean(losses[-500:]):.4f}")
    drive_ok()
    torch.save({"ema": ema.state_dict(), "steps": STEPS, "loss": float(np.mean(losses[-500:]))}, model_path)
    out(f"trained: final loss {np.mean(losses[-500:]):.4f}")
ema.eval()


@torch.no_grad()
def sample(season, z, batch=20000):
    """DDIM sampling with classifier-free guidance."""
    outs = []
    ts = np.linspace(T_DIFF - 1, 0, SAMPLE_STEPS).round().astype(int)
    for a in range(0, len(season), batch):
        sb, zb = season[a:a + batch], z[a:a + batch]
        n = len(sb)
        c = torch.tensor(cond_matrix(sb, zb, np.ones(n, "float32")), device=DEV)
        c0 = torch.zeros_like(c)
        x = torch.randn(n, D_IN, device=DEV)
        for j, t in enumerate(ts):
            tt = torch.full((n,), int(t), device=DEV, dtype=torch.long)
            e = (1 + GUIDANCE) * ema(x, tt, c) - GUIDANCE * ema(x, tt, c0)
            ab = abar[t]
            x0 = ((x - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-6, 6)
            ab_prev = abar[ts[j + 1]] if j + 1 < len(ts) else torch.tensor(1.0, device=DEV)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * e
        outs.append(x.cpu().numpy())
    return qt.inverse_transform(np.concatenate(outs).astype("float64"))


T_SETS = [("t2m_min", "t2m_mean", "t2m_max")]


def construct(X, season, z):
    """Snap to a real cell, repair physics, compute distance features exactly."""
    df = pd.DataFrame(X, columns=BASE_FEATS)
    _, idx = tree.query(np.c_[df.lat.values, df.lon.values])
    c = cells.iloc[idx]
    df["lat"], df["lon"], df["elev_m"], df["slope_deg"] = c.lat.values, c.lon.values, c.elev_m.values, c.slope_deg.values
    df["cell"] = c.cell.values
    df["season"] = season
    df["z"] = z
    rep = {}
    for l in (0, 1, 2):
        cols = [f"t2m_min_l{l}", f"t2m_mean_l{l}", f"t2m_max_l{l}"]
        t = np.sort(df[cols].values, axis=1)
        rep[f"temp_order_l{l}"] = float(np.mean(np.any(t != df[cols].values, axis=1)))
        df[cols] = t
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
    thr = thr_sea[season, df.cell.values]
    sc = sc_full[df.cell.values, season]
    df["thr"], df["scale"] = thr, sc
    for l in (0, 1, 2):
        df[f"dist_l{l}"] = df[f"t2m_max_l{l}"] - thr
        df[f"dist_std_l{l}"] = df[f"dist_l{l}"] / sc
    df["dist_trend"] = df["dist_l0"] - df["dist_l1"]
    df["tmax_change"] = df["t2m_max_l0"] - df["t2m_max_l1"]
    return df, rep


def gpd_draw(k, n, lower_u=0.0):
    xi, sig, ub = GPD[k]
    u = rng.uniform(lower_u, 1.0, n)
    z = stats.genpareto.ppf(np.minimum(u, 1 - 1e-9), xi, scale=sig)
    return np.minimum(z, ub * 0.999) if np.isfinite(ub) else z


def request(name):
    """Season and severity requested for each synthetic row."""
    if name == "ddpm_emp":
        i = rng.integers(0, n_pos, N_SYN)
        return pos.season.values[i].astype(int), pos.z.values[i].astype("float32")
    seasons = rng.choice(season_share.index.values, N_SYN, p=season_share.values)
    z = np.zeros(N_SYN, "float32")
    for k in np.unique(seasons):
        m = seasons == k
        if name == "ddpm_evt":
            z[m] = gpd_draw(k, m.sum())
        else:                                          # ddpm_evt_tail
            half = rng.random(m.sum()) < 0.5
            zz = gpd_draw(k, m.sum())
            zz[half] = gpd_draw(k, half.sum(), lower_u=0.8)
            z[m] = zz
    return seasons, z


section("3. GENERATE SYNTHETIC EXTREMES")
syn, reps = {}, []
for name in ("ddpm_emp", "ddpm_evt", "ddpm_evt_tail"):
    path = f"{OUT}/synthetic_{name}.parquet"
    if os.path.exists(path):
        syn[name] = pd.read_parquet(path)
        out(f"{name}: loaded {len(syn[name]):,} rows")
        continue
    season, z = request(name)
    X = sample(season, z)
    df, rep = construct(X, season, z)
    rep["name"] = name
    reps.append(rep)
    drive_ok()
    df.to_parquet(path, index=False)
    syn[name] = df
    progress(f"{name}: {len(df):,} rows")

if RUN_CTGAN:
    path = f"{OUT}/synthetic_ctgan.parquet"
    if os.path.exists(path):
        syn["ctgan"] = pd.read_parquet(path)
        out(f"ctgan: loaded {len(syn['ctgan']):,} rows")
    else:
        from ctgan import CTGAN
        progress("training CTGAN baseline")
        real = pos[BASE_FEATS + ["z"]].copy()
        real["season"] = pos.season.astype(str)
        g = CTGAN(epochs=CTGAN_EPOCHS, batch_size=500, cuda=(DEV == "cuda"), verbose=False)
        g.fit(real, discrete_columns=["season"])
        s_ = g.sample(N_SYN)
        season = s_["season"].astype(int).values
        df, rep = construct(s_[BASE_FEATS].values, season, s_["z"].clip(lower=1e-4).values.astype("float32"))
        rep["name"] = "ctgan"
        reps.append(rep)
        drive_ok()
        df.to_parquet(path, index=False)
        syn["ctgan"] = df
        progress(f"ctgan: {len(df):,} rows")
if reps:
    out("\nshare of rows whose temperatures had to be re-ordered (Tmin<=Tmean<=Tmax):")
    out(pd.DataFrame(reps).set_index("name").round(3).to_string())
save_summary()

# ================================================================ 4. control check
section("4. CONTROL CHECK: do generated rows look as severe as requested?")
# a severity "judge" trained on real extremes: features -> z
idx = rng.permutation(n_pos)
cut = int(0.8 * n_pos)
judge = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=SEED)
judge.fit(pos[FEATS].values[idx[:cut]], pos.z.values[idx[:cut]])
held = pos.iloc[idx[cut:]]
rows = [dict(source="real (held out)", n=len(held),
             spearman=round(stats.spearmanr(held.z, judge.predict(held[FEATS].values)).correlation, 3),
             mean_z_requested=round(held.z.mean(), 3), mean_z_judged=round(judge.predict(held[FEATS].values).mean(), 3),
             top10pct_requested=round(np.quantile(held.z, .9), 3),
             top10pct_judged=round(np.quantile(judge.predict(held[FEATS].values), .9), 3))]
for name, df in syn.items():
    sub = df.sample(min(len(df), 50000), random_state=SEED)
    pj = judge.predict(sub[FEATS].values)
    rows.append(dict(source=name, n=len(df),
                     spearman=round(stats.spearmanr(sub.z, pj).correlation, 3),
                     mean_z_requested=round(sub.z.mean(), 3), mean_z_judged=round(pj.mean(), 3),
                     top10pct_requested=round(np.quantile(sub.z, .9), 3), top10pct_judged=round(np.quantile(pj, .9), 3)))
ctrl = pd.DataFrame(rows)
ctrl.to_csv(f"{OUT}/control_check.csv", index=False)
out(ctrl.to_string(index=False))
out("""spearman: how well the judge's severity estimate follows the requested severity. For real data this
is the best achievable; a generator that obeys its severity input should come close to it.""")

# severity of requested z vs GPD, per season
rows = []
for k, s in enumerate(SNAMES):
    xi, sig, ub = GPD[k]
    for src, d in [("real", pos)] + list(syn.items()):
        z = d.loc[d.season == k, "z"].values
        z = z[z > 0]
        if len(z) < 50:
            continue
        rows.append(dict(season=s, source=src, n=len(z), z_q99=round(np.quantile(z, .99), 2), z_max=round(z.max(), 2),
                         share_beyond_real_max=round(float(np.mean(z > pos.z[pos.season == k].max())), 4),
                         ks_to_gpd=round(stats.kstest(z, "genpareto", args=(xi, 0, sig)).statistic, 3)))
sev = pd.DataFrame(rows)
sev.to_csv(f"{OUT}/severity_requested.csv", index=False)
out("\nseverity of the synthetic extremes (z) vs the fitted GPD:")
out(sev.to_string(index=False))
out(f"\nfiles in {OUT}")
out(f"END  (took {(time.time() - T0) / 60:.1f} min)")
save_summary()
print(f"\nSummary saved to {BASE}/step7_summary.txt")
