# ============================================================================
# STEP 5: TAIL-CONDITIONED, PHYSICS-CONSISTENT CTGAN + CONFIDENCE INTERVALS
# Paste into ONE Colab cell and run on a T4 GPU runtime (Runtime > Change runtime type).
# Time: ~45-60 min. Re-run the same cell after a disconnect: finished parts are skipped.
#
# Why: in Step 4, EVT could only re-weight what the GAN happened to produce. The GAN never
# made the most severe extremes, and 84% of its rows were physically impossible.
#
# What changes:
#  1. TAIL-CONDITIONED GAN. Each real extreme gets a severity bin (5 bins of the fitted GPD:
#     probabilities 50/30/15/4/1 %), combined with its season into one condition. After
#     training, we ASK the GAN for a chosen number of samples in every season x severity bin:
#       cgan_physics   bin counts = as observed (no EVT)                    <- ablation
#       cgan_evt       bin counts = GPD probabilities                       <- EVT
#       cgan_evt_tail  bin counts = GPD probabilities tempered to favour
#                      the severe bins (p^0.5, renormalised)                <- EVT + tail emphasis
#  2. PHYSICS BY CONSTRUCTION. Every generated row is snapped to its nearest real grid cell;
#     distance-to-threshold features are then COMPUTED from that cell's real threshold and
#     variability, Tmin <= Tmean <= Tmax is enforced by sorting, ranges are clipped.
#     Rows are repaired, not thrown away.
#  3. CONFIDENCE INTERVALS for every method (Step 4 + Step 5), resampling whole test months.
#  4. WEIGHTED BASELINE FIXED (milder weight; the Step 4 run stopped after 1 tree).
# Same classifier and settings as Step 4, so all results are comparable.
#
# Reads (never changes): data/model_ready/*, results/step3b/*, results/step4/*
# Writes ONLY to:        results/step5/
# ============================================================================
!pip install -q lightgbm ctgan

import os, time, shutil, warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
import lightgbm as lgb
import torch
from ctgan import CTGAN
from google.colab import drive

warnings.filterwarnings("ignore")
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/himachal_project"
DRIVE_DATA = f"{BASE}/data/model_ready"
DATA = "/content/local_data"
S3B, S4, OUT = f"{BASE}/results/step3b", f"{BASE}/results/step4", f"{BASE}/results/step5"

# ---------------------------------------------------------------- settings (same as Step 4)
TARGET = "y_heat_season"
SEASON_NAMES = ["winter", "pre_monsoon", "monsoon", "post_monsoon"]
TRAIN_FRACTION = 0.5
SYNTH_RATIO = 0.10
GPU = torch.cuda.is_available()
GAN_EPOCHS = 300 if GPU else 80
BIN_PROBS = [0.0, 0.50, 0.80, 0.95, 0.99, 1.0]      # GPD-probability edges of the 5 severity bins
TEMPER = 0.5                                         # cgan_evt_tail: p_bin ** TEMPER
BOOT = 100                                           # bootstrap resamples of test months
SEED = 42
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, max_bin=127,
                  n_estimators=1000, random_state=SEED, n_jobs=-1, verbose=-1)
# ----------------------------------------------------------------
NB = len(BIN_PROBS) - 1
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


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
    src, dst = f"{DRIVE_DATA}/{name}", f"{DATA}/{name}"
    for _ in range(4):
        try:
            drive_ok()
            if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
                shutil.copy(src, dst + ".part")
                os.replace(dst + ".part", dst)
            return
        except OSError as e:
            print(f"   copy of {name} failed ({e}); retrying", flush=True)
            time.sleep(10)
    raise RuntimeError(f"could not copy {name}; re-run the cell")


os.makedirs(DATA, exist_ok=True)
for f in ("train.parquet", "val.parquet", "test.parquet", "thresholds.nc", "cells.csv"):
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
    for p in (f"{OUT}/step5_summary.txt", f"{BASE}/step5_summary.txt"):
        with open(p, "w") as fh:
            fh.write(txt)


def section(t):
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


out(f"STEP 5  {time.strftime('%Y-%m-%d %H:%M')}  target {TARGET}  GPU={GPU}  GAN epochs={GAN_EPOCHS}")

# ================================================================ 1. data (identical to Step 4)
section("1. DATA")
kind, lab = "heat", "season"
thr_col, exc_col = "thr_heat_season_next", "excess_heat_season_next"
scales = pd.read_csv(f"{S3B}/cell_scales.csv")[["cell", "season_next", "heat_scale"]]
cols0 = pd.read_parquet(f"{DATA}/val.parquet").columns
BASE_FEATS = [c for c in cols0 if c not in ("date", "cell", "season_next", "doy_sin", "doy_cos")
              and not c.startswith(("y_", "thr_", "excess_", "tmax_next", "tmin_next"))]
DERIVED = [f"dist_l{l}" for l in (0, 1, 2)] + [f"dist_std_l{l}" for l in (0, 1, 2)] + ["dist_trend", "tmax_change"]
FEATS = BASE_FEATS + DERIVED


def add_derived(df):
    sc = df["heat_scale"]
    for lag in (0, 1, 2):
        df[f"dist_l{lag}"] = df[f"t2m_max_l{lag}"] - df[thr_col]
        df[f"dist_std_l{lag}"] = df[f"dist_l{lag}"] / sc
    df["dist_trend"] = df["dist_l0"] - df["dist_l1"]
    df["tmax_change"] = df["t2m_max_l0"] - df["t2m_max_l1"]
    return df


def load_split(name):
    extra = ["date"] if name != "train" else []
    df = pd.read_parquet(f"{DATA}/{name}.parquet",
                         columns=BASE_FEATS + [TARGET, "cell", "season_next", thr_col, exc_col] + extra)
    if name == "train" and TRAIN_FRACTION < 1:
        df = df.sample(frac=TRAIN_FRACTION, random_state=SEED)       # same rows as Step 4
    df = df.merge(scales, on=["cell", "season_next"], how="left")
    df = add_derived(df)
    df["z"] = (df[exc_col] / df["heat_scale"]).where(df[TARGET] == 1)
    return df.reset_index(drop=True)


tr, va, te = load_split("train"), load_split("val"), load_split("test")
ytr, yva, yte = tr[TARGET].values, va[TARGET].values, te[TARGET].values
n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())
N_SYN = int(SYNTH_RATIO * n_neg) - n_pos
for nm, df in (("train", tr), ("val", va), ("test", te)):
    out(f"{nm:<5} rows {len(df):>9,}  extremes {int(df[TARGET].sum()):>7,}  ({100 * df[TARGET].mean():.2f}%)")
out(f"each generator adds {N_SYN:,} synthetic extreme days")
onset_te = te["dist_l0"].values <= 0
top_te = (yte == 1) & (te["z"].values >= np.nanquantile(te["z"].values, 0.9))
progress("data ready")

# cell tables for snapping synthetic rows to real cells
cells = pd.read_csv(f"{DATA}/cells.csv")
thr_sea = xr.open_dataset(f"{DATA}/thresholds.nc")["thr_heat_season"].values      # (season, cell)
scale_tab = scales.pivot(index="cell", columns="season_next", values="heat_scale").reindex(cells.cell).values
tree = cKDTree(np.c_[cells.lat.values, cells.lon.values])

# ================================================================ 2. severity bins from the fitted GPD
section("2. SEVERITY BINS FROM THE FITTED GPD (standardised excess z)")
fits = pd.read_csv(f"{S3B}/evt_fits_std.csv").set_index("group")
GPD, EDGES = {}, {}
for k, s in enumerate(SEASON_NAMES):
    xi, sig = float(fits.loc[f"heat/{s}", "xi_std"]), float(fits.loc[f"heat/{s}", "sigma_std"])
    ub = sig / -xi if xi < 0 else np.inf
    GPD[k] = (xi, sig, ub)
    e = stats.genpareto.ppf(BIN_PROBS, xi, scale=sig)
    e[0], e[-1] = 0.0, np.inf
    EDGES[k] = e


def zbin(z, k):
    return np.clip(np.searchsorted(EDGES[k], z, side="right") - 1, 0, NB - 1)


pos = tr.loc[ytr == 1].copy()
pos["bin"] = [zbin(z, int(k)) for z, k in zip(pos.z.values, pos.season_next.values)]
season_share = pos.season_next.value_counts(normalize=True).sort_index()
emp = pos.groupby(["season_next", "bin"]).size().unstack(fill_value=0)
emp_p = emp.div(emp.sum(axis=1), axis=0)
gpd_p = np.diff(BIN_PROBS)
temp_p = gpd_p ** TEMPER / (gpd_p ** TEMPER).sum()
tab = emp.copy()
tab.index = [SEASON_NAMES[i] for i in tab.index]
tab.columns = [f"bin{b} ({100 * gpd_p[b]:.0f}%)" for b in tab.columns]
out("real extremes per season x severity bin (GPD share of each bin in brackets):")
out(tab.to_string())
out(f"bin shares used:  observed (per season) | GPD {np.round(gpd_p, 3).tolist()} | "
    f"tempered {np.round(temp_p, 3).tolist()}")
save_summary()

# ================================================================ 3. conditional CTGAN
section("3. TAIL-CONDITIONED CTGAN")
GEN_COLS = BASE_FEATS + ["z"]
real = pos[GEN_COLS].copy()
real["sb"] = [f"{int(k)}_{int(b)}" for k, b in zip(pos.season_next, pos.bin)]
gan_path = f"{OUT}/ctgan_conditional.pkl"
gan = None
if os.path.exists(gan_path):
    try:
        gan = CTGAN.load(gan_path)
    except Exception:
        try:
            gan = torch.load(gan_path, weights_only=False, map_location="cpu")
            gan.set_device(torch.device("cuda" if GPU else "cpu"))
        except Exception:
            gan = None
    if gan is not None:
        out("loaded saved conditional CTGAN")
if gan is None:
    progress(f"training conditional CTGAN on {len(real):,} extremes, {real.sb.nunique()} season x severity conditions")
    gan = CTGAN(epochs=GAN_EPOCHS, batch_size=500, cuda=GPU, verbose=False)
    gan.fit(real, discrete_columns=["sb"])
    drive_ok()
    gan.save(gan_path)
    progress("conditional CTGAN trained and saved")

T_COLS = ["t2m_min", "t2m_mean", "t2m_max"]


def construct(df, k, b):
    """Snap to a real cell, compute derived features exactly, repair physics."""
    df = df.copy().reset_index(drop=True)
    repairs = {}
    _, idx = tree.query(np.c_[df.lat.values, df.lon.values])
    c = cells.iloc[idx]
    df["lat"], df["lon"] = c.lat.values, c.lon.values
    df["elev_m"], df["slope_deg"] = c.elev_m.values, c.slope_deg.values
    df["season_next"] = k
    df[thr_col] = thr_sea[k, c.cell.values]
    df["heat_scale"] = scale_tab[c.cell.values, k]
    for l in (0, 1, 2):
        t = np.sort(df[[f"{v}_l{l}" for v in T_COLS]].values, axis=1)
        repairs[f"t_order_l{l}"] = float(np.mean(np.any(t != df[[f"{v}_l{l}" for v in T_COLS]].values, axis=1)))
        df[[f"{v}_l{l}" for v in T_COLS]] = t
        repairs[f"rh_l{l}"] = float(np.mean(~df[f"rh_l{l}"].between(0, 100)))
        df[f"rh_l{l}"] = df[f"rh_l{l}"].clip(0, 100)
        df[f"tp_l{l}"] = df[f"tp_l{l}"].clip(lower=0)
        df[f"ssrd_l{l}"] = df[f"ssrd_l{l}"].clip(lower=0)
        df[f"swvl1_l{l}"] = df[f"swvl1_l{l}"].clip(0, 0.8)
        df[f"d2m_l{l}"] = np.minimum(df[f"d2m_l{l}"], df[f"t2m_mean_l{l}"])
    lo, hi = EDGES[k][b], EDGES[k][b + 1]
    if np.isfinite(GPD[k][2]):
        hi = min(hi, GPD[k][2])
    repairs["z_outside_bin"] = float(np.mean((df.z < lo) | (df.z >= hi)))
    df["z"] = df["z"].clip(lo + 1e-6, hi - 1e-6 if np.isfinite(hi) else None)
    return add_derived(df), repairs


def bin_shares(name, k):
    if name == "cgan_physics":
        return emp_p.loc[k].reindex(range(NB), fill_value=0).values
    if name == "cgan_evt":
        return gpd_p
    return temp_p


syn, repair_log = {}, []
for name in ("cgan_physics", "cgan_evt", "cgan_evt_tail"):
    path = f"{OUT}/synthetic_{name}.parquet"
    if os.path.exists(path):
        syn[name] = pd.read_parquet(path)
        continue
    parts = []
    for k in range(len(SEASON_NAMES)):
        p = bin_shares(name, k)
        for b in range(NB):
            n = int(round(N_SYN * season_share.get(k, 0) * p[b]))
            if n == 0 or f"{k}_{b}" not in set(real.sb):
                continue
            s_ = gan.sample(n, condition_column="sb", condition_value=f"{k}_{b}")
            s_, rep = construct(s_, k, b)
            rep.update(name=name, season=SEASON_NAMES[k], bin=b, n=n)
            repair_log.append(rep)
            parts.append(s_)
    syn[name] = pd.concat(parts, ignore_index=True)
    drive_ok()
    syn[name].to_parquet(path, index=False)
    progress(f"{name}: {len(syn[name]):,} rows")
if repair_log:
    rl = pd.DataFrame(repair_log)
    rl.to_csv(f"{OUT}/repairs.csv", index=False)
    agg = rl.groupby("name").apply(lambda g: pd.Series(
        {c: np.average(g[c], weights=g.n) for c in rl.columns if c not in ("name", "season", "bin", "n")}))
    out("\nshare of generated rows that needed each repair (weighted by rows):")
    out(agg.round(3).to_string())
save_summary()

# ================================================================ 4. tail consistency
section("4. TAIL OF THE SYNTHETIC EXTREMES (z), PER SEASON")
s4 = {}
for n in ("ctgan_vanilla", "ctgan_evt"):
    p = f"{S4}/synthetic_{n}.parquet"
    if os.path.exists(p):
        s4[n] = pd.read_parquet(p, columns=["season_next", "z"])
rows = []
for k, s in enumerate(SEASON_NAMES):
    xi_t, sig_t, ub = GPD[k]
    zr = pos.loc[pos.season_next == k, "z"].values
    srcs = [("real", pos)] + [(f"step4_{n}", d) for n, d in s4.items()] + list(syn.items())
    for src, d in srcs:
        z = d.loc[d.season_next == k, "z"].values
        z = z[z > 0]
        if len(z) < 50:
            continue
        xi = stats.genpareto.fit(z, floc=0)[0]
        rows.append(dict(season=s, source=src, n=len(z), xi=round(xi, 3),
                         share_top_bin=round(float(np.mean(z >= EDGES[k][-2])), 4),
                         z_q99=round(np.quantile(z, .99), 2), z_max=round(z.max(), 2),
                         n_beyond_real_q99=int(np.sum(z > np.quantile(zr, .99))),
                         ks_to_gpd=round(stats.kstest(z, "genpareto", args=(xi_t, 0, sig_t)).statistic, 3)))
tail = pd.DataFrame(rows)
tail.to_csv(f"{OUT}/tail_consistency.csv", index=False)
out(tail.to_string(index=False))
out(f"share_top_bin: share in the most severe bin (GPD says {gpd_p[-1]:.0%}; tempered asks for {temp_p[-1]:.0%})")

fig, axs = plt.subplots(1, 4, figsize=(17, 4))
for k, (ax, s) in enumerate(zip(axs, SEASON_NAMES)):
    xi_t, sig_t, ub = GPD[k]
    for src, d, c in [("real", pos, "k"), ("step4 vanilla", s4.get("ctgan_vanilla"), "C1"),
                      ("cgan_evt", syn["cgan_evt"], "C0"), ("cgan_evt_tail", syn["cgan_evt_tail"], "C2")]:
        if d is None:
            continue
        z = np.sort(d.loc[(d.season_next == k) & (d.z > 0), "z"].values)
        if len(z) < 20:
            continue
        ax.semilogy(z, 1 - (np.arange(1, len(z) + 1) - 0.5) / len(z), color=c, lw=1.2, label=src)
    zz = np.linspace(0, min(ub, 4) if np.isfinite(ub) else 4, 200)
    ax.semilogy(zz, stats.genpareto.sf(zz, xi_t, scale=sig_t), "k--", lw=0.8, label="fitted GPD")
    ax.set_title(s); ax.set_xlabel("severity z"); ax.set_ylim(1e-4, 1)
axs[0].set_ylabel("share of extremes more severe"); axs[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/tail_survival.png", dpi=120); plt.close()
save_summary()

# ================================================================ 5. classifiers
section("5. SAME CLASSIFIER AS STEP 4, NEW TRAINING DATA")
res_path, pred_path, vpred_path = f"{OUT}/results.csv", f"{OUT}/test_predictions.parquet", f"{OUT}/val_predictions.parquet"
done = pd.read_csv(res_path) if os.path.exists(res_path) else pd.DataFrame()
preds = pd.read_parquet(pred_path) if os.path.exists(pred_path) else pd.DataFrame(index=range(len(te)))
vpreds = pd.read_parquet(vpred_path) if os.path.exists(vpred_path) else pd.DataFrame(index=range(len(va)))
MONO = [1 if f in ("dist_l0", "dist_std_l0") else 0 for f in FEATS]
Xtr = tr[FEATS].values.astype("float32")
Xva, Xte = va[FEATS].values.astype("float32"), te[FEATS].values.astype("float32")


def pick_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return th[int(np.nanargmax(f1[:-1]))]


def evaluate(name, p_va, p_te, note):
    thr = pick_threshold(yva, p_va)
    yhat = p_te >= thr
    tp = int((yhat & (yte == 1)).sum()); fp = int((yhat & (yte == 0)).sum()); fn = int((~yhat & (yte == 1)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(model=name, val_pr_auc=round(average_precision_score(yva, p_va), 4),
                test_pr_auc=round(average_precision_score(yte, p_te), 4),
                precision=round(prec, 3), recall=round(rec, 3),
                f1=round(2 * prec * rec / max(prec + rec, 1e-12), 3), csi=round(tp / max(tp + fp + fn, 1), 3),
                onset_pr_auc=round(average_precision_score(yte[onset_te], p_te[onset_te]), 4),
                onset_recall=round(float(yhat[onset_te & (yte == 1)].mean()), 3),
                top10_recall=round(float(yhat[top_te].mean()), 3), note=note)


def save_result(r, p_va, p_te):
    global done, preds, vpreds
    drive_ok()
    done = pd.concat([done[done.model != r["model"]] if len(done) else done, pd.DataFrame([r])], ignore_index=True)
    done.to_csv(res_path, index=False)
    preds[r["model"]] = p_te.astype("float32"); preds.to_parquet(pred_path, index=False)
    vpreds[r["model"]] = p_va.astype("float32"); vpreds.to_parquet(vpred_path, index=False)


def fit_eval(name, X, y, note="", **extra):
    if len(done) and name in set(done.model):
        out(f"   {name}: already done, skipped")
        return
    progress(f"training {name} on {len(y):,} rows")
    m = lgb.LGBMClassifier(**LGB_PARAMS, monotone_constraints=MONO, **extra)
    m.fit(X, y, eval_set=[(Xva, yva)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(60, verbose=False)])
    pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
    r = evaluate(name, pv, pt, f"{note}{m.best_iteration_} trees")
    save_result(r, pv, pt)
    progress(f"{name}: test PR-AUC {r['test_pr_auc']}")


w = float(np.sqrt(n_neg / n_pos))
fit_eval("weighted_fixed", Xtr, ytr, note=f"weight {w:.0f}, ", scale_pos_weight=w)
for name in ("cgan_physics", "cgan_evt", "cgan_evt_tail"):
    Xg = syn[name][FEATS].values.astype("float32")
    fit_eval(name, np.vstack([Xtr, Xg]), np.concatenate([ytr, np.ones(len(Xg), dtype=ytr.dtype)]),
             note=f"+{len(Xg):,} synthetic, ")

# bring in Step 4 results (same test rows, same classifier settings)
s4res = pd.read_csv(f"{S4}/results.csv") if os.path.exists(f"{S4}/results.csv") else pd.DataFrame()
s4pred = pd.read_parquet(f"{S4}/test_predictions.parquet") if os.path.exists(f"{S4}/test_predictions.parquet") else pd.DataFrame()
keep4 = [m for m in ("persistence", "plain", "smote", "ctgan_vanilla", "ctgan_physical", "ctgan_evt") if m in s4pred.columns]
allres = pd.concat([s4res[s4res.model.isin(keep4)].assign(step=4), done.assign(step=5)], ignore_index=True)
allpred = pd.concat([s4pred[keep4].rename(columns={m: m for m in keep4}), preds], axis=1)
out("\nTest 2016-2025, all methods (Step 4 rows reused; decision threshold picked on 2011-2015):")
cols = ["step", "model", "test_pr_auc", "precision", "recall", "f1", "csi", "onset_pr_auc", "onset_recall", "top10_recall", "note"]
out(allres[[c for c in cols if c in allres.columns]].to_string(index=False))
save_summary()

# ================================================================ 6. confidence intervals
section(f"6. 95% CONFIDENCE INTERVALS ({BOOT} resamples of whole test months)")
month = pd.to_datetime(te["date"]).dt.to_period("M").astype(str).values
blocks = pd.Series(np.arange(len(te))).groupby(month).apply(np.array)
bnames = blocks.index.values
models = list(allpred.columns)
P = {m: allpred[m].values for m in models}


def fast_ap(y, s):
    o = np.argsort(-s, kind="stable")
    y = y[o]
    tp = np.cumsum(y)
    npos = tp[-1]
    if npos == 0:
        return np.nan
    prec = tp / np.arange(1, len(y) + 1)
    return float(np.sum(prec[y == 1]) / npos)


ap_b = {m: [] for m in models}
on_b = {m: [] for m in models}
for i in range(BOOT):
    idx = np.concatenate(blocks.loc[rng.choice(bnames, len(bnames), replace=True)].values)
    y = yte[idx]
    on = onset_te[idx]
    for m in models:
        s = P[m][idx]
        ap_b[m].append(fast_ap(y, s))
        on_b[m].append(fast_ap(y[on], s[on]))
    if (i + 1) % 25 == 0:
        progress(f"bootstrap {i + 1}/{BOOT}")
ci = pd.DataFrame({m: dict(pr_auc=round(fast_ap(yte, P[m]), 4),
                           pr_auc_lo=round(np.nanpercentile(ap_b[m], 2.5), 4),
                           pr_auc_hi=round(np.nanpercentile(ap_b[m], 97.5), 4),
                           onset_pr_auc=round(fast_ap(yte[onset_te], P[m][onset_te]), 4),
                           onset_lo=round(np.nanpercentile(on_b[m], 2.5), 4),
                           onset_hi=round(np.nanpercentile(on_b[m], 97.5), 4)) for m in models}).T
ci.to_csv(f"{OUT}/confidence_intervals.csv")
out(ci.to_string())

pairs = []
for a in [m for m in ("cgan_evt", "cgan_evt_tail", "cgan_physics", "ctgan_evt") if m in models]:
    for b in [m for m in ("persistence", "plain", "smote", "ctgan_vanilla", "cgan_physics") if m in models and m != a]:
        d = np.array(ap_b[a]) - np.array(ap_b[b])
        do = np.array(on_b[a]) - np.array(on_b[b])
        pairs.append(dict(method=a, vs=b,
                          pr_auc_diff=round(float(np.nanmean(d)), 4),
                          diff_ci=f"[{np.nanpercentile(d, 2.5):+.4f}, {np.nanpercentile(d, 97.5):+.4f}]",
                          share_better=round(float(np.nanmean(d > 0)), 2),
                          onset_diff=round(float(np.nanmean(do)), 4),
                          onset_share_better=round(float(np.nanmean(do > 0)), 2)))
pairs = pd.DataFrame(pairs)
pairs.to_csv(f"{OUT}/paired_differences.csv", index=False)
out("\nPaired differences (same resampled months for both methods):")
out(pairs.to_string(index=False))
out("""
A difference is convincing when its interval does not include 0 (share_better near 1.0 or 0.0).
onset = days where today is not yet past tomorrow's threshold.""")

plt.figure(figsize=(6.5, 5))
for m in models:
    pr, rc, _ = precision_recall_curve(yte, P[m])
    plt.plot(rc, pr, lw=2 if m.startswith("cgan_evt") else 1, label=f"{m} ({ci.loc[m, 'pr_auc']:.3f})")
plt.axhline(yte.mean(), color="grey", ls="--", lw=0.8, label=f"random ({yte.mean():.3f})")
plt.xlabel("recall"); plt.ylabel("precision"); plt.title(f"{TARGET}, test 2016-2025")
plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{OUT}/pr_curves.png", dpi=120); plt.close()

out(f"\nfiles in {OUT}")
out(f"END  (took {(time.time() - T0) / 60:.1f} min)")
save_summary()
print(f"\nSummary saved to {BASE}/step5_summary.txt")
