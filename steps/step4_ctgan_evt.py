# ============================================================================
# STEP 4: VANILLA CTGAN vs EVT-GUIDED CTGAN vs CLASSIC REBALANCING
# Paste into ONE Colab cell and run.
# BEFORE RUNNING: Runtime > Change runtime type > T4 GPU (free). Works on CPU too,
#   just slower (fewer GAN epochs are used automatically).
# The data is first copied from Drive to Colab's local disk (a few minutes), and Drive
#   is re-mounted automatically if its connection drops.
# Time: ~30-45 min on a T4 runtime. Every finished part is saved to Drive, so if
#   Colab disconnects, run the cell again: finished parts are skipped.
#
# Reads (never changes): data/model_ready/*.parquet, results/step3b/{cell_scales,evt_fits_std}.csv
# Writes ONLY to:        results/step4/
#
# THE EXPERIMENT. Same classifier (LightGBM, settings fixed below) every time; only the
# extra extreme-day examples it gets for training change:
#   plain          no rebalancing
#   weighted       extreme days weighted up
#   smote          SMOTE-made extreme days                          (classic, no theory)
#   ctgan_vanilla  CTGAN-made extreme days, used as they come       (deep generative)
#   ctgan_physical CTGAN + physical-consistency filter               (ablation)
#   ctgan_evt      CTGAN + filter + EVT: synthetic extremes resampled so their severity
#                  follows the fitted GPD tail of their season, and nothing beyond the
#                  tail's upper bound                                 (THE PROPOSED METHOD)
# All generators use the SAME trained CTGAN, so differences come from EVT, not the GAN.
#
# Severity z = (tomorrow's Tmax - threshold) / cell's variability. The GAN generates z
# alongside the features so EVT can steer it; z is NEVER given to the classifier.
# ============================================================================
!pip install -q lightgbm imbalanced-learn ctgan

import os, time, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import torch
from ctgan import CTGAN
from google.colab import drive

warnings.filterwarnings("ignore")
import shutil
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/himachal_project"
DRIVE_DATA = f"{BASE}/data/model_ready"
DATA = "/content/local_data"          # local copy: faster, and immune to Drive dropping out
S3B = f"{BASE}/results/step3b"
OUT = f"{BASE}/results/step4"


def drive_ok():
    """Re-mount Google Drive if the connection dropped ('Transport endpoint is not connected')."""
    for attempt in range(3):
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
    for attempt in range(4):
        try:
            drive_ok()
            if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
                shutil.copy(src, dst + ".part")
                os.replace(dst + ".part", dst)
            return
        except OSError as e:
            print(f"   copy of {name} failed ({e}); retrying", flush=True)
            time.sleep(10)
    raise RuntimeError(f"could not copy {name} from Drive; re-run the cell")


os.makedirs(DATA, exist_ok=True)
for f in ("train.parquet", "val.parquet", "test.parquet"):
    copy_local(f)
    print(f"   local copy ready: {f}", flush=True)

# ---------------------------------------------------------------- settings
TARGET = "y_heat_season"
SEASON_NAMES = ["winter", "pre_monsoon", "monsoon", "post_monsoon"]
TRAIN_FRACTION = 0.5          # random half of 1981-2010 rows (keeps the ~1% imbalance); 1.0 = all
SYNTH_RATIO = 0.10            # every generator adds extremes up to 10% of normal days (same as SMOTE)
GPU = torch.cuda.is_available()
GAN_EPOCHS = 300 if GPU else 80
EVT_POOL = 3                  # EVT version samples 3x more, then keeps the tail-consistent ones
SEED = 42
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, max_bin=127,
                  n_estimators=1000, random_state=SEED, n_jobs=-1, verbose=-1)
# ----------------------------------------------------------------

drive_ok()
os.makedirs(OUT, exist_ok=True)
LINES, T0 = [], time.time()
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def progress(msg):
    print(f"   ... [{(time.time() - T0) / 60:4.1f} min] {msg}", flush=True)


def save_summary():
    drive_ok()
    txt = "\n".join(LINES) + "\n"
    for p in (f"{OUT}/step4_summary.txt", f"{BASE}/step4_summary.txt"):
        with open(p, "w") as fh:
            fh.write(txt)


def section(t):
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


out(f"STEP 4  {time.strftime('%Y-%m-%d %H:%M')}  target {TARGET}  GPU={GPU}  GAN epochs={GAN_EPOCHS}")

# ================================================================ 1. data
section("1. DATA (same features as Step 3b, calendar features removed)")
kind = "heat" if "heat" in TARGET else "cold"
lab = TARGET.split("_", 2)[2]
thr_col, exc_col = f"thr_{kind}_{lab}_next", f"excess_{kind}_{lab}_next"
drive_ok()
scales = pd.read_csv(f"{S3B}/cell_scales.csv")[["cell", "season_next", f"{kind}_scale"]]
cols0 = pd.read_parquet(f"{DATA}/val.parquet").columns
BASE_FEATS = [c for c in cols0 if c not in ("date", "cell", "season_next", "doy_sin", "doy_cos")
              and not c.startswith(("y_", "thr_", "excess_", "tmax_next", "tmin_next"))]
v = "t2m_max" if kind == "heat" else "t2m_min"
sign = 1 if kind == "heat" else -1


def load_split(name):
    df = pd.read_parquet(f"{DATA}/{name}.parquet",
                         columns=BASE_FEATS + [TARGET, "cell", "season_next", thr_col, exc_col])
    if name == "train" and TRAIN_FRACTION < 1:
        df = df.sample(frac=TRAIN_FRACTION, random_state=SEED)
    df = df.merge(scales, on=["cell", "season_next"], how="left")
    sc = df[f"{kind}_scale"]
    for lag in (0, 1, 2):
        df[f"dist_l{lag}"] = sign * (df[f"{v}_l{lag}"] - df[thr_col])
        df[f"dist_std_l{lag}"] = df[f"dist_l{lag}"] / sc
    df["dist_trend"] = df["dist_l0"] - df["dist_l1"]
    df["tmax_change"] = df["t2m_max_l0"] - df["t2m_max_l1"]
    df["z"] = (df[exc_col] / sc).where(df[TARGET] == 1)          # severity, extremes only
    return df.reset_index(drop=True)


tr, va, te = load_split("train"), load_split("val"), load_split("test")
FEATS = BASE_FEATS + [f"dist_l{l}" for l in (0, 1, 2)] + [f"dist_std_l{l}" for l in (0, 1, 2)] + \
        ["dist_trend", "tmax_change"]
for nm, df in (("train", tr), ("val", va), ("test", te)):
    out(f"{nm:<5} rows {len(df):>9,}  extremes {int(df[TARGET].sum()):>7,}  ({100 * df[TARGET].mean():.2f}%)")
out(f"{len(FEATS)} classifier features (no calendar, no z)")
ytr, yva, yte = tr[TARGET].values, va[TARGET].values, te[TARGET].values
n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())
N_SYN = int(SYNTH_RATIO * n_neg) - n_pos
out(f"each generator adds {N_SYN:,} synthetic extreme days")
# test subsets for the extra scores
onset_te = te["dist_l0"].values <= 0                       # today NOT past tomorrow's threshold
z_te = te["z"].values
top_te = (yte == 1) & (z_te >= np.nanquantile(z_te, 0.9))  # 10% most severe test extremes
save_summary()

# ================================================================ 2. EVT parameters from Step 3b
fits = pd.read_csv(f"{S3B}/evt_fits_std.csv").set_index("group")
GPD = {}
for k, s in enumerate(SEASON_NAMES):
    r = fits.loc[f"{kind}/{s}"]
    xi, sig = float(r.xi_std), float(r.sigma_std)
    GPD[k] = (xi, sig, sig / -xi if xi < 0 else np.inf)
out("\nGPD tails used to guide generation (standardised excess):")
for k, s in enumerate(SEASON_NAMES):
    xi, sig, ub = GPD[k]
    out(f"   {s:<13} xi {xi:+.3f}  sigma {sig:.3f}  upper bound {ub:.2f}")

# ================================================================ 3. CTGAN
section("3. TRAIN CTGAN ON THE REAL EXTREME DAYS")
GEN_COLS = FEATS + ["season_next", "z"]
gan_path = f"{OUT}/ctgan_{kind}_{lab}.pkl"
real_pos = tr.loc[ytr == 1, GEN_COLS].copy()
real_pos["season_next"] = real_pos["season_next"].astype(int).astype(str)
gan = None
if os.path.exists(gan_path):
    try:
        gan = CTGAN.load(gan_path)
    except Exception:                       # newer torch refuses pickled models by default
        try:
            gan = torch.load(gan_path, weights_only=False, map_location="cpu")
            gan.set_device(torch.device("cuda" if GPU else "cpu"))
        except Exception as e:
            out(f"could not load saved CTGAN ({e.__class__.__name__}); training a new one")
            gan = None
    if gan is not None:
        out(f"loaded saved CTGAN from {gan_path}")
if gan is None:
    progress(f"training CTGAN on {len(real_pos):,} extreme days x {len(GEN_COLS)} columns")
    gan = CTGAN(epochs=GAN_EPOCHS, batch_size=500, cuda=GPU, verbose=False)
    gan.fit(real_pos, discrete_columns=["season_next"])
    drive_ok()
    gan.save(gan_path)
    progress("CTGAN trained and saved")


def physical_ok(df):
    ok = df["z"] > 0
    for l in (0, 1, 2):
        ok &= (df[f"t2m_min_l{l}"] <= df[f"t2m_mean_l{l}"] + 0.3) & (df[f"t2m_mean_l{l}"] <= df[f"t2m_max_l{l}"] + 0.3)
        ok &= df[f"rh_l{l}"].between(0, 100) & (df[f"tp_l{l}"] >= 0) & (df[f"ssrd_l{l}"] >= 0)
        ok &= df[f"swvl1_l{l}"].between(0, 0.8) & (df[f"d2m_l{l}"] <= df[f"t2m_mean_l{l}"] + 0.5)
        ok &= np.sign(df[f"dist_l{l}"]) == np.sign(df[f"dist_std_l{l}"])
    for c in ("lat", "lon", "elev_m", "slope_deg"):
        ok &= df[c].between(real_pos[c].min(), real_pos[c].max())
    return ok


def clean(df):
    df = df.copy()
    df["season_next"] = df["season_next"].astype(int)
    df["dist_trend"] = df["dist_l0"] - df["dist_l1"]          # keep derived columns consistent
    df["tmax_change"] = df["t2m_max_l0"] - df["t2m_max_l1"]
    return df


def evt_resample(pool, n_total):
    """Keep synthetic extremes so that, per season, z follows the fitted GPD."""
    share = pd.Series(tr.loc[ytr == 1, "season_next"]).value_counts(normalize=True)
    parts = []
    for k in range(len(SEASON_NAMES)):
        xi, sig, ub = GPD[k]
        p = pool[(pool.season_next == k) & (pool.z < ub)]
        n_k = int(round(n_total * share.get(k, 0)))
        if len(p) == 0 or n_k == 0:
            continue
        # 20 bins of equal probability under the GPD; weight = target share / generated share
        edges = stats.genpareto.ppf(np.linspace(0, 1, 21), xi, scale=sig)
        edges[-1] = np.inf if not np.isfinite(ub) else ub
        b = np.clip(np.searchsorted(edges, p.z.values, side="right") - 1, 0, 19)
        cnt = np.bincount(b, minlength=20).astype(float)
        w = (1 / 20) / np.maximum(cnt[b] / len(p), 1e-12)
        parts.append(p.sample(n=n_k, replace=True, weights=w, random_state=SEED))
    return pd.concat(parts, ignore_index=True)


syn = {}
for name in ("ctgan_vanilla", "ctgan_physical", "ctgan_evt"):
    path = f"{OUT}/synthetic_{name}.parquet"
    if os.path.exists(path):
        syn[name] = pd.read_parquet(path)
        continue
    if name == "ctgan_vanilla":
        s_ = clean(gan.sample(N_SYN))
        out(f"vanilla: {len(s_):,} samples, {100 * physical_ok(s_).mean():.1f}% pass the physical checks")
    elif name == "ctgan_physical":
        pool = clean(gan.sample(int(N_SYN * 1.6)))
        s_ = pool[physical_ok(pool)].head(N_SYN)
        out(f"physical: kept {len(s_):,} of {len(pool):,}")
    else:
        pool = clean(gan.sample(int(N_SYN * EVT_POOL)))
        pool = pool[physical_ok(pool)]
        s_ = evt_resample(pool, N_SYN)
        out(f"EVT: {len(pool):,} physically valid in the pool -> {len(s_):,} tail-consistent samples")
    drive_ok()
    s_.to_parquet(path, index=False)
    syn[name] = s_
    progress(f"{name} ready")
save_summary()

# ================================================================ 4. tail consistency of the synthetic extremes
section("4. DO SYNTHETIC EXTREMES HAVE THE RIGHT TAIL? (z = standardised severity)")
rows = []
real_pos_df = tr.loc[ytr == 1]
for k, s in enumerate(SEASON_NAMES):
    xi_t, sig_t, ub = GPD[k]
    zr = real_pos_df.loc[real_pos_df.season_next == k, "z"].values
    real_max = zr.max() if len(zr) else np.nan
    for src, d in [("real", real_pos_df)] + [(n, syn[n]) for n in syn]:
        z = d.loc[d.season_next == k, "z"].values
        n_all = len(z)
        z = z[z > 0]
        if len(z) < 50:
            continue
        xi, _, sig = stats.genpareto.fit(z, floc=0)
        ks = stats.kstest(z, "genpareto", args=(xi_t, 0, sig_t)).statistic
        rows.append(dict(season=s, source=src, n=n_all, share_z_le_0=round(1 - len(z) / max(n_all, 1), 3),
                         xi=round(xi, 3), sigma=round(sig, 3),
                         z_q90=round(np.quantile(z, .9), 2), z_q99=round(np.quantile(z, .99), 2),
                         z_max=round(z.max(), 2), share_beyond_real_max=round(float(np.mean(z > real_max)), 4),
                         ks_to_gpd=round(ks, 3)))
tail = pd.DataFrame(rows)
tail.to_csv(f"{OUT}/tail_consistency.csv", index=False)
out(tail.to_string(index=False))
out("ks_to_gpd = distance between the source's z distribution and the fitted GPD (0 = identical).")

fig, axs = plt.subplots(1, 4, figsize=(17, 4))
for k, (ax, s) in enumerate(zip(axs, SEASON_NAMES)):
    xi_t, sig_t, ub = GPD[k]
    for src, d, c in [("real", tr.loc[ytr == 1], "k"), ("ctgan_vanilla", syn["ctgan_vanilla"], "C1"),
                      ("ctgan_evt", syn["ctgan_evt"], "C0")]:
        z = np.sort(d.loc[(d.season_next == k) & (d.z > 0), "z"].values)
        if len(z) < 20:
            continue
        surv = 1 - (np.arange(1, len(z) + 1) - 0.5) / len(z)
        ax.semilogy(z, surv, color=c, lw=1.2, label=src)
    zz = np.linspace(0, min(ub, 4), 200) if np.isfinite(ub) else np.linspace(0, 4, 200)
    ax.semilogy(zz, stats.genpareto.sf(zz, xi_t, scale=sig_t), "k--", lw=0.8, label="fitted GPD")
    ax.set_title(s); ax.set_xlabel("severity z"); ax.set_ylim(1e-4, 1)
axs[0].set_ylabel("share of extremes more severe"); axs[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/tail_survival.png", dpi=120); plt.close()
save_summary()

# ================================================================ 5. classifiers
section("5. SAME CLASSIFIER, DIFFERENT TRAINING DATA")
res_path = f"{OUT}/results.csv"
done = pd.read_csv(res_path) if os.path.exists(res_path) else pd.DataFrame()
preds_path = f"{OUT}/test_predictions.parquet"
preds = pd.read_parquet(preds_path) if os.path.exists(preds_path) else pd.DataFrame(index=range(len(te)))
MONO = [1 if f in ("dist_l0", "dist_std_l0") else 0 for f in FEATS]
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
    r = dict(model=name, val_pr_auc=round(average_precision_score(yva, p_va), 4),
             test_pr_auc=round(average_precision_score(yte, p_te), 4),
             test_roc_auc=round(roc_auc_score(yte, p_te), 4),
             precision=round(prec, 3), recall=round(rec, 3),
             f1=round(2 * prec * rec / max(prec + rec, 1e-12), 3), csi=round(tp / max(tp + fp + fn, 1), 3),
             onset_pr_auc=round(average_precision_score(yte[onset_te], p_te[onset_te]), 4),
             onset_recall=round(float(yhat[onset_te & (yte == 1)].mean()), 3),
             top10_recall=round(float(yhat[top_te].mean()), 3), note=note)
    return r


def save_result(r, p_te):
    global done, preds
    drive_ok()
    done = pd.concat([done[done.model != r["model"]] if len(done) else done, pd.DataFrame([r])], ignore_index=True)
    done.to_csv(res_path, index=False)
    preds[r["model"]] = p_te.astype("float32")
    preds.to_parquet(preds_path, index=False)


def fit_eval(name, X, y, note="", **extra):
    if len(done) and name in set(done.model):
        out(f"   {name}: already done, skipped")
        return
    progress(f"training {name} on {len(y):,} rows")
    m = lgb.LGBMClassifier(**LGB_PARAMS, monotone_constraints=MONO, **extra)
    m.fit(X, y, eval_set=[(Xva, yva)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(60, verbose=False)])
    r = evaluate(name, m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], f"{note}{m.best_iteration_} trees")
    save_result(r, m.predict_proba(Xte)[:, 1])
    progress(f"{name}: test PR-AUC {r['test_pr_auc']}")


if not (len(done) and "persistence" in set(done.model)):
    save_result(evaluate("persistence", va["dist_l0"].values, te["dist_l0"].values, "no model"),
                te["dist_l0"].values)
Xtr = tr[FEATS].values.astype("float32")
fit_eval("plain", Xtr, ytr)
fit_eval("weighted", Xtr, ytr, note=f"weight {n_neg / n_pos:.0f}, ", scale_pos_weight=n_neg / n_pos)
if not (len(done) and "smote" in set(done.model)):
    progress("running SMOTE")
    Xs, ys = SMOTE(sampling_strategy=SYNTH_RATIO, random_state=SEED).fit_resample(Xtr, ytr)
    fit_eval("smote", Xs, ys, note=f"+{len(ys) - len(ytr):,} synthetic, ")
    del Xs, ys
for name in ("ctgan_vanilla", "ctgan_physical", "ctgan_evt"):
    Xg = syn[name][FEATS].values.astype("float32")
    fit_eval(name, np.vstack([Xtr, Xg]), np.concatenate([ytr, np.ones(len(Xg), dtype=ytr.dtype)]),
             note=f"+{len(Xg):,} synthetic, ")

order = ["persistence", "plain", "weighted", "smote", "ctgan_vanilla", "ctgan_physical", "ctgan_evt"]
done = done.set_index("model").reindex([o for o in order if o in set(done.model)]).reset_index()
out("\nTest years 2016-2025 (decision threshold picked on 2011-2015):")
out(done.to_string(index=False))
out("""
test_pr_auc   main score for rare events (random = share of extremes)
onset_*       only days where TODAY is not extreme: can the model see an extreme coming?
top10_recall  share of the 10% most severe test extremes that were caught
csi           hits / (hits + false alarms + misses)""")

rows = []
for k, s in enumerate(SEASON_NAMES):
    m = te.season_next.values == k
    r = {"season": s, "extremes": int(yte[m].sum())}
    for c in [c for c in order if c in preds.columns]:
        r[c] = round(average_precision_score(yte[m], preds[c].values[m]), 3)
    rows.append(r)
bys = pd.DataFrame(rows)
bys.to_csv(f"{OUT}/results_by_season.csv", index=False)
out("\nTest PR-AUC by season:")
out(bys.to_string(index=False))

plt.figure(figsize=(6.5, 5))
for c in [c for c in order if c in preds.columns]:
    pr, rc, _ = precision_recall_curve(yte, preds[c].values)
    plt.plot(rc, pr, lw=2 if c == "ctgan_evt" else 1, label=f"{c} ({average_precision_score(yte, preds[c]):.3f})")
plt.axhline(yte.mean(), color="grey", ls="--", lw=0.8, label=f"random ({yte.mean():.3f})")
plt.xlabel("recall"); plt.ylabel("precision"); plt.title(f"{TARGET}, test 2016-2025")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/pr_curves.png", dpi=120); plt.close()

out(f"\nfiles in {OUT}")
out(f"END  (took {(time.time() - T0) / 60:.1f} min)")
save_summary()
print(f"\nSummary saved to {BASE}/step4_summary.txt")
