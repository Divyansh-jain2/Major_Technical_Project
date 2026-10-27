# ============================================================================
# STEP 3b: FIX THE TWO PROBLEMS STEP 3 FOUND
# Paste into ONE Colab cell and run. Free Colab CPU: about 15-20 minutes.
#
# Problem 1 (EVT): pooling all 529 cells mixed calm low cells with jumpy high cells,
#   so several GPD fits looked heavy-tailed when the real tail is bounded.
#   Fix: divide each excess by that cell's own day-to-day variability (the std of
#   its temperature anomaly in that season, 1981-2010) before pooling. The GPD is then
#   fitted to "excess in units of the cell's normal variability".
#
# Problem 2 (classifiers): the no-model persistence rule beat LightGBM because the
#   model never saw "how far is today from tomorrow's threshold" directly.
#   Fix: add those distances (today, yesterday, the day before) as features.
#   Thresholds come from 1981-2010 only, so this is not leakage.
#
# Reads (never changes):  data/model_ready/{train,val,test}.parquet
# Writes ONLY to:         results/step3b/
#   cell_scales.csv       per cell and season: variability used to standardise (Step 4 needs it)
#   evt_fits_std.csv      GPD fits on standardised excesses, next to the Step 3 raw fits
#   evt_qq_std.png        Q-Q plots, raw (grey) vs standardised (blue)
#   baselines_v2.csv, baselines_v2_by_season.csv, pr_curves_v2.png, feature_importance_v2.csv
#   step3b_summary.txt    (also copied to himachal_project/step3b_summary.txt)
# ============================================================================
!pip install -q lightgbm imbalanced-learn

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, brier_score_loss
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from google.colab import drive

warnings.filterwarnings("ignore")
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/himachal_project"
DATA = f"{BASE}/data/model_ready"
OUT = f"{BASE}/results/step3b"
STEP3 = f"{BASE}/results/step3"

# ---------------------------------------------------------------- settings
TARGET = "y_heat_season"
SEASON_NAMES = ["winter", "pre_monsoon", "monsoon", "post_monsoon"]
BOOT = 100
SMOTE_RATIO = 0.10
TRAIN_FRACTION = 1.0
SEED = 42
# ----------------------------------------------------------------

os.makedirs(OUT, exist_ok=True)
LINES, T0 = [], time.time()
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
rng = np.random.default_rng(SEED)


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def progress(msg):
    print(f"   ... [{(time.time() - T0) / 60:4.1f} min] {msg}", flush=True)


def save_summary():
    txt = "\n".join(LINES) + "\n"
    for p in (f"{OUT}/step3b_summary.txt", f"{BASE}/step3b_summary.txt"):
        with open(p, "w") as fh:
            fh.write(txt)


def section(t):
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


out(f"STEP 3b  {time.strftime('%Y-%m-%d %H:%M')}")

# ================================================================ PART A: standardised EVT
section("A. GPD ON STANDARDISED EXCESSES (train years 1981-2010)")
need = ["date", "cell", "season_next", "tmax_anom_l0", "tmin_anom_l0"]
for kind in ("heat", "cold"):
    for lab in ("season", "annual", "doy"):
        need += [f"y_{kind}_{lab}", f"excess_{kind}_{lab}_next"]
tr = pd.read_parquet(f"{DATA}/train.parquet", columns=need)
te = pd.read_parquet(f"{DATA}/test.parquet", columns=need)
progress("loaded")

# per-cell variability: std of the daily anomaly (vs that cell's normal for the date)
sc_sea = tr.groupby(["cell", "season_next"]).agg(heat_scale=("tmax_anom_l0", "std"),
                                                 cold_scale=("tmin_anom_l0", "std")).reset_index()
sc_all = tr.groupby("cell").agg(heat_scale_all=("tmax_anom_l0", "std"),
                                cold_scale_all=("tmin_anom_l0", "std")).reset_index()
scales = sc_sea.merge(sc_all, on="cell")
scales.to_csv(f"{OUT}/cell_scales.csv", index=False)
out("cell variability used for standardising (std of daily anomaly, C), spread over cells:")
out(scales[["heat_scale", "cold_scale"]].describe().round(2).T.to_string())


def add_scales(df):
    return df.merge(scales, on=["cell", "season_next"], how="left")


tr, te = add_scales(tr), add_scales(te)


def gpd_fit(x):
    c, _, s = stats.genpareto.fit(x, floc=0)
    return c, s


def groups(df):
    for kind in ("heat", "cold"):
        for k, s in enumerate(SEASON_NAMES):
            yield f"{kind}/{s}", kind, "season", (df[f"y_{kind}_season"] == 1) & (df.season_next == k)
        yield f"{kind}/whole_year", kind, "annual", df[f"y_{kind}_annual"] == 1
        yield f"{kind}/day_of_year", kind, "doy", df[f"y_{kind}_doy"] == 1


def tail_misfit(x, xi, sig):
    """Mean |observed - model| over the top 10% of points, relative to the model's 90-99.9% range."""
    xs = np.sort(x)
    p = (np.arange(1, len(xs) + 1) - 0.5) / len(xs)
    th = stats.genpareto.ppf(p, xi, scale=sig)
    top = p >= 0.9
    span = stats.genpareto.ppf(0.999, xi, scale=sig) - stats.genpareto.ppf(0.9, xi, scale=sig)
    return float(np.mean(np.abs(xs[top] - th[top])) / max(span, 1e-9))


raw_fits = pd.read_csv(f"{STEP3}/evt_fits.csv").set_index("group") if os.path.exists(f"{STEP3}/evt_fits.csv") else None
fits, qq = [], {}
for name, kind, lab, sel in groups(tr):
    scol = f"{kind}_scale" if lab == "season" else f"{kind}_scale_all"
    x_raw = tr.loc[sel, f"excess_{kind}_{lab}_next"].values.astype("float64")
    s = tr.loc[sel, scol].values
    d = tr.loc[sel, "date"].values
    ok = (x_raw > 0) & (s > 0)
    x_raw, s, d = x_raw[ok], s[ok], d[ok]
    z = x_raw / s
    xi, sig = gpd_fit(z)
    xi_r, sig_r = gpd_fit(x_raw)
    ud = np.unique(d)
    idx_by_day = pd.Series(np.arange(len(d))).groupby(d).apply(np.array)
    bx = np.array([gpd_fit(z[np.concatenate(idx_by_day.loc[rng.choice(ud, len(ud))].values)])
                   for _ in range(BOOT)])
    stab = [gpd_fit(z[z > np.quantile(z, q)] - np.quantile(z, q))[0] for q in (0.25, 0.5)]
    sel_te = dict((n_, s_) for n_, _, _, s_ in groups(te))[name]
    zt = (te.loc[sel_te, f"excess_{kind}_{lab}_next"] / te.loc[sel_te, scol]).values
    zt = zt[zt > 0]
    q90 = stats.genpareto.ppf(0.9, xi, scale=sig)
    fits.append(dict(group=name, n=len(z), event_days=len(ud),
                     xi_raw=round(xi_r, 3), misfit_raw=round(tail_misfit(x_raw, xi_r, sig_r), 3),
                     xi_std=round(xi, 3), xi_lo=round(np.percentile(bx[:, 0], 2.5), 3),
                     xi_hi=round(np.percentile(bx[:, 0], 97.5), 3), sigma_std=round(sig, 3),
                     xi_at_q25=round(stab[0], 3), xi_at_q50=round(stab[1], 3),
                     misfit_std=round(tail_misfit(z, xi, sig), 3),
                     upper_bound_std=round(sig / -xi, 2) if xi < 0 else "none",
                     test_share_beyond_q90=round(float(np.mean(zt > q90)), 3) if len(zt) else np.nan))
    qq[name] = (x_raw, xi_r, sig_r, z, xi, sig)
    progress(f"fitted {name}")
fits = pd.DataFrame(fits)
fits.to_csv(f"{OUT}/evt_fits_std.csv", index=False)
out(fits.to_string(index=False))
out("""
xi_raw / misfit_raw   Step 3 fit on raw excesses (C) and how badly its far tail misses
xi_std [xi_lo, xi_hi] fit on standardised excesses, with 95% interval (resampling whole days)
misfit                mean gap between data and model over the top 10% of points, as a share of the
                      model's 90th-99.9th pct range: below ~0.1 = good, above ~0.3 = poor
upper_bound_std       largest standardised excess the tail allows (x the cell's scale = C)
test_share_beyond_q90 share of 2016-2025 extremes beyond the fitted 90th pct (0.10 = unchanged tail)""")

fig, axs = plt.subplots(2, 6, figsize=(19, 6.6))
for ax, (name, (xr_, xir, sgr, z, xi, sig)) in zip(np.ravel(axs), qq.items()):
    for x, a, b, col, lab in ((xr_ / xr_.std(), xir, sgr / xr_.std(), "0.6", "raw (rescaled)"),
                              (z, xi, sig, "C0", "standardised")):
        xs = np.sort(x)
        p = (np.arange(1, len(xs) + 1) - 0.5) / len(xs)
        ax.plot(stats.genpareto.ppf(p, a, scale=b), xs, ".", ms=2, color=col, alpha=0.5, label=lab)
    lim = np.max(z) * 1.3
    ax.plot([0, lim], [0, lim], "k-", lw=0.8)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(f"{name}\nxi raw {xir:.2f} -> std {xi:.2f}", fontsize=9)
axs[0, 0].legend(fontsize=7, markerscale=4)
plt.tight_layout(); plt.savefig(f"{OUT}/evt_qq_std.png", dpi=110); plt.close()
del tr, te
save_summary()

# ================================================================ PART B: baselines with threshold-distance features
section(f"B. BASELINES WITH THRESHOLD-DISTANCE FEATURES ({TARGET})")
kind = "heat" if "heat" in TARGET else "cold"
lab = TARGET.split("_", 2)[2]
thr_col = f"thr_{kind}_{lab}_next"
base_cols = pd.read_parquet(f"{DATA}/val.parquet").columns
BASE_FEATS = [c for c in base_cols if c not in ("date", "cell", "season_next")
              and not c.startswith(("y_", "thr_", "excess_", "tmax_next", "tmin_next"))]


def add_features(df):
    """New features: distance of today / yesterday / day before to TOMORROW's threshold."""
    v = "t2m_max" if kind == "heat" else "t2m_min"
    sign = 1 if kind == "heat" else -1
    sc = df[f"{kind}_scale"]
    for lag in (0, 1, 2):
        df[f"dist_l{lag}"] = sign * (df[f"{v}_l{lag}"] - df[thr_col])
        df[f"dist_std_l{lag}"] = df[f"dist_l{lag}"] / sc
    df["dist_trend"] = df["dist_l0"] - df["dist_l1"]
    df["tmax_change"] = df["t2m_max_l0"] - df["t2m_max_l1"]
    return df


NEW_FEATS = [f"dist_l{l}" for l in (0, 1, 2)] + [f"dist_std_l{l}" for l in (0, 1, 2)] + ["dist_trend", "tmax_change"]
scales = pd.read_csv(f"{OUT}/cell_scales.csv")


def load_split(name):
    df = pd.read_parquet(f"{DATA}/{name}.parquet", columns=BASE_FEATS + [TARGET, "cell", "season_next", thr_col])
    if name == "train" and TRAIN_FRACTION < 1:
        df = df.sample(frac=TRAIN_FRACTION, random_state=SEED)
    df = df.merge(scales[["cell", "season_next", f"{kind}_scale"]], on=["cell", "season_next"], how="left")
    return add_features(df)


trd, vad, ted = load_split("train"), load_split("val"), load_split("test")
FEATS = BASE_FEATS + NEW_FEATS
FEATS_NODOY = [f for f in FEATS if f not in ("doy_sin", "doy_cos")]
yva, yte = vad[TARGET].values, ted[TARGET].values
for nm, df in (("train", trd), ("val", vad), ("test", ted)):
    out(f"{nm:<5} rows {len(df):>9,}  extremes {int(df[TARGET].sum()):>7,}  ({100 * df[TARGET].mean():.2f}%)")
progress("data loaded")


def pick_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return th[int(np.nanargmax(f1[:-1]))]


def scores(y, p, thr):
    yhat = p >= thr
    tp = int((yhat & (y == 1)).sum()); fp = int((yhat & (y == 0)).sum()); fn = int((~yhat & (y == 1)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    pr, rc, _ = precision_recall_curve(y, p)
    return dict(pr_auc=round(average_precision_score(y, p), 4), roc_auc=round(roc_auc_score(y, p), 4),
                precision=round(prec, 3), recall=round(rec, 3),
                f1=round(2 * prec * rec / max(prec + rec, 1e-12), 3),
                f2=round(5 * prec * rec / max(4 * prec + rec, 1e-12), 3),
                csi=round(tp / max(tp + fp + fn, 1), 3),
                recall_at_prec50=round(float(rc[pr >= 0.5].max()) if (pr >= 0.5).any() else 0.0, 3))


LGB_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, max_bin=127,
                  n_estimators=400, random_state=SEED, n_jobs=-1, verbose=-1)
results, preds, models = [], {}, {}


def run(name, feats, X=None, y=None, note="", **extra):
    Xtr = trd[feats].values.astype("float32") if X is None else X
    ytr = trd[TARGET].values if y is None else y
    Xva, Xte = vad[feats].values.astype("float32"), ted[feats].values.astype("float32")
    m = lgb.LGBMClassifier(**LGB_PARAMS, **extra)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(40, verbose=False)])
    record(name, m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], f"{note}{m.best_iteration_} trees")
    models[name] = (m, feats)


def record(name, p_va, p_te, note):
    thr = pick_threshold(yva, p_va)
    for split, y, p in (("val", yva, p_va), ("test", yte, p_te)):
        r = dict(model=name, split=split, **scores(y, p, thr))
        if split == "test" and np.all((p >= 0) & (p <= 1)):
            r["brier"] = round(brier_score_loss(y, p), 5)
        r["note"] = note
        results.append(r)
    preds[name] = p_te
    progress(f"{name} done")


record("persistence", vad["dist_l0"].values, ted["dist_l0"].values, "no model")
run("lgbm_plain", FEATS)
run("lgbm_plain_no_doy", FEATS_NODOY, note="no calendar features, ")
w = (trd[TARGET] == 0).sum() / max((trd[TARGET] == 1).sum(), 1)
run("lgbm_weighted", FEATS, note=f"weight {w:.0f}, ", scale_pos_weight=w)
progress("running SMOTE")
Xs, ys = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=SEED, k_neighbors=5).fit_resample(
    trd[FEATS].values.astype("float32"), trd[TARGET].values)
run("lgbm_smote", FEATS, X=Xs, y=ys, note=f"+{len(ys) - len(trd):,} synthetic, ")
del Xs, ys

res = pd.DataFrame(results)
res.to_csv(f"{OUT}/baselines_v2.csv", index=False)
out("\nScores (decision threshold picked on 2011-2015, applied to 2016-2025):")
out(res.to_string(index=False))
if os.path.exists(f"{STEP3}/baselines.csv"):
    old = pd.read_csv(f"{STEP3}/baselines.csv")
    old = old[old.split == "test"].set_index("model")["pr_auc"]
    new = res[res.split == "test"].set_index("model")["pr_auc"]
    cmp = pd.DataFrame({"step3_test_pr_auc": old, "step3b_test_pr_auc": new})
    out("\nTest PR-AUC, before vs after the new features:")
    out(cmp.to_string())

rows = []
for k, s in enumerate(SEASON_NAMES):
    m = ted.season_next.values == k
    r = {"season": s, "extremes": int(yte[m].sum())}
    for name, p in preds.items():
        r[name] = round(average_precision_score(yte[m], p[m]), 3) if yte[m].sum() else np.nan
    rows.append(r)
bys = pd.DataFrame(rows)
bys.to_csv(f"{OUT}/baselines_v2_by_season.csv", index=False)
out("\nTest PR-AUC by season:")
out(bys.to_string(index=False))

m, feats = models["lgbm_plain"]
imp = pd.Series(m.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
imp = (100 * imp / imp.sum()).round(1)
imp.to_csv(f"{OUT}/feature_importance_v2.csv", header=["gain_pct"])
out("\nTop 12 features (lgbm_plain, % of gain):")
out(imp.head(12).to_string())

plt.figure(figsize=(6, 5))
for name, p in preds.items():
    pr, rc, _ = precision_recall_curve(yte, p)
    plt.plot(rc, pr, label=f"{name} ({average_precision_score(yte, p):.3f})")
plt.axhline(yte.mean(), color="grey", ls="--", lw=0.8, label=f"random ({yte.mean():.3f})")
plt.xlabel("recall"); plt.ylabel("precision"); plt.title(f"{TARGET}, test 2016-2025")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/pr_curves_v2.png", dpi=120); plt.close()

out(f"\nfiles in {OUT}")
out(f"END  (took {(time.time() - T0) / 60:.1f} min)")
save_summary()
print(f"\nSummary saved to {BASE}/step3b_summary.txt")
