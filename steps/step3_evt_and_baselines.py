# ============================================================================
# STEP 3: EVT TAIL FITS + BASELINE CLASSIFIERS
# Paste into ONE Colab cell and run. Free Colab CPU: about 15-30 minutes.
#
# Reads (never changes):  himachal_project/data/model_ready/{train,val,test}.parquet
# Writes ONLY to:         himachal_project/results/step3/
#   evt_fits.csv          GPD fit per season / whole year / day-of-year, heat and cold
#   evt_qq.png            Q-Q plots: does the GPD fit the observed exceedances?
#   baselines.csv         every baseline's scores on validation and test
#   baselines_by_season.csv   test PR-AUC per season
#   pr_curves.png         precision-recall curves on the test years
#   feature_importance.csv
#   step3_summary.txt     (also copied to himachal_project/step3_summary.txt)
#
# PART A - EVT. For every extreme day, "excess" = how far Tmax went above the
#   threshold (or Tmin below it, for cold). Peaks-over-threshold theory says these
#   excesses follow a Generalized Pareto Distribution (GPD) with shape xi:
#     xi < 0  bounded tail (there is a physical maximum)    <- expected for temperature
#     xi = 0  exponential tail;  xi > 0  heavy tail.
#   Fitted on 1981-2010 only. Confidence intervals resample whole DAYS, because
#   one hot day makes many neighbouring cells extreme together.
#
# PART B - BASELINES for next-day extreme prediction (target set below):
#   persistence     no model: "is today already past tomorrow's threshold?"
#   lgbm_plain      LightGBM, no rebalancing
#   lgbm_weighted   LightGBM, extreme days weighted up (class weights)
#   lgbm_smote      LightGBM on data with SMOTE-made extra extreme days
#   Trained on 1981-2010, decision threshold picked on 2011-2015, scored on 2016-2025.
# ============================================================================
!pip install -q lightgbm imbalanced-learn

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             brier_score_loss)
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from google.colab import drive

warnings.filterwarnings("ignore")
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/himachal_project"
DATA = f"{BASE}/data/model_ready"
OUT = f"{BASE}/results/step3"

# ---------------------------------------------------------------- settings
TARGET = "y_heat_season"           # main label for the classifiers
SEASON_NAMES = ["winter", "pre_monsoon", "monsoon", "post_monsoon"]
BOOT = 100                         # bootstrap resamples for GPD confidence intervals
SMOTE_RATIO = 0.10                 # SMOTE grows extremes to 10% of normal days
TRAIN_FRACTION = 1.0               # <1.0 = random sample of train rows (keeps the imbalance)
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
    for p in (f"{OUT}/step3_summary.txt", f"{BASE}/step3_summary.txt"):
        with open(p, "w") as fh:
            fh.write(txt)


def section(t):
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


out(f"STEP 3  {time.strftime('%Y-%m-%d %H:%M')}   target for classifiers: {TARGET}")

# ================================================================ PART A: EVT
section("A. EVT: GPD FITS TO THE EXCESSES (train years 1981-2010)")
need = ["date", "season_next"]
for kind in ("heat", "cold"):
    for lab in ("season", "annual", "doy"):
        need += [f"y_{kind}_{lab}", f"excess_{kind}_{lab}_next"]
tr_evt = pd.read_parquet(f"{DATA}/train.parquet", columns=need)
te_evt = pd.read_parquet(f"{DATA}/test.parquet", columns=need)
progress("loaded excess columns")


def gpd_fit(x):
    c, _, s = stats.genpareto.fit(x, floc=0)
    return c, s


def groups(df):
    """(name, kind, label, rows) for every fit we want."""
    for kind in ("heat", "cold"):
        for k, s in enumerate(SEASON_NAMES):
            yield f"{kind}/{s}", kind, "season", (df[f"y_{kind}_season"] == 1) & (df.season_next == k)
        yield f"{kind}/whole_year", kind, "annual", df[f"y_{kind}_annual"] == 1
        yield f"{kind}/day_of_year", kind, "doy", df[f"y_{kind}_doy"] == 1


days_per_season = {"winter": 90.25, "pre_monsoon": 92, "monsoon": 122, "post_monsoon": 61}
n_years_train = tr_evt.date.dt.year.nunique()
fits, qq = [], {}
for name, kind, lab, sel in groups(tr_evt):
    x = tr_evt.loc[sel, f"excess_{kind}_{lab}_next"].values.astype("float64")
    d = tr_evt.loc[sel, "date"].values
    x, d = x[x > 0], d[x > 0]
    if len(x) < 50:
        out(f"   {name}: only {len(x)} exceedances, skipped")
        continue
    xi, sig = gpd_fit(x)
    # bootstrap over event days
    ud = np.unique(d)
    idx_by_day = pd.Series(np.arange(len(d))).groupby(d).apply(np.array)
    bx = []
    for _ in range(BOOT):
        pick = rng.choice(ud, size=len(ud), replace=True)
        ii = np.concatenate(idx_by_day.loc[pick].values)
        bx.append(gpd_fit(x[ii]))
    bx = np.array(bx)
    # threshold stability: refit above higher thresholds (xi should stay similar)
    stab = []
    for q in (0.25, 0.5):
        u2 = np.quantile(x, q)
        stab.append(gpd_fit(x[x > u2] - u2)[0])
    # return levels in degrees beyond the local threshold, per cell
    if lab == "season":
        sname = name.split("/")[1]
        ny = days_per_season[sname]
        zeta = sel.sum() / ((tr_evt.season_next == SEASON_NAMES.index(sname)).sum())
    else:
        ny = 365.25
        zeta = sel.sum() / len(tr_evt)
    rl = {}
    for m in (10, 50, 100):
        nm = m * ny * zeta
        rl[m] = sig / xi * (nm ** xi - 1) if abs(xi) > 1e-6 else sig * np.log(nm)
    upper = sig / -xi if xi < 0 else np.inf
    # does 2016-2025 follow the 1981-2010 tail?
    sel_te = dict((n, s) for n, _, _, s in groups(te_evt))[name]
    xt = te_evt.loc[sel_te, f"excess_{kind}_{lab}_next"].values
    xt = xt[xt > 0]
    q90_model = stats.genpareto.ppf(0.9, xi, scale=sig)
    fits.append(dict(group=name, n_exceed=len(x), event_days=len(ud),
                     xi=round(xi, 3), xi_lo=round(np.percentile(bx[:, 0], 2.5), 3),
                     xi_hi=round(np.percentile(bx[:, 0], 97.5), 3), sigma=round(sig, 3),
                     xi_at_q25=round(stab[0], 3), xi_at_q50=round(stab[1], 3),
                     mean_excess=round(x.mean(), 2), max_excess=round(x.max(), 2),
                     upper_bound=round(upper, 2) if np.isfinite(upper) else "none",
                     rl10=round(rl[10], 2), rl50=round(rl[50], 2), rl100=round(rl[100], 2),
                     test_n=len(xt), test_mean_excess=round(xt.mean(), 2) if len(xt) else np.nan,
                     test_share_beyond_model_q90=round(np.mean(xt > q90_model), 3) if len(xt) else np.nan))
    qq[name] = (x, xi, sig)
    progress(f"fitted {name} (n={len(x)})")
fits = pd.DataFrame(fits)
fits.to_csv(f"{OUT}/evt_fits.csv", index=False)
out(fits.to_string(index=False))
out("""
How to read it (all values in C beyond the local threshold):
  xi [xi_lo, xi_hi]   tail shape and 95% interval; negative = bounded tail
  xi_at_q25/q50       xi refitted above higher thresholds; similar values = stable fit
  upper_bound         largest excess the fitted tail allows (only when xi < 0)
  rl10/50/100         10-, 50-, 100-year return level of the excess, per cell
  test_share_beyond_model_q90   share of 2016-2025 extremes beyond the GPD's 90th pct
                      (0.10 = recent extremes follow the old tail; higher = recent tail heavier)""")

# Q-Q plots
n = len(qq)
cols = 6
rows = int(np.ceil(n / cols))
fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
for ax, (name, (x, xi, sig)) in zip(np.ravel(axs), qq.items()):
    xs = np.sort(x)
    p = (np.arange(1, len(xs) + 1) - 0.5) / len(xs)
    th = stats.genpareto.ppf(p, xi, scale=sig)
    ax.plot(th, xs, ".", ms=2, alpha=0.4)
    lim = [0, max(th.max(), xs.max())]
    ax.plot(lim, lim, "k-", lw=0.8)
    ax.set_title(f"{name}\nxi={xi:.2f}", fontsize=9)
    ax.set_xlabel("GPD quantile (C)", fontsize=8)
    ax.set_ylabel("observed excess (C)", fontsize=8)
for ax in np.ravel(axs)[n:]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/evt_qq.png", dpi=110)
plt.close()
del tr_evt, te_evt
save_summary()

# ================================================================ PART B: baselines
section(f"B. BASELINE CLASSIFIERS FOR {TARGET} (next-day extreme)")
cols_all = pd.read_parquet(f"{DATA}/val.parquet").columns
FEATS = [c for c in cols_all if c not in ("date", "cell", "season_next")
         and not c.startswith(("y_", "thr_", "excess_", "tmax_next", "tmin_next"))]
kind = "heat" if "heat" in TARGET else "cold"
lab = TARGET.split("_", 2)[2]
pers_thr = f"thr_{kind}_{lab}_next"
load = FEATS + [TARGET, "season_next", pers_thr]


def load_split(name):
    df = pd.read_parquet(f"{DATA}/{name}.parquet", columns=load)
    if name == "train" and TRAIN_FRACTION < 1:
        df = df.sample(frac=TRAIN_FRACTION, random_state=SEED)
    return df


tr, va, te = load_split("train"), load_split("val"), load_split("test")
for nm, df in (("train", tr), ("val", va), ("test", te)):
    out(f"{nm:<5} rows {len(df):>9,}  extremes {int(df[TARGET].sum()):>7,}  ({100 * df[TARGET].mean():.2f}%)")
Xtr, ytr = tr[FEATS].values.astype("float32"), tr[TARGET].values
Xva, yva = va[FEATS].values.astype("float32"), va[TARGET].values
Xte, yte = te[FEATS].values.astype("float32"), te[TARGET].values
progress("data loaded")


def persistence_score(df):
    t = df["t2m_max_l0"] if kind == "heat" else -df["t2m_min_l0"]
    thr = df[pers_thr] if kind == "heat" else -df[pers_thr]
    return (t - thr).values


def pick_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    k = int(np.nanargmax(f1[:-1]))
    return th[k]


def scores(y, p, thr):
    yhat = p >= thr
    tp = int((yhat & (y == 1)).sum()); fp = int((yhat & (y == 0)).sum()); fn = int((~yhat & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    f2 = 5 * prec * rec / max(4 * prec + rec, 1e-12)
    pr, rc, _ = precision_recall_curve(y, p)
    rec_at_p50 = float(rc[pr >= 0.5].max()) if (pr >= 0.5).any() else 0.0
    return dict(pr_auc=round(average_precision_score(y, p), 4), roc_auc=round(roc_auc_score(y, p), 4),
                precision=round(prec, 3), recall=round(rec, 3), f1=round(f1, 3), f2=round(f2, 3),
                csi=round(tp / max(tp + fp + fn, 1), 3), recall_at_prec50=round(rec_at_p50, 3))


LGB_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, max_bin=127,
                  n_estimators=400, random_state=SEED, n_jobs=-1, verbose=-1)


def fit_lgbm(X, y, **extra):
    m = lgb.LGBMClassifier(**LGB_PARAMS, **extra)
    m.fit(X, y, eval_set=[(Xva, yva)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(40, verbose=False)])
    return m


results, preds = [], {}


def record(name, p_va, p_te, extra=""):
    thr = pick_threshold(yva, p_va)
    for split, y, p in (("val", yva, p_va), ("test", yte, p_te)):
        r = dict(model=name, split=split, **scores(y, p, thr))
        if split == "test" and np.all((p >= 0) & (p <= 1)):
            r["brier"] = round(brier_score_loss(y, p), 5)
        r["note"] = extra
        results.append(r)
    preds[name] = p_te
    progress(f"{name} done")


record("persistence", persistence_score(va), persistence_score(te), "no model")

m_plain = fit_lgbm(Xtr, ytr)
record("lgbm_plain", m_plain.predict_proba(Xva)[:, 1], m_plain.predict_proba(Xte)[:, 1],
       f"{m_plain.best_iteration_} trees")

w = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
m_w = fit_lgbm(Xtr, ytr, scale_pos_weight=w)
record("lgbm_weighted", m_w.predict_proba(Xva)[:, 1], m_w.predict_proba(Xte)[:, 1],
       f"weight {w:.0f}, {m_w.best_iteration_} trees")

progress("running SMOTE")
Xs, ys = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=SEED, k_neighbors=5).fit_resample(Xtr, ytr)
m_s = fit_lgbm(Xs, ys)
record("lgbm_smote", m_s.predict_proba(Xva)[:, 1], m_s.predict_proba(Xte)[:, 1],
       f"+{len(ys) - len(ytr):,} synthetic, {m_s.best_iteration_} trees")
del Xs, ys

res = pd.DataFrame(results)
res.to_csv(f"{OUT}/baselines.csv", index=False)
out("\nScores (decision threshold chosen on 2011-2015 to maximise F1, then applied to 2016-2025):")
out(res.to_string(index=False))
out("""
pr_auc   area under precision-recall curve: the main score for rare events (random = share of extremes)
recall   share of real extremes caught;  precision = share of alarms that were real
csi      hits / (hits + false alarms + misses), common in weather forecasting
brier    probability error (only for models that output probabilities)""")

# per-season test PR-AUC
rows = []
for k, s in enumerate(SEASON_NAMES):
    m = te.season_next.values == k
    r = {"season": s, "test_rows": int(m.sum()), "extremes": int(yte[m].sum())}
    for name, p in preds.items():
        r[name] = round(average_precision_score(yte[m], p[m]), 3) if yte[m].sum() else np.nan
    rows.append(r)
bys = pd.DataFrame(rows)
bys.to_csv(f"{OUT}/baselines_by_season.csv", index=False)
out("\nTest PR-AUC by season:")
out(bys.to_string(index=False))

imp = pd.Series(m_plain.booster_.feature_importance("gain"), index=FEATS).sort_values(ascending=False)
imp = (100 * imp / imp.sum()).round(1)
imp.to_csv(f"{OUT}/feature_importance.csv", header=["gain_pct"])
out("\nTop 12 features (lgbm_plain, % of total gain):")
out(imp.head(12).to_string())

plt.figure(figsize=(6, 5))
for name, p in preds.items():
    pr, rc, _ = precision_recall_curve(yte, p)
    plt.plot(rc, pr, label=f"{name} (PR-AUC {average_precision_score(yte, p):.3f})")
plt.axhline(yte.mean(), color="grey", ls="--", lw=0.8, label=f"random ({yte.mean():.3f})")
plt.xlabel("recall"); plt.ylabel("precision"); plt.title(f"{TARGET}, test 2016-2025")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/pr_curves.png", dpi=120); plt.close()

out(f"\nfiles in {OUT}")
out(f"END  (took {(time.time() - T0) / 60:.1f} min)")
save_summary()
print(f"\nSummary saved to {BASE}/step3_summary.txt")
