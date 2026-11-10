# ============================================================================
# STEP 12: WHICH GENERATOR HELPS LIGHTGBM MOST?  (fixed classifier, 5 seeds)
# Paste into ONE Colab cell. CPU runtime is fine. ~1.5-2 h (set SEEDS = [42, 1, 2] for ~1 h).
# Re-run the same cell after a disconnect: finished models are skipped.
#
# Fixed: LightGBM (same settings as Steps 8-10), next-day seasonal heat extreme, train 1981-2010,
#        validation 2011-2015 (all choices), test 2016-2025 (scores only).
# Varied: the extra extreme days added to training - always 1 synthetic per real extreme
#        (the best amount in Steps 9-10), drawn again for every seed.
#   none                                  real data only
#   smote, borderline_smote, adasyn       interpolation (made here, on the classifier's features)
#   gaussian_copula, tvae, ctgan,         from Step 11, each as _plain (generator as it is)
#   copulagan, cwgan                      and _tail (EVT tail focus)
#   diffusion_plain, diffusion_tail       from Step 7 (= ddpm_emp, ddpm_evt_tail); their models are
#                                         reused from Step 10 (same recipe), not retrained
# Measured:
#   A. per-seed scores (PR-AUC, onset PR-AUC, severe PR-AUC): mean, spread, wins over "none"
#   B. seed-averaged models (mean of 5) with 95% month-block intervals and paired tests
#   C. quality of each synthetic set: can a classifier tell it from real extremes (detection AUC,
#      0.5 = indistinguishable), how close it sits to real rows (copying check), plus Step 11's
#      physics, tail and control numbers
#   D. one ranking table + charts
#
# Reads (never changes): data/model_ready_v3/*, results/step7/synthetic_ddpm_*.parquet,
#                        results/step10/preds/*, results/step11/*
# Writes ONLY to:        results/step12/
# ============================================================================
!pip install -q lightgbm imbalanced-learn

import os, json, time, shutil, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
from google.colab import drive

warnings.filterwarnings("ignore")
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/himachal_project"
V3, S7, S10, S11 = (f"{BASE}/data/model_ready_v3", f"{BASE}/results/step7", f"{BASE}/results/step10",
                    f"{BASE}/results/step11")
OUT = f"{BASE}/results/step12"
PRED, PRED10 = f"{OUT}/preds", f"{S10}/preds"
LOCAL, LOCAL_PRED = "/content/local_v3", "/content/local_v3/preds12"

# ---------------------------------------------------------------- settings
SEEDS = [42, 1, 2, 3, 4]               # Step 10 used these five; [42, 1, 2] is faster
MAIN_SEED = 42
STEP11_GENS = ["gaussian_copula", "tvae", "ctgan", "copulagan", "cwgan"]
SMOTE_FAMILY = ["smote", "borderline_smote", "adasyn"]
METHODS = (["none"] + SMOTE_FAMILY + [f"{g}_{v}" for g in STEP11_GENS for v in ("plain", "tail")]
           + ["diffusion_plain", "diffusion_tail"])
FAMILY = {"none": "none", "smote": "interpolation", "borderline_smote": "interpolation", "adasyn": "interpolation",
          "gaussian_copula": "statistical", "tvae": "VAE", "ctgan": "GAN", "copulagan": "GAN", "cwgan": "GAN",
          "diffusion": "diffusion"}
STEP10_KEY = {"none": "lgbm/none", "diffusion_plain": "lgbm/ddpm_emp@1", "diffusion_tail": "lgbm/ddpm_evt_tail@1"}
SMOTE_NEG = 300_000                    # normal days used as neighbours by Borderline-SMOTE / ADASYN
BOOT = 200
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, max_bin=127,
                  n_estimators=1000, n_jobs=-1, verbose=-1)
# ----------------------------------------------------------------
rng = np.random.default_rng(MAIN_SEED)


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


def copy_local(src, dst_dir=LOCAL):
    dst = f"{dst_dir}/{os.path.basename(src)}"
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


os.makedirs(LOCAL_PRED, exist_ok=True)
for f in ("train.parquet", "val.parquet", "test.parquet", "features.json"):
    copy_local(f"{V3}/{f}")
SYN_SRC = {"diffusion_plain": f"{S7}/synthetic_ddpm_emp.parquet", "diffusion_tail": f"{S7}/synthetic_ddpm_evt_tail.parquet"}
for g in STEP11_GENS:
    for v in ("plain", "tail"):
        SYN_SRC[f"{g}_{v}"] = f"{S11}/synthetic_{g}_{v}.parquet"
for m, p in SYN_SRC.items():
    drive_ok()
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing: run Step 7 / Step 11 first")
drive_ok()
os.makedirs(PRED, exist_ok=True)
LINES, T0 = [], time.time()
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def progress(msg):
    print(f"   ... [{(time.time() - T0) / 60:4.1f} min] {msg}", flush=True)


def save_summary():
    drive_ok()
    txt = "\n".join(LINES) + "\n"
    for p in (f"{OUT}/step12_summary.txt", f"{BASE}/step12_summary.txt"):
        open(p, "w").write(txt)


def section(t):
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


out(f"STEP 12  {time.strftime('%Y-%m-%d %H:%M')}  seeds={SEEDS}")

# ================================================================ 1. data
section("1. DATA")
FEATS = json.load(open(f"{LOCAL}/features.json"))["FEATS"]
keep = FEATS + ["y", "z", "onset", "month"]
tr = pd.read_parquet(f"{LOCAL}/train.parquet", columns=keep)
va = pd.read_parquet(f"{LOCAL}/val.parquet", columns=keep)
te = pd.read_parquet(f"{LOCAL}/test.parquet", columns=keep)
ytr, yva, yte = tr.y.values.astype("int8"), va.y.values.astype("int8"), te.y.values.astype("int8")
Xtr = np.nan_to_num(tr[FEATS].values.astype("float32"))
Xva = np.nan_to_num(va[FEATS].values.astype("float32"))
Xte = np.nan_to_num(te[FEATS].values.astype("float32"))
n_pos = int(ytr.sum())
del tr


def subsets(df, y):
    z = df.z.values
    cut = np.nanquantile(z[y == 1], 0.9)
    return {"pr_auc": np.ones(len(y), bool), "onset_pr_auc": df.onset.values == 1,
            "severe_pr_auc": (y == 0) | (z >= cut)}, cut


VSETS, vcut = subsets(va, yva)
TSETS, tcut = subsets(te, yte)
for nm, y in (("train", ytr), ("val", yva), ("test", yte)):
    out(f"{nm:<5} rows {len(y):>9,}  extremes {int(y.sum()):>7,}  ({100 * y.mean():.2f}%)")
out(f"each method adds {n_pos:,} synthetic extremes (1 per real extreme), drawn again for every seed")
progress("data loaded")

# ================================================================ 2. synthetic sets
section("2. SYNTHETIC SETS (SMOTE family made here; the rest from Steps 7 and 11)")


def make_smote_family():
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
    pos_i = np.where(ytr == 1)[0]
    neg_i = rng.choice(np.where(ytr == 0)[0], SMOTE_NEG, replace=False)
    Xs = np.r_[Xtr[pos_i], Xtr[neg_i]].astype("float64")
    ys = np.r_[np.ones(len(pos_i), int), np.zeros(len(neg_i), int)]
    want = (4 * n_pos) / SMOTE_NEG                      # 3 x n_pos new extremes on top of the real ones
    makers = {"smote": SMOTE(sampling_strategy=want, k_neighbors=5, random_state=MAIN_SEED),
              "borderline_smote": BorderlineSMOTE(sampling_strategy=want, k_neighbors=5, m_neighbors=10,
                                                  random_state=MAIN_SEED),
              "adasyn": ADASYN(sampling_strategy=want, n_neighbors=5, random_state=MAIN_SEED)}
    for name, mk in makers.items():
        p = f"{OUT}/synthetic_{name}.parquet"
        drive_ok()
        if os.path.exists(p):
            continue
        t1 = time.time()
        Xr, yr = mk.fit_resample(Xs, ys)
        new = Xr[len(Xs):][yr[len(Xs):] == 1]
        pd.DataFrame(new.astype("float32"), columns=FEATS).to_parquet(p, index=False)
        progress(f"{name}: {len(new):,} rows in {(time.time() - t1) / 60:.1f} min")


make_smote_family()
for m in SMOTE_FAMILY:
    SYN_SRC[m] = f"{OUT}/synthetic_{m}.parquet"
SYN = {}
for m, p in SYN_SRC.items():
    SYN[m] = np.nan_to_num(pd.read_parquet(copy_local(p), columns=FEATS)[FEATS].values.astype("float32"))
out("rows available per method: " + ", ".join(f"{m} {len(v):,}" for m, v in SYN.items()))
progress("synthetic sets loaded")


def training_set(m, seed):
    """Same recipe as Steps 9-10: real training data + n_pos synthetic extremes drawn with this seed."""
    if m == "none":
        return Xtr, ytr
    Xg = SYN[m]
    Xg = Xg[np.random.default_rng(seed).choice(len(Xg), n_pos, replace=len(Xg) < n_pos)]
    return np.vstack([Xtr, Xg]), np.concatenate([ytr, np.ones(len(Xg), "int8")])


# ================================================================ 3. classifier and bookkeeping (as Step 10)
MONO = [1 if f in ("dist_l0", "dist_std_l0") else 0 for f in FEATS]


def fast_ap(y, s):
    o = np.argsort(-s, kind="stable")
    y = y[o]
    tp = np.cumsum(y)
    return float(np.sum((tp / np.arange(1, len(y) + 1))[y == 1]) / max(tp[-1], 1))


def fit_lgbm(X, y, seed):
    m = lgb.LGBMClassifier(**dict(LGB_PARAMS, random_state=seed, monotone_constraints=MONO))
    m.fit(X, y, eval_set=[(Xva, yva)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(60, verbose=False)])
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], f"{m.best_iteration_} trees"


def pick_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return th[int(np.nanargmax(f1[:-1]))]


def fname(key):
    return key.replace("/", "__").replace("@", "_at_").replace("#", "_").replace("[", "(").replace("]", ")").replace("+", "_plus_") + ".npz"


res_path = f"{OUT}/results.csv"
drive_ok()
done = pd.read_csv(res_path) if os.path.exists(res_path) else pd.DataFrame(columns=["key"])


def has(key):
    if key not in set(done.key):
        return False
    drive_ok()
    return os.path.exists(f"{LOCAL_PRED}/{fname(key)}") or os.path.exists(f"{PRED}/{fname(key)}")


def load_pred(key):
    loc = f"{LOCAL_PRED}/{fname(key)}"
    if not os.path.exists(loc):
        drive_ok()
        shutil.copy(f"{PRED}/{fname(key)}", loc)
    d = np.load(loc)
    return d["va"], d["te"]


def record(key, pv, pt, note, quiet=False):
    global done
    r = {"key": key}
    for s, m_ in VSETS.items():
        r[f"val_{s}"] = round(fast_ap(yva[m_], pv[m_]), 4)
    r["val_score"] = round((r["val_pr_auc"] + r["val_onset_pr_auc"] + r["val_severe_pr_auc"]) / 3, 4)
    for s, m_ in TSETS.items():
        r[s] = round(fast_ap(yte[m_], pt[m_]), 4)
    thr = pick_threshold(yva, pv)
    yhat = pt >= thr
    tp = int((yhat & (yte == 1)).sum()); fp = int((yhat & (yte == 0)).sum()); fn = int((~yhat & (yte == 1)).sum())
    r.update(precision=round(tp / max(tp + fp, 1), 3), recall=round(tp / max(tp + fn, 1), 3),
             csi=round(tp / max(tp + fp + fn, 1), 3), note=note)
    np.savez(f"{LOCAL_PRED}/{fname(key)}", va=pv.astype("float32"), te=pt.astype("float32"))
    drive_ok()
    shutil.copy(f"{LOCAL_PRED}/{fname(key)}", f"{PRED}/{fname(key)}")
    done = pd.concat([done[done.key != key], pd.DataFrame([r])], ignore_index=True)
    done.to_csv(res_path, index=False)
    if not quiet:
        progress(f"{key}: test PR-AUC {r['pr_auc']}  onset {r['onset_pr_auc']}  severe {r['severe_pr_auc']}")


def skey(m, seed):
    return f"{m}#s{seed}"


def get_single(m, seed):
    key = skey(m, seed)
    if has(key):
        return key
    if m in STEP10_KEY:                                 # same recipe in Step 10: reuse, do not retrain
        k10 = STEP10_KEY[m] + ("" if seed == MAIN_SEED else f"#s{seed}")
        p10 = f"{PRED10}/{fname(k10)}"
        drive_ok()
        if os.path.exists(p10):
            d = np.load(copy_local(p10, LOCAL_PRED))
            record(key, d["va"], d["te"], "reused from step 10")
            return key
    progress(f"training {key}")
    X, y = training_set(m, seed)
    pv, pt, note = fit_lgbm(X, y, seed)
    record(key, pv, pt, note)
    del X, y
    return key


_X, _y = training_set("smote", 3)
assert len(_y) == len(ytr) + n_pos and _y[len(ytr):].all(), "training_set check failed"
del _X, _y
out("self-check OK")

# ================================================================ 4. train
section(f"3. TRAIN LIGHTGBM: {len(METHODS)} methods x {len(SEEDS)} seeds")
if not has("persistence"):
    record("persistence", va.dist_l0.values, te.dist_l0.values, "no model")
for seed in SEEDS:                                     # seed-major: a disconnect still leaves complete seeds
    for m in METHODS:
        get_single(m, seed)
    save_summary()
D = done.set_index("key")

# ================================================================ 5. per-seed results
section("4. PER-SEED RESULTS (test 2016-2025): mean +- sd, and wins over real-data-only (out of seeds)")
S3 = ("pr_auc", "onset_pr_auc", "severe_pr_auc")
ns = len(SEEDS)
rows = []
for m in METHODS:
    r = {"method": m}
    for s in S3:
        v = np.array([D.loc[skey(m, sd), s] for sd in SEEDS], float)
        r[f"{s}_mean"], r[f"{s}_sd"] = round(v.mean(), 4), round(v.std(ddof=1), 4)
        if m != "none":
            d = v - np.array([D.loc[skey("none", sd), s] for sd in SEEDS], float)
            r[f"{s}_wins_vs_none"] = f"{int((d > 0).sum())}/{ns}"
    rows.append(r)
per_seed = pd.DataFrame(rows)
per_seed.to_csv(f"{OUT}/per_seed_summary.csv", index=False)
show = per_seed[["method"] + [c for s in S3 for c in (f"{s}_mean", f"{s}_sd", f"{s}_wins_vs_none")]]
out(show.to_string(index=False))

out(f"\nEVT tail focus vs the same generator as it is (same seed; wins out of {ns}):")
rows = []
for g in STEP11_GENS + ["diffusion"]:
    a, b = f"{g}_tail", f"{g}_plain"
    r = {"generator": g}
    for s in S3:
        d = np.array([D.loc[skey(a, sd), s] - D.loc[skey(b, sd), s] for sd in SEEDS], float)
        r[f"{s}_diff"] = round(float(d.mean()), 4)
        r[f"{s}_wins"] = f"{int((d > 0).sum())}/{ns}"
    rows.append(r)
evt_eff = pd.DataFrame(rows)
evt_eff.to_csv(f"{OUT}/evt_tail_vs_plain.csv", index=False)
out(evt_eff.to_string(index=False))
save_summary()

# ================================================================ 6. seed-averaged models
section("5. SEED-AVERAGED MODELS (mean probability of the seeds)")
ENS = {}
for m in METHODS:
    key = f"ens[{m}]"
    ENS[m] = key
    if not has(key):
        P = [load_pred(skey(m, s)) for s in SEEDS]
        record(key, np.mean([p[0] for p in P], 0), np.mean([p[1] for p in P], 0), f"mean of {ns} seeds", quiet=True)
D = done.set_index("key")
out(D.loc[["persistence"] + list(ENS.values()), ["val_score", "pr_auc", "onset_pr_auc", "severe_pr_auc",
                                                  "precision", "recall", "csi"]].to_string())

keys = ["persistence"] + list(ENS.values())
P = {k: (te.dist_l0.values if k == "persistence" else load_pred(k)[1]) for k in keys}
um, inv = np.unique(te.month.values, return_inverse=True)
order = {(k, s): np.argsort(-P[k][m_], kind="stable") for k in keys for s, m_ in TSETS.items()}


def wap(y, w, o):
    y, w = y[o], w[o]
    tpw = np.cumsum(w * y)
    allw = np.cumsum(w)
    return float(np.sum((w * y)[w > 0] * (tpw / np.maximum(allw, 1e-12))[w > 0]) / max(tpw[-1], 1e-12))


boot = {(k, s): [] for k in keys for s in TSETS}
for b in range(BOOT):
    cnt = np.bincount(rng.integers(0, len(um), len(um)), minlength=len(um)).astype("float64")
    w_all = cnt[inv]
    for s, m_ in TSETS.items():
        ys, ws = yte[m_], w_all[m_]
        for k in keys:
            boot[(k, s)].append(wap(ys, ws, order[(k, s)]))
    if (b + 1) % 50 == 0:
        progress(f"bootstrap {b + 1}/{BOOT}")
del order

cmp = [(ENS[m], ENS["none"]) for m in METHODS if m != "none"]
cmp += [(ENS[f"{g}_tail"], ENS[f"{g}_plain"]) for g in STEP11_GENS + ["diffusion"]]
cmp += [(ENS[m], ENS["diffusion_tail"]) for m in METHODS if m not in ("none", "diffusion_tail")]
cmp += [(k, "persistence") for k in ENS.values()]
pairs = []
for a, b in dict.fromkeys(cmp):
    r = {"model": a, "vs": b}
    for s in TSETS:
        d = np.array(boot[(a, s)]) - np.array(boot[(b, s)])
        r[f"{s}_diff"] = round(float(d.mean()), 4)
        r[f"{s}_share_better"] = round(float(np.mean(d > 0)), 2)
    pairs.append(r)
pairs = pd.DataFrame(pairs)
pairs.to_csv(f"{OUT}/paired_tests.csv", index=False)
out(f"\nPaired comparisons ({BOOT} resamples of whole test months). share_better >= 0.95 = convincing win, "
    "<= 0.05 = convincing loss:")
out(pairs.to_string(index=False))
ci_rows = []
for k in keys:
    r = {"key": k}
    for s in TSETS:
        v = np.array(boot[(k, s)])
        r[s] = D.loc[k, s]
        r[f"{s}_ci"] = f"[{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]"
    ci_rows.append(r)
pd.DataFrame(ci_rows).to_csv(f"{OUT}/confidence_intervals.csv", index=False)
save_summary()

# ================================================================ 7. quality of the synthetic sets
section("6. QUALITY OF EACH SYNTHETIC SET")
pos_X = Xtr[ytr == 1]
mu, sd = pos_X.mean(0), pos_X.std(0) + 1e-6
ref_i = rng.permutation(len(pos_X))
REFX, HELD = (pos_X[ref_i[:int(0.8 * len(pos_X))]] - mu) / sd, (pos_X[ref_i[int(0.8 * len(pos_X)):]] - mu) / sd
nn_ = NearestNeighbors(n_neighbors=1).fit(REFX)
NQ = 5000
d_real = np.median(nn_.kneighbors(HELD[:NQ])[0][:, 0])
qual = []
for m in METHODS:
    if m == "none":
        continue
    nq = min(20000, len(SYN[m]), len(pos_X))
    S = SYN[m][rng.choice(len(SYN[m]), nq, replace=False)]
    R = pos_X[rng.choice(len(pos_X), len(S), replace=False)]
    Xd, yd = np.r_[R, S], np.r_[np.zeros(len(R)), np.ones(len(S))]
    oof = np.zeros(len(yd))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=MAIN_SEED).split(Xd, yd):
        c = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, verbose=-1, random_state=MAIN_SEED)
        oof[b] = c.fit(Xd[a], yd[a]).predict_proba(Xd[b])[:, 1]
    d_syn = np.median(nn_.kneighbors((S[:NQ] - mu) / sd)[0][:, 0])
    qual.append(dict(method=m, detection_auc=round(roc_auc_score(yd, oof), 3),
                     closeness_ratio=round(float(d_syn / d_real), 2)))
qual = pd.DataFrame(qual)
out(qual.to_string(index=False))
out("""detection_auc: can a classifier tell synthetic extremes from real ones? 0.5 = indistinguishable, 1.0 = trivially.
closeness_ratio: median distance from a synthetic row to its nearest real extreme, divided by the same for unseen
  REAL extremes. About 1 = as close as real data is to itself; far below 1 = copying real rows; far above 1 = unlike
  any real day.""")
progress("quality checks done")

# ================================================================ 8. ranking
section("7. RANKING (seed-averaged models, ordered by VALIDATION score; test scores shown)")
s11 = pd.read_csv(copy_local(f"{S11}/generator_stats.csv", LOCAL)).set_index("generator")
rank = []
for m in METHODS:
    k = ENS[m]
    g = m.rsplit("_", 1)[0] if m.endswith(("_plain", "_tail")) else m
    r = dict(method=m, family=FAMILY.get(g, "none"), evt_tail="yes" if m.endswith("_tail") else "no",
             val_score=D.loc[k, "val_score"], pr_auc=D.loc[k, "pr_auc"], onset=D.loc[k, "onset_pr_auc"],
             severe=D.loc[k, "severe_pr_auc"],
             severe_seed_wins_vs_none="-" if m == "none" else per_seed.set_index("method").loc[m, "severe_pr_auc_wins_vs_none"])
    if m != "none":
        pr_ = pairs[(pairs.model == k) & (pairs.vs == ENS["none"])].iloc[0]
        r["severe_vs_none_share"] = pr_.severe_pr_auc_share_better
        q = qual.set_index("method").loc[m]
        r["detection_auc"], r["closeness"] = q.detection_auc, q.closeness_ratio
    if g in s11.index:
        r["raw_physics_breaking_pct"] = s11.loc[g, "raw_rule_breaking_pct"]
        r["train_min"] = s11.loc[g, "train_minutes"]
    rank.append(r)
rank = pd.DataFrame(rank).sort_values("val_score", ascending=False)
rank.to_csv(f"{OUT}/ranking.csv", index=False)
out(rank.to_string(index=False))
pr_ = D.loc["persistence"]
out(f"\npersistence (no model): PR-AUC {pr_.pr_auc}  onset {pr_.onset_pr_auc}  severe {pr_.severe_pr_auc}")
best = rank.iloc[0]
out(f"best on validation: {best.method}  (test: PR-AUC {best.pr_auc}, onset {best.onset}, severe {best.severe})")

# ---------------------------------------------------------------- charts
COL = {"none": "#8A94A6", "interpolation": "#6B8E23", "statistical": "#B07AA1", "VAE": "#E8A33D",
       "GAN": "#2F6DB5", "diffusion": "#C8461E"}
fig, axs = plt.subplots(3, 1, figsize=(15, 13), sharex=True)
for ax, (s, lab) in zip(axs, [("severe_pr_auc", "severe PR-AUC"), ("pr_auc", "PR-AUC"), ("onset_pr_auc", "onset PR-AUC")]):
    for i, m in enumerate(METHODS):
        g = m.rsplit("_", 1)[0] if m.endswith(("_plain", "_tail")) else m
        v = [D.loc[skey(m, sd), s] for sd in SEEDS]
        c = COL[FAMILY.get(g, "none")]
        ax.scatter([i] * len(v), v, s=22, color=c, alpha=0.75, marker="^" if m.endswith("_tail") else "o")
        ax.plot([i - 0.3, i + 0.3], [np.mean(v)] * 2, color=c, lw=2.5)
    ax.axhline(D.loc["persistence", s], color="grey", ls=":", lw=1.2, label="persistence")
    ax.axhline(np.mean([D.loc[skey("none", sd), s] for sd in SEEDS]), color="k", ls="--", lw=1, label="real data only")
    ax.set_ylabel(lab)
    ax.legend(fontsize=8, loc="lower right")
axs[0].set_title("LightGBM test scores per training seed (dots; triangles = EVT tail focus), bar = mean")
axs[-1].set_xticks(range(len(METHODS)))
axs[-1].set_xticklabels(METHODS, rotation=45, ha="right", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/methods_by_seed.png", dpi=120); plt.close()

q = rank.dropna(subset=["detection_auc"])
fig, ax = plt.subplots(figsize=(9, 6))
for _, r in q.iterrows():
    ax.scatter(r.detection_auc, r.severe, s=60, color=COL[r.family], marker="^" if r.evt_tail == "yes" else "o")
    ax.annotate(r.method, (r.detection_auc, r.severe), fontsize=7, xytext=(4, 3), textcoords="offset points")
ax.axhline(D.loc[ENS["none"], "severe_pr_auc"], color="k", ls="--", lw=1, label="real data only")
ax.set_xlabel("detection AUC (lower = harder to tell from real extremes)")
ax.set_ylabel("severe PR-AUC, seed-averaged model (test)")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/realism_vs_benefit.png", dpi=120); plt.close()

out(f"\nfiles in {OUT}")
out(f"END  (took {(time.time() - T0) / 60:.1f} min)")
save_summary()
print(f"\nSummary saved to {BASE}/step12_summary.txt")
