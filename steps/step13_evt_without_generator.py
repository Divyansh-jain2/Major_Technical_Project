# ============================================================================
# STEP 13: CONTROL - DO WE NEED A GENERATOR, OR JUST THE EVT TAIL EMPHASIS?
# Paste into ONE Colab cell. CPU runtime is fine. ~35-45 min.
# Re-run the same cell after a disconnect: finished models are skipped.
#
# Step 12: the best methods add synthetic extremes chosen with the EVT tail focus, even when the
# synthetic days are unrealistic. So is it the GENERATOR or the EVT EMPHASIS that helps?
# Same fixed LightGBM, same 5 seeds, same 1 extra extreme per real extreme. New methods, no generator:
#   real_dup_plain   random oversampling: copies of random real extremes
#   real_dup_tail    copies of real extremes chosen with the same EVT tail target as the generators
#                    (60% of the added rows from the top 20% of the tail, season shares as real)
#   real_weight_tail no copies: real extremes get a training weight that adds the same EVT tail emphasis
# Compared with (models reused from Step 12, not retrained): none, ctgan_tail, tvae_tail,
# gaussian_copula_tail, diffusion_tail, and persistence.
#
# Reads (never changes): data/model_ready_v3/*, results/step3b/evt_fits_std.csv, results/step12/preds/*
# Writes ONLY to:        results/step13/
# ============================================================================
!pip install -q lightgbm

import os, json, time, shutil, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import lightgbm as lgb
from google.colab import drive

warnings.filterwarnings("ignore")
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/himachal_project"
V3, S12, OUT = f"{BASE}/data/model_ready_v3", f"{BASE}/results/step12", f"{BASE}/results/step13"
PRED, PRED12 = f"{OUT}/preds", f"{S12}/preds"
LOCAL, LOCAL_PRED = "/content/local_v3", "/content/local_v3/preds13"

# ---------------------------------------------------------------- settings
SEEDS = [42, 1, 2, 3, 4]
MAIN_SEED = 42
NEW = ["real_dup_plain", "real_dup_tail", "real_weight_tail"]
REUSED = ["none", "ctgan_tail", "tvae_tail", "gaussian_copula_tail", "diffusion_tail"]
METHODS = REUSED[:1] + NEW + REUSED[1:]
TAIL_FROM, TAIL_SHARE = 0.8, 0.6        # same EVT tail target as Steps 7 and 11
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
drive_ok()
os.makedirs(PRED, exist_ok=True)
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
    for p in (f"{OUT}/step13_summary.txt", f"{BASE}/step13_summary.txt"):
        open(p, "w").write(txt)


def section(t):
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


out(f"STEP 13  {time.strftime('%Y-%m-%d %H:%M')}  seeds={SEEDS}")

# ================================================================ 1. data
section("1. DATA")
FEATS = json.load(open(f"{LOCAL}/features.json"))["FEATS"]
keep = FEATS + ["y", "z", "onset", "month", "season"]
tr = pd.read_parquet(f"{LOCAL}/train.parquet", columns=keep)
va = pd.read_parquet(f"{LOCAL}/val.parquet", columns=FEATS + ["y", "z", "onset", "month"])
te = pd.read_parquet(f"{LOCAL}/test.parquet", columns=FEATS + ["y", "z", "onset", "month"])
ytr, yva, yte = tr.y.values.astype("int8"), va.y.values.astype("int8"), te.y.values.astype("int8")
Xtr = np.nan_to_num(tr[FEATS].values.astype("float32"))
Xva = np.nan_to_num(va[FEATS].values.astype("float32"))
Xte = np.nan_to_num(te[FEATS].values.astype("float32"))
pos_i = np.where(ytr == 1)[0]
n_pos = len(pos_i)
pos_season = tr.season.values[pos_i].astype(int)
pos_z = tr.z.values[pos_i]
del tr


def subsets(df, y):
    z = df.z.values
    cut = np.nanquantile(z[y == 1], 0.9)
    return {"pr_auc": np.ones(len(y), bool), "onset_pr_auc": df.onset.values == 1,
            "severe_pr_auc": (y == 0) | (z >= cut)}, cut


VSETS, _ = subsets(va, yva)
TSETS, _ = subsets(te, yte)
for nm, y in (("train", ytr), ("val", yva), ("test", yte)):
    out(f"{nm:<5} rows {len(y):>9,}  extremes {int(y.sum()):>7,}  ({100 * y.mean():.2f}%)")

# tail position u of every real training extreme: its rank among real extremes of its season
# (the fitted GPD matches these severities closely, KS ~0.003 in Step 7, so this is its GPD probability)
pos_u = np.zeros(n_pos)
for k in np.unique(pos_season):
    m = pos_season == k
    pos_u[m] = (pd.Series(pos_z[m]).rank(method="average").values) / (m.sum() + 1)
season_share = pd.Series(pos_season).value_counts(normalize=True).sort_index()
EDGES = np.linspace(0, 1, 21)
MASS = np.where(EDGES[:-1] >= TAIL_FROM - 1e-9, TAIL_SHARE / 4, (1 - TAIL_SHARE) / 16)
pos_bin = np.clip(np.digitize(pos_u, EDGES) - 1, 0, 19)
out(f"{n_pos:,} real extremes; {int((pos_u >= TAIL_FROM).sum()):,} of them in the top 20% of their season's tail")

# ================================================================ 2. the three no-generator methods
def dup_tail_rows(seed):
    """n_pos copies of real extremes following the EVT tail target (copies repeat the rare severe ones)."""
    r = np.random.default_rng(seed)
    take = []
    for k, share in season_share.items():
        nk = int(round(n_pos * share))
        for j in range(20):
            m = int(round(nk * MASS[j]))
            cand = np.where((pos_season == k) & (pos_bin == j))[0]
            if m == 0 or len(cand) == 0:
                continue
            take.append(r.choice(cand, m, replace=len(cand) < m))
    return pos_i[np.concatenate(take)]


TAIL_W = np.ones(len(ytr), "float32")              # weights giving the same expected emphasis, no copies
for k, share in season_share.items():
    nk = n_pos * share
    for j in range(20):
        cand = np.where((pos_season == k) & (pos_bin == j))[0]
        if len(cand):
            TAIL_W[pos_i[cand]] += nk * MASS[j] / len(cand)
out(f"real_weight_tail: extreme-day weights from {TAIL_W[pos_i].min():.2f} to {TAIL_W[pos_i].max():.2f} "
    f"(total extra weight {TAIL_W[pos_i].sum() - n_pos:,.0f} = 1 x real extremes)")


def training_set(m, seed):
    if m == "real_dup_plain":
        add = pos_i[np.random.default_rng(seed).integers(0, n_pos, n_pos)]
    elif m == "real_dup_tail":
        add = dup_tail_rows(seed)
    else:
        return Xtr, ytr, (TAIL_W if m == "real_weight_tail" else None)
    return np.vstack([Xtr, Xtr[add]]), np.concatenate([ytr, ytr[add]]), None


# ================================================================ 3. classifier and bookkeeping (as Step 12)
MONO = [1 if f in ("dist_l0", "dist_std_l0") else 0 for f in FEATS]


def fast_ap(y, s):
    o = np.argsort(-s, kind="stable")
    y = y[o]
    tp = np.cumsum(y)
    return float(np.sum((tp / np.arange(1, len(y) + 1))[y == 1]) / max(tp[-1], 1))


def fit_lgbm(X, y, w, seed):
    m = lgb.LGBMClassifier(**dict(LGB_PARAMS, random_state=seed, monotone_constraints=MONO))
    m.fit(X, y, sample_weight=w, eval_set=[(Xva, yva)], eval_metric="average_precision",
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
    if m in REUSED:
        p12 = f"{PRED12}/{fname(key)}"
        drive_ok()
        if not os.path.exists(p12):
            raise FileNotFoundError(f"{p12} missing: Step 12 must finish first")
        d = np.load(copy_local(p12, LOCAL_PRED))
        record(key, d["va"], d["te"], "reused from step 12", quiet=True)
        return key
    progress(f"training {key}")
    X, y, w = training_set(m, seed)
    pv, pt, note = fit_lgbm(X, y, w, seed)
    record(key, pv, pt, note)
    del X, y
    return key


_X, _y, _ = training_set("real_dup_tail", 3)
assert len(_y) == len(ytr) + n_pos or abs(len(_y) - len(ytr) - n_pos) < 50, "training_set check failed"
assert _y[len(ytr):].all()
del _X, _y
out("self-check OK")

# ================================================================ 4. train
section(f"2. TRAIN: {len(NEW)} new methods x {len(SEEDS)} seeds (+ reused Step 12 models)")
if not has("persistence"):
    record("persistence", va.dist_l0.values, te.dist_l0.values, "no model", quiet=True)
for seed in SEEDS:
    for m in METHODS:
        get_single(m, seed)
    save_summary()
D = done.set_index("key")

# ================================================================ 5. per-seed
section("3. PER-SEED RESULTS (test 2016-2025): mean +- sd")
S3 = ("pr_auc", "onset_pr_auc", "severe_pr_auc")
ns = len(SEEDS)
rows = []
for m in METHODS:
    r = {"method": m}
    for s in S3:
        v = np.array([D.loc[skey(m, sd), s] for sd in SEEDS], float)
        r[s] = f"{v.mean():.4f} +- {v.std(ddof=1):.4f}"
    rows.append(r)
out(pd.DataFrame(rows).to_string(index=False))

out(f"\nSame-seed differences (wins out of {ns}):")
CMP_SEED = [("real_dup_tail", "real_dup_plain"), ("real_dup_plain", "none"), ("real_dup_tail", "none"),
            ("real_weight_tail", "none"), ("ctgan_tail", "real_dup_tail"), ("tvae_tail", "real_dup_tail"),
            ("gaussian_copula_tail", "real_dup_tail"), ("diffusion_tail", "real_dup_tail"),
            ("ctgan_tail", "real_weight_tail"), ("tvae_tail", "real_weight_tail")]
rows = []
for a, b in CMP_SEED:
    r = {"model": a, "vs": b}
    for s in S3:
        d = np.array([D.loc[skey(a, sd), s] - D.loc[skey(b, sd), s] for sd in SEEDS], float)
        r[f"{s}_diff"] = round(float(d.mean()), 4)
        r[f"{s}_wins"] = f"{int((d > 0).sum())}/{ns}"
    rows.append(r)
seed_cmp = pd.DataFrame(rows)
seed_cmp.to_csv(f"{OUT}/per_seed_comparisons.csv", index=False)
out(seed_cmp.to_string(index=False))
save_summary()

# ================================================================ 6. seed-averaged models + tests
section("4. SEED-AVERAGED MODELS, 95% INTERVALS AND PAIRED TESTS")
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

cmp = [(ENS[a], ENS[b]) for a, b in CMP_SEED] + [(ENS[m], "persistence") for m in METHODS]
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

# ---------------------------------------------------------------- chart
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, (s, lab) in zip(axs, [("severe_pr_auc", "severe PR-AUC"), ("pr_auc", "PR-AUC"), ("onset_pr_auc", "onset PR-AUC")]):
    for i, m in enumerate(METHODS):
        v = [D.loc[skey(m, sd), s] for sd in SEEDS]
        c = "#8A94A6" if m == "none" else ("#6B8E23" if m.startswith("real_") else "#C8461E")
        ax.scatter([i] * len(v), v, s=20, color=c, alpha=0.8)
        ax.plot([i - 0.3, i + 0.3], [np.mean(v)] * 2, color=c, lw=2.5)
    ax.axhline(D.loc["persistence", s], color="grey", ls=":", lw=1.2, label="persistence")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"{lab} (green: no generator, orange: generator)", fontsize=9)
    ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/generator_vs_reweighting.png", dpi=120); plt.close()

out(f"\nfiles in {OUT}")
out(f"END  (took {(time.time() - T0) / 60:.1f} min)")
save_summary()
print(f"\nSummary saved to {BASE}/step13_summary.txt")
