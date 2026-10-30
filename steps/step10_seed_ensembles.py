# ============================================================================
# STEP 10: 5-SEED CHECK AND SEED-AVERAGED (ENSEMBLE) MODELS - LightGBM, CPU is fine
# Paste into ONE Colab cell. CPU runtime OK. ~35-45 min.
# Re-run the same cell after a disconnect: finished models are skipped.
#
# Why: in Step 9, the EVT vs no-EVT gain on severe days changed sign between training seeds.
#   A. Train every setting with 5 seeds (42, 1, 2, 3, 4). The seed changes LightGBM's
#      randomness AND which synthetic rows are picked. Models already trained in Step 9
#      are reused, not retrained.
#   B. Per-seed table: does EVT beat no-EVT in most seeds (k out of 5)?
#   C. Seed-averaged ensembles: average the 5 models of each setting (less luck, usually
#      a bit better), then the same month-resampling tests as before.
#   D. Blends of the plain ensemble with each augmented ensemble (weight chosen on validation).
# Settings: none | ddpm_emp, ddpm_evt, ddpm_evt_tail at 1 and at full (8.7) synthetic per real extreme.
# Everything is chosen on validation (2011-2015); the test years 2016-2025 only score.
#
# Reads (never changes): data/model_ready_v3/*, results/step7/synthetic_*.parquet, results/step9/preds/*
# Writes ONLY to:        results/step10/
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
V3, S7, OUT = f"{BASE}/data/model_ready_v3", f"{BASE}/results/step7", f"{BASE}/results/step10"
PRED, PRED9 = f"{OUT}/preds", f"{BASE}/results/step9/preds"
LOCAL, LOCAL_PRED = "/content/local_v3", "/content/local_v3/preds10"

# ---------------------------------------------------------------- settings
GEN = ["ddpm_emp", "ddpm_evt", "ddpm_evt_tail"]
CONFIGS = [("none", "-")] + [(m, r) for r in ("1", "full") for m in GEN]
SEEDS = [42, 1, 2, 3, 4]
MAIN_SEED = 42
BLEND_W = np.round(np.linspace(0, 1, 11), 2)
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
syn_paths = {}
for m in GEN:
    p = f"{S7}/synthetic_{m}.parquet"
    drive_ok()
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing: run Step 7 first")
    syn_paths[m] = copy_local(p)
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
    for p in (f"{OUT}/step10_summary.txt", f"{BASE}/step10_summary.txt"):
        open(p, "w").write(txt)


def section(t):
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


out(f"STEP 10  {time.strftime('%Y-%m-%d %H:%M')}")

# ================================================================ 1. data (same as Step 9)
section("1. DATA")
FEATS = json.load(open(f"{LOCAL}/features.json"))["FEATS"]
keep = FEATS + ["y", "z", "onset", "month", "season"]
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
out(f"severe cut: val z >= {vcut:.2f}, test z >= {tcut:.2f}")
SYN = {m: np.nan_to_num(pd.read_parquet(syn_paths[m], columns=FEATS)[FEATS].values.astype("float32")) for m in GEN}
progress("data loaded")


def training_set(m, r, seed):                      # identical to Step 9, so Step 9 models can be reused
    if m == "none":
        return Xtr, ytr
    Xg = SYN[m]
    if r != "full":
        n = int(round(float(r) * n_pos))
        Xg = Xg[np.random.default_rng(seed).choice(len(Xg), n, replace=False)]
    return np.vstack([Xtr, Xg]), np.concatenate([ytr, np.ones(len(Xg), "int8")])


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


def logit(p):
    p = np.clip(p.astype("float64"), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


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
        progress(f"{key}: val score {r['val_score']}  | test PR-AUC {r['pr_auc']}  onset {r['onset_pr_auc']}  severe {r['severe_pr_auc']}")


def cfg_name(m, r):
    return "lgbm/none" if m == "none" else f"lgbm/{m}@{r}"


def single_key(m, r, seed):
    return cfg_name(m, r) + ("" if seed == MAIN_SEED else f"#s{seed}")


def get_single(m, r, seed):
    key = single_key(m, r, seed)
    if has(key):
        return key
    p9 = f"{PRED9}/{fname(key)}"
    drive_ok()
    if os.path.exists(p9):                          # trained in Step 9 with the same recipe
        shutil.copy(p9, f"{LOCAL_PRED}/{fname(key)}")
        d = np.load(f"{LOCAL_PRED}/{fname(key)}")
        record(key, d["va"], d["te"], "reused from step 9")
        return key
    progress(f"training {key}")
    X, y = training_set(m, r, seed)
    pv, pt, note = fit_lgbm(X, y, seed)
    if len(y) > len(ytr):
        note += f", +{len(y) - len(ytr):,} synthetic"
    record(key, pv, pt, note)
    del X, y
    return key


# ---------------------------------------------------------------- self-check
_X, _y = training_set("ddpm_evt", "1", 3)
assert len(_y) == len(ytr) + n_pos and _y[len(ytr):].all(), "training_set check failed"
del _X, _y
out("self-check OK")

# ================================================================ 2. train 5 seeds per setting
section("2. TRAIN EVERY SETTING WITH 5 SEEDS")
if not has("persistence"):
    record("persistence", va.dist_l0.values, te.dist_l0.values, "no model")
for seed in SEEDS:                                   # seed-major order: a disconnect still leaves complete seeds
    for m, r in CONFIGS:
        get_single(m, r, seed)
    save_summary()
D = done.set_index("key")

# ================================================================ 3. per-seed results
section("3. PER-SEED RESULTS (test 2016-2025): mean, spread, and how often EVT wins")
rows = []
for m, r in CONFIGS:
    ks = [single_key(m, r, s) for s in SEEDS]
    row = {"setting": cfg_name(m, r)}
    for s in ("pr_auc", "onset_pr_auc", "severe_pr_auc"):
        v = D.loc[ks, s].values.astype(float)
        row[s] = f"{v.mean():.4f} +- {v.std(ddof=1):.4f}"
    rows.append(row)
seed_tab = pd.DataFrame(rows)
seed_tab.to_csv(f"{OUT}/per_seed_summary.csv", index=False)
out(seed_tab.to_string(index=False))

out("\nSame-seed differences (wins = seeds where the first is better, out of 5):")
SEED_CMP = [(("ddpm_evt_tail", "1"), ("ddpm_emp", "1")), (("ddpm_evt", "1"), ("ddpm_emp", "1")),
            (("ddpm_evt_tail", "full"), ("ddpm_emp", "full")), (("ddpm_evt", "full"), ("ddpm_emp", "full")),
            (("ddpm_evt_tail", "1"), ("none", "-")), (("ddpm_evt", "full"), ("none", "-")),
            (("ddpm_emp", "1"), ("none", "-")), (("ddpm_emp", "full"), ("none", "-"))]
rows = []
for a, b in SEED_CMP:
    row = {"model": cfg_name(*a), "vs": cfg_name(*b)}
    for s in ("pr_auc", "onset_pr_auc", "severe_pr_auc"):
        d = np.array([D.loc[single_key(*a, sd), s] - D.loc[single_key(*b, sd), s] for sd in SEEDS], float)
        row[f"{s}_diff"] = round(float(d.mean()), 4)
        row[f"{s}_wins"] = f"{int((d > 0).sum())}/5"
    rows.append(row)
seed_cmp = pd.DataFrame(rows)
seed_cmp.to_csv(f"{OUT}/per_seed_comparisons.csv", index=False)
out(seed_cmp.to_string(index=False))

fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
names = [cfg_name(m, r) for m, r in CONFIGS]
for ax, (s, lab) in zip(axs, [("pr_auc", "PR-AUC"), ("onset_pr_auc", "onset PR-AUC"), ("severe_pr_auc", "severe PR-AUC")]):
    for i, (m, r) in enumerate(CONFIGS):
        v = [D.loc[single_key(m, r, sd), s] for sd in SEEDS]
        ax.scatter([i] * len(v), v, s=18, alpha=0.7)
        ax.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, "k-", lw=2)
    ax.axhline(D.loc["persistence", s], color="grey", ls=":", lw=1, label="persistence")
    ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace("lgbm/", "") for n in names], rotation=40, ha="right", fontsize=8)
    ax.set_title(f"{lab}: 5 seeds (dots), mean (bar)"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/seed_spread.png", dpi=120); plt.close()
save_summary()

# ================================================================ 4. seed-averaged ensembles and blends
section("4. SEED-AVERAGED ENSEMBLES (mean probability of the 5 seeds) AND BLENDS")
ENS = {}
for m, r in CONFIGS:
    key = f"ens5[{cfg_name(m, r)}]"
    ENS[(m, r)] = key
    if not has(key):
        P = [load_pred(single_key(m, r, s)) for s in SEEDS]
        record(key, np.mean([p[0] for p in P], 0), np.mean([p[1] for p in P], 0), "mean of 5 seeds", quiet=True)
BL = {}
base = ENS[("none", "-")]
for m, r in CONFIGS[1:]:
    key = f"blend[{base}+{ENS[(m, r)]}]"
    BL[(m, r)] = key
    if has(key):
        continue
    (va_a, te_a), (va_b, te_b) = load_pred(base), load_pred(ENS[(m, r)])
    la_v, lb_v, la_t, lb_t = logit(va_a), logit(va_b), logit(te_a), logit(te_b)
    best_w, best_s = 0.0, -1
    for w in BLEND_W:
        pv = (1 - w) * la_v + w * lb_v
        sc = np.mean([fast_ap(yva[m_], pv[m_]) for m_ in VSETS.values()])
        if sc > best_s:
            best_w, best_s = w, sc
    record(key, (1 - best_w) * la_v + best_w * lb_v, (1 - best_w) * la_t + best_w * lb_t, f"w={best_w}", quiet=True)
save_summary()
D = done.set_index("key")
cols = ["val_score", "pr_auc", "onset_pr_auc", "severe_pr_auc", "precision", "recall", "csi", "note"]
out(D.loc[["persistence"] + list(ENS.values()) + list(BL.values()), cols].to_string())

# ================================================================ 5. intervals and paired tests
section(f"5. 95% INTERVALS ({BOOT} resamples of whole test months) AND PAIRED TESTS")
keys = ["persistence"] + list(ENS.values()) + list(BL.values())
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

ci_rows = []
for k in keys:
    r = {"key": k, "val_score": D.loc[k, "val_score"]}
    for s in TSETS:
        v = np.array(boot[(k, s)])
        r[s] = D.loc[k, s]
        r[f"{s}_ci"] = f"[{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]"
    ci_rows.append(r)
ci = pd.DataFrame(ci_rows)
ci.to_csv(f"{OUT}/confidence_intervals.csv", index=False)
out(ci.to_string(index=False))

E = ENS
cmp = []
for r in ("1", "full"):
    cmp += [(E[("ddpm_evt_tail", r)], E[("ddpm_emp", r)]), (E[("ddpm_evt", r)], E[("ddpm_emp", r)]),
            (E[("ddpm_evt_tail", r)], E[("none", "-")]), (E[("ddpm_evt", r)], E[("none", "-")]),
            (E[("ddpm_emp", r)], E[("none", "-")]),
            (BL[("ddpm_evt_tail", r)], BL[("ddpm_emp", r)]), (BL[("ddpm_evt", r)], BL[("ddpm_emp", r)])]
cmp += [(k, "persistence") for k in [E[("none", "-")]] + [E[c] for c in CONFIGS[1:]] + list(BL.values())]
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
out("\nPaired comparisons (same resampled months). share_better >= 0.95 = convincing win, <= 0.05 = convincing loss:")
out(pairs.to_string(index=False))

# ================================================================ 6. final model by validation
section("6. FINAL MODEL (chosen by validation score among ensembles and blends)")
cand = done[done.key.isin(list(ENS.values()) + list(BL.values()))].sort_values("val_score", ascending=False)
best = cand.iloc[0]
pr_ = D.loc["persistence"]
out(f"chosen on validation: {best.key}  ({best.note}, val score {best.val_score})")
out(f"   test: PR-AUC {best.pr_auc}  onset {best.onset_pr_auc}  severe {best.severe_pr_auc}  "
    f"precision {best.precision}  recall {best.recall}  CSI {best.csi}")
out(f"   persistence: PR-AUC {pr_.pr_auc}  onset {pr_.onset_pr_auc}  severe {pr_.severe_pr_auc}  "
    f"precision {pr_.precision}  recall {pr_.recall}  CSI {pr_.csi}")
out("\n" + cand[["key", "val_score", "pr_auc", "onset_pr_auc", "severe_pr_auc", "precision", "recall", "csi", "note"]].to_string(index=False))
out(f"\nfiles in {OUT}")
out(f"END  (took {(time.time() - T0) / 60:.1f} min)")
save_summary()
print(f"\nSummary saved to {BASE}/step10_summary.txt")
