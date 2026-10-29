# ============================================================================
# STEP 9: HOW MUCH SYNTHETIC DATA, BLENDING, AND SEED CHECK
# Paste into ONE Colab cell. T4 GPU runtime recommended. ~60-90 min.
# Re-run the same cell after a disconnect: finished models are skipped.
#
# Why: Step 8 added 8.7 synthetic extremes per real one. EVT diffusion then gave LightGBM
# +0.11 PR-AUC on the most severe extremes, but it cost onset skill and hurt the MLP.
#   A. Ratio sweep  add 0.25x, 0.5x, 1x or all ("full", 8.7x) synthetic extremes per real one,
#                   for ddpm_emp (no EVT), ddpm_evt and ddpm_evt_tail, with both classifiers.
#   B. Seed check   repeat lgbm none / ddpm_emp@full / ddpm_evt@full with 2 more seeds.
#   C. Blending     average a plain model with an augmented one (logit average).
#                   The weight is chosen on validation only. Controls: blending with the
#                   no-EVT model, and blending two plain models.
# Everything is chosen on VALIDATION (2011-2015) with one score: the mean of the
# validation PR-AUC, onset PR-AUC and severe PR-AUC. The test years (2016-2025) are never
# used for choosing anything.
#
# Reads (never changes): data/model_ready_v3/*, results/step7/synthetic_*.parquet
# Writes ONLY to:        results/step9/
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
import torch
import torch.nn as nn
from google.colab import drive

warnings.filterwarnings("ignore")
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/himachal_project"
V3, S7, OUT = f"{BASE}/data/model_ready_v3", f"{BASE}/results/step7", f"{BASE}/results/step9"
PRED = f"{OUT}/preds"
LOCAL, LOCAL_PRED = "/content/local_v3", "/content/local_v3/preds9"

# ---------------------------------------------------------------- settings
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GEN = ["ddpm_emp", "ddpm_evt", "ddpm_evt_tail"]
RATIOS = ["0.25", "0.5", "1", "full"]          # synthetic extremes per real extreme
CLASSIFIERS = ["lgbm", "mlp"]
SEED, EXTRA_SEEDS = 42, [1, 2]
MLP_SEEDS, MLP_EPOCHS = [0, 1], 6
BLEND_W = np.round(np.linspace(0, 1, 11), 2)
BOOT = 200
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, max_bin=127,
                  n_estimators=1000, n_jobs=-1, verbose=-1)
# ----------------------------------------------------------------
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
    for p in (f"{OUT}/step9_summary.txt", f"{BASE}/step9_summary.txt"):
        open(p, "w").write(txt)


def section(t):
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


out(f"STEP 9  {time.strftime('%Y-%m-%d %H:%M')}  device={DEV}")

# ================================================================ 1. data
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


def subsets(df, y):
    """all days / onset days / normal days + 10% most severe extremes (cut from that period's extremes)"""
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
out("synthetic extremes available: " + ", ".join(f"{m} {len(v):,}" for m, v in SYN.items())
    + f"  (= {len(SYN['ddpm_evt']) / n_pos:.1f} per real extreme)")
del tr
progress("data loaded")


def training_set(m, r, seed):
    if m == "none":
        return Xtr, ytr
    Xg = SYN[m]
    if r != "full":
        n = int(round(float(r) * n_pos))
        assert n <= len(Xg), f"not enough synthetic rows for ratio {r}"
        Xg = Xg[np.random.default_rng(seed).choice(len(Xg), n, replace=False)]
    return np.vstack([Xtr, Xg]), np.concatenate([ytr, np.ones(len(Xg), "int8")])


# ================================================================ 2. classifiers (same as Step 8)
MONO = [1 if f in ("dist_l0", "dist_std_l0") else 0 for f in FEATS]
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6


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


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 256), nn.SiLU(), nn.Dropout(0.1),
                               nn.Linear(256, 256), nn.SiLU(), nn.Dropout(0.1),
                               nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.f(x).squeeze(1)


@torch.no_grad()
def predict(net, X):
    net.eval()
    outp = []
    for a in range(0, len(X), 200000):
        xb = torch.tensor(np.clip((X[a:a + 200000] - mu) / sd, -10, 10), device=DEV)
        outp.append(torch.sigmoid(net(xb)).cpu().numpy())
    return np.concatenate(outp)


def fit_mlp(X, y, seeds=MLP_SEEDS, epochs=MLP_EPOCHS):
    pv_all, pt_all = [], []
    Xg = torch.tensor(np.clip((X - mu) / sd, -10, 10), device=DEV)
    yg = torch.tensor(y.astype("float32"), device=DEV)
    for seed in seeds:
        torch.manual_seed(seed)
        net = MLP(X.shape[1]).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        best, best_state = -1, None
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(Xg), device=DEV)
            for a in range(0, len(Xg), 4096):
                i = perm[a:a + 4096]
                loss = lossf(net(Xg[i]), yg[i])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            ap = fast_ap(yva, predict(net, Xva))
            if ap > best:
                best, best_state = ap, {k: v.clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        pv_all.append(predict(net, Xva))
        pt_all.append(predict(net, Xte))
    del Xg, yg
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return np.mean(pv_all, 0), np.mean(pt_all, 0), f"{len(seeds)} seeds"


def pick_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return th[int(np.nanargmax(f1[:-1]))]


def logit(p):
    p = np.clip(p.astype("float64"), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- results store (resumable)
res_path = f"{OUT}/results.csv"
drive_ok()
done = pd.read_csv(res_path) if os.path.exists(res_path) else pd.DataFrame(columns=["key"])


def pfile(key, local=False):
    name = key.replace("/", "__").replace("@", "_at_").replace("#", "_").replace("[", "(").replace("]", ")").replace("+", "_plus_")
    return f"{LOCAL_PRED if local else PRED}/{name}.npz"


def has(key):
    if key not in set(done.key):
        return False
    drive_ok()
    return os.path.exists(pfile(key, True)) or os.path.exists(pfile(key))


def load_pred(key):
    if not os.path.exists(pfile(key, True)):
        drive_ok()
        shutil.copy(pfile(key), pfile(key, True))
    d = np.load(pfile(key, True))
    return d["va"], d["te"]


def record(key, pv, pt, note):
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
    np.savez(pfile(key, True), va=pv.astype("float32"), te=pt.astype("float32"))
    drive_ok()
    shutil.copy(pfile(key, True), pfile(key))
    done = pd.concat([done[done.key != key], pd.DataFrame([r])], ignore_index=True)
    done.to_csv(res_path, index=False)
    progress(f"{key}: val score {r['val_score']}  | test PR-AUC {r['pr_auc']}  onset {r['onset_pr_auc']}  severe {r['severe_pr_auc']}")


def run(clf, m, r="-", seed=SEED):
    key = f"{clf}/none" if m == "none" else f"{clf}/{m}@{r}"
    if seed != SEED:
        key += f"#s{seed}"
    if has(key):
        out(f"   {key}: already done, skipped")
        return key
    progress(f"training {key}")
    X, y = training_set(m, r, seed)
    if clf == "lgbm":
        pv, pt, note = fit_lgbm(X, y, seed)
    else:
        pv, pt, note = fit_mlp(X, y)
    if len(y) > len(ytr):
        note += f", +{len(y) - len(ytr):,} synthetic"
    record(key, pv, pt, note)
    del X, y
    return key


# ---------------------------------------------------------------- self-checks (about 1 minute)
_X, _y = training_set("ddpm_evt", "0.25", SEED)
assert len(_y) == len(ytr) + round(0.25 * n_pos) and _y[len(ytr):].all(), "training_set check failed"
del _X, _y
_i = rng.choice(len(Xtr), 20000, replace=False)
_pv, _pt, _ = fit_mlp(Xtr[_i], ytr[_i], seeds=[0], epochs=1)
assert len(_pt) == len(yte) and np.isfinite(_pt).all(), "MLP self-check failed"
_b = 0.5 * logit(np.array([0.1, 0.9])) + 0.5 * logit(np.array([0.2, 0.8]))
assert _b[0] < 0 < _b[1], "blend check failed"
out("self-checks OK")

# ================================================================ 2. ratio sweep
section("2. RATIO SWEEP (synthetic extremes per real extreme)")
if not has("persistence"):
    record("persistence", va.dist_l0.values, te.dist_l0.values, "no model")
for clf in CLASSIFIERS:
    run(clf, "none")
    for m in GEN:
        for r in RATIOS:
            run(clf, m, r)
save_summary()
D = done.set_index("key")
SEL = {}
rows = []
for clf in CLASSIFIERS:
    for m in GEN:
        cands = [f"{clf}/{m}@{r}" for r in RATIOS]
        SEL[(clf, m)] = max(cands, key=lambda k: D.loc[k, "val_score"])
        for k in [f"{clf}/none"] + cands:
            rows.append(dict(key=k, val_score=D.loc[k, "val_score"], pr_auc=D.loc[k, "pr_auc"],
                             onset=D.loc[k, "onset_pr_auc"], severe=D.loc[k, "severe_pr_auc"],
                             chosen="<- chosen on val" if k == SEL[(clf, m)] else ""))
sweep = pd.DataFrame(rows).drop_duplicates("key")
sweep.to_csv(f"{OUT}/ratio_sweep.csv", index=False)
out(sweep.to_string(index=False))
out(f"persistence: PR-AUC {D.loc['persistence', 'pr_auc']}  onset {D.loc['persistence', 'onset_pr_auc']}  "
    f"severe {D.loc['persistence', 'severe_pr_auc']}")

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
xs = [0.25, 0.5, 1, len(SYN["ddpm_evt"]) / n_pos]
for i, clf in enumerate(CLASSIFIERS):
    for j, (s, lab) in enumerate([("pr_auc", "PR-AUC"), ("onset_pr_auc", "onset PR-AUC"), ("severe_pr_auc", "severe PR-AUC")]):
        ax = axs[i, j]
        for m in GEN:
            ax.plot(xs, [D.loc[f"{clf}/{m}@{r}", s] for r in RATIOS], "o-", label=m)
        ax.axhline(D.loc[f"{clf}/none", s], color="k", ls="--", lw=1, label=f"{clf} no synthetic")
        ax.axhline(D.loc["persistence", s], color="grey", ls=":", lw=1, label="persistence")
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(["0.25", "0.5", "1", "full"])
        ax.set_title(f"{clf}: {lab} (test)"); ax.set_xlabel("synthetic extremes per real extreme")
        if i == 0 and j == 0:
            ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/ratio_sweep.png", dpi=120); plt.close()

# ================================================================ 3. seed check
section("3. SEED CHECK (LightGBM, 3 training seeds: 42, 1, 2)")
for s in EXTRA_SEEDS:
    for m, r in (("none", "-"), ("ddpm_emp", "full"), ("ddpm_evt", "full")):
        run("lgbm", m, r, s)
save_summary()
D = done.set_index("key")
rows = []
for base in ("lgbm/none", "lgbm/ddpm_emp@full", "lgbm/ddpm_evt@full"):
    ks = [base] + [f"{base}#s{s}" for s in EXTRA_SEEDS]
    r = {"model": base}
    for s in ("pr_auc", "onset_pr_auc", "severe_pr_auc"):
        v = D.loc[ks, s].values
        r[s] = f"{v.mean():.4f} (min {v.min():.4f}, max {v.max():.4f})"
    rows.append(r)
out(pd.DataFrame(rows).to_string(index=False))
for s in [SEED] + EXTRA_SEEDS:
    suf = "" if s == SEED else f"#s{s}"
    out(f"seed {s:>2}: severe PR-AUC  ddpm_evt - ddpm_emp = "
        f"{D.loc['lgbm/ddpm_evt@full' + suf, 'severe_pr_auc'] - D.loc['lgbm/ddpm_emp@full' + suf, 'severe_pr_auc']:+.4f}   "
        f"ddpm_evt - none = {D.loc['lgbm/ddpm_evt@full' + suf, 'severe_pr_auc'] - D.loc['lgbm/none' + suf, 'severe_pr_auc']:+.4f}")

# ================================================================ 4. blending
section("4. BLENDING (logit average, weight on the 2nd model chosen on validation)")
BLENDS = []
for base in ("lgbm/none", "mlp/none"):
    for m in GEN:
        BLENDS.append((base, SEL[("lgbm", m)]))
    for m in GEN:
        BLENDS.append((base, SEL[("mlp", m)]))
BLENDS.append(("mlp/none", "lgbm/none"))                 # control: two plain models, no synthetic data
BLENDS = list(dict.fromkeys(BLENDS))
BKEY = {}
for a, b in BLENDS:
    key = f"blend[{a}+{b}]"
    BKEY[(a, b)] = key
    if has(key):
        out(f"   {key}: already done, skipped")
        continue
    (va_a, te_a), (va_b, te_b) = load_pred(a), load_pred(b)
    la_v, lb_v, la_t, lb_t = logit(va_a), logit(va_b), logit(te_a), logit(te_b)
    best_w, best_s = 0.0, -1
    for w in BLEND_W:
        pv = (1 - w) * la_v + w * lb_v
        s = np.mean([fast_ap(yva[m_], pv[m_]) for m_ in VSETS.values()])
        if s > best_s:
            best_w, best_s = w, s
    record(key, (1 - best_w) * la_v + best_w * lb_v, (1 - best_w) * la_t + best_w * lb_t, f"w={best_w}")
save_summary()
D = done.set_index("key")
out(D.loc[list(BKEY.values()), ["val_score", "pr_auc", "onset_pr_auc", "severe_pr_auc", "recall", "csi", "note"]].to_string())

# ================================================================ 5. intervals and paired tests
section(f"5. 95% INTERVALS ({BOOT} resamples of whole test months) AND PAIRED TESTS")
keys = ["persistence", "lgbm/none", "mlp/none"] + list(dict.fromkeys(SEL.values())) + list(BKEY.values())
P = {k: (te.dist_l0.values if k == "persistence" else load_pred(k)[1]) for k in keys}
months = te.month.values
um, inv = np.unique(months, return_inverse=True)
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

cmp = []
for clf in CLASSIFIERS:
    e, t, p = SEL[(clf, "ddpm_evt")], SEL[(clf, "ddpm_evt_tail")], SEL[(clf, "ddpm_emp")]
    cmp += [(e, p), (t, p), (e, f"{clf}/none"), (t, f"{clf}/none"), (p, f"{clf}/none")]
for base in ("lgbm/none", "mlp/none"):
    be, bt, bp = (BKEY[(base, SEL[("lgbm", m)])] for m in ("ddpm_evt", "ddpm_evt_tail", "ddpm_emp"))
    cmp += [(be, bp), (bt, bp), (be, base), (be, "persistence")]
cmp += [(BKEY[("mlp/none", SEL[("lgbm", "ddpm_evt")])], BKEY[("mlp/none", "lgbm/none")]),
        (BKEY[("mlp/none", "lgbm/none")], "mlp/none")]
pairs = []
for a, b in dict.fromkeys(cmp):
    if a == b:
        continue
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

# ================================================================ 6. best model by validation
section("6. BEST MODEL (chosen by validation score only) AND ALL SCORES")
cand = done[~done.key.str.contains("#s")].sort_values("val_score", ascending=False)
best = cand.iloc[0]
out(f"best on validation: {best.key}  (val score {best.val_score})")
out(f"   test: PR-AUC {best.pr_auc}  onset {best.onset_pr_auc}  severe {best.severe_pr_auc}  "
    f"precision {best.precision}  recall {best.recall}  CSI {best.csi}")
pr_ = D.loc["persistence"]
out(f"   persistence: PR-AUC {pr_.pr_auc}  onset {pr_.onset_pr_auc}  severe {pr_.severe_pr_auc}  "
    f"precision {pr_.precision}  recall {pr_.recall}  CSI {pr_.csi}")
out("\n" + cand.to_string(index=False))
out(f"\nfiles in {OUT}")
out(f"END  (took {(time.time() - T0) / 60:.1f} min)")
save_summary()
print(f"\nSummary saved to {BASE}/step9_summary.txt")
