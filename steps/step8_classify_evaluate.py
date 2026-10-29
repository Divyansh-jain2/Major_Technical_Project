# ============================================================================
# STEP 8: TWO CLASSIFIERS x ALL REBALANCING METHODS, SCORED ON THE EXTREME REGION
# Paste into ONE Colab cell. T4 GPU runtime recommended (the MLP uses it). ~50-70 min.
# Re-run the same cell after a disconnect: finished models are skipped.
#
# Classifiers (same settings for every training set):
#   lgbm  LightGBM (strong; literature: balancing rarely helps it)
#   mlp   neural network, 3 x 256, averaged over 2 seeds (augmentation should matter more)
# Training sets:
#   none           real data only                     weighted   real, extremes weighted up
#   smote          + SMOTE extremes                    ctgan      + CTGAN extremes
#   ddpm_emp       + diffusion, observed severities (no EVT)
#   ddpm_evt       + diffusion, GPD severities (EVT)    ddpm_evt_tail  + EVT with tail emphasis
# Scores on test 2016-2025 (threshold-free, with 95% month-block bootstrap intervals):
#   pr_auc         all days
#   onset_pr_auc   only days where today is not already past tomorrow's threshold
#   severe_pr_auc  normal days vs the 10% MOST SEVERE extremes (the extreme region)
# plus precision / recall / CSI at a threshold chosen on 2011-2015.
# Key comparisons (paired): ddpm_evt vs ddpm_emp (= the effect of EVT), vs smote, vs none, vs ctgan.
#
# Reads (never changes): data/model_ready_v3/*, results/step7/synthetic_*.parquet
# Writes ONLY to:        results/step8/
# ============================================================================
!pip install -q lightgbm imbalanced-learn

import os, json, time, shutil, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
from google.colab import drive

warnings.filterwarnings("ignore")
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/himachal_project"
V3, S7, OUT = f"{BASE}/data/model_ready_v3", f"{BASE}/results/step7", f"{BASE}/results/step8"
LOCAL = "/content/local_v3"

# ---------------------------------------------------------------- settings
SYNTH_RATIO = 0.10
DEV = "cuda" if torch.cuda.is_available() else "cpu"
METHODS = ["none", "weighted", "smote", "ctgan", "ddpm_emp", "ddpm_evt", "ddpm_evt_tail"]
CLASSIFIERS = ["lgbm", "mlp"]
MLP_SEEDS = [0, 1]
MLP_EPOCHS = 6
BOOT = 200
SEED = 42
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, max_bin=127,
                  n_estimators=1000, random_state=SEED, n_jobs=-1, verbose=-1)
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
for f in ("train.parquet", "val.parquet", "test.parquet", "features.json"):
    copy_local(f"{V3}/{f}")
syn_paths = {}
for m in METHODS:
    p = f"{S7}/synthetic_{m}.parquet"
    if m in ("ctgan", "ddpm_emp", "ddpm_evt", "ddpm_evt_tail"):
        try:
            drive_ok()
            if os.path.exists(p):
                syn_paths[m] = copy_local(p)
        except Exception:
            pass
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
    for p in (f"{OUT}/step8_summary.txt", f"{BASE}/step8_summary.txt"):
        open(p, "w").write(txt)


def section(t):
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


out(f"STEP 8  {time.strftime('%Y-%m-%d %H:%M')}  device={DEV}")
missing = [m for m in ("ctgan", "ddpm_emp", "ddpm_evt", "ddpm_evt_tail") if m not in syn_paths]
if missing:
    out(f"NOTE: no synthetic data found for {missing} (run Step 7 first); those methods are skipped")
METHODS = [m for m in METHODS if m not in missing]

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
n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())
for nm, y in (("train", ytr), ("val", yva), ("test", yte)):
    out(f"{nm:<5} rows {len(y):>9,}  extremes {int(y.sum()):>7,}  ({100 * y.mean():.2f}%)")
onset_te = te.onset.values == 1
zt = te.z.values
severe_cut = np.nanquantile(zt[yte == 1], 0.9)
severe_rows = (yte == 0) | (zt >= severe_cut)            # normal days + 10% most severe extremes
out(f"test: {int((onset_te & (yte == 1)).sum()):,} onset extremes; severe set = extremes with z >= {severe_cut:.2f} "
    f"({int(((yte == 1) & (zt >= severe_cut)).sum()):,} rows)")
progress("data loaded")


def synthetic(m):
    d = pd.read_parquet(syn_paths[m], columns=FEATS)
    return np.nan_to_num(d[FEATS].values.astype("float32"))


def training_set(m):
    if m in ("none", "weighted"):
        return Xtr, ytr
    if m == "smote":
        Xs, ys = SMOTE(sampling_strategy=SYNTH_RATIO, random_state=SEED).fit_resample(Xtr, ytr)
        return Xs.astype("float32"), ys.astype("int8")
    Xg = synthetic(m)
    return np.vstack([Xtr, Xg]), np.concatenate([ytr, np.ones(len(Xg), "int8")])


# ================================================================ 2. classifiers
MONO = [1 if f in ("dist_l0", "dist_std_l0") else 0 for f in FEATS]
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6


def fit_lgbm(X, y, weighted, n_fixed=None):
    params = dict(LGB_PARAMS, monotone_constraints=MONO)
    if weighted:
        params.update(scale_pos_weight=float(np.sqrt(n_neg / n_pos)), n_estimators=n_fixed)
        m = lgb.LGBMClassifier(**params)
        m.fit(X, y)
    else:
        m = lgb.LGBMClassifier(**params)
        m.fit(X, y, eval_set=[(Xva, yva)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(60, verbose=False)])
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], (m.best_iteration_ or n_fixed)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 256), nn.SiLU(), nn.Dropout(0.1),
                               nn.Linear(256, 256), nn.SiLU(), nn.Dropout(0.1),
                               nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.f(x).squeeze(1)


def fast_ap(y, s):
    o = np.argsort(-s, kind="stable")
    y = y[o]
    tp = np.cumsum(y)
    return float(np.sum((tp / np.arange(1, len(y) + 1))[y == 1]) / max(tp[-1], 1))


@torch.no_grad()
def predict(net, X):
    net.eval()
    outp = []
    for a in range(0, len(X), 200000):
        xb = torch.tensor(np.clip((X[a:a + 200000] - mu) / sd, -10, 10), device=DEV)
        outp.append(torch.sigmoid(net(xb)).cpu().numpy())
    return np.concatenate(outp)


def fit_mlp(X, y, weighted):
    pv_all, pt_all = [], []
    Xg = torch.tensor(np.clip((X - mu) / sd, -10, 10), device=DEV)
    yg = torch.tensor(y.astype("float32"), device=DEV)
    pw = torch.tensor(float(np.sqrt(n_neg / n_pos)) if weighted else 1.0, device=DEV)
    for seed in MLP_SEEDS:
        torch.manual_seed(seed)
        net = MLP(X.shape[1]).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        best, best_state = -1, None
        for ep in range(MLP_EPOCHS):
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
    return np.mean(pv_all, 0), np.mean(pt_all, 0), f"{len(MLP_SEEDS)} seeds"


def pick_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return th[int(np.nanargmax(f1[:-1]))]


res_path, pred_path = f"{OUT}/results.csv", f"{OUT}/test_predictions.parquet"
done = pd.read_csv(res_path) if os.path.exists(res_path) else pd.DataFrame()
preds = pd.read_parquet(pred_path) if os.path.exists(pred_path) else pd.DataFrame(index=range(len(te)))


def record(key, p_va, p_te, note):
    global done, preds
    thr = pick_threshold(yva, p_va)
    yhat = p_te >= thr
    tp = int((yhat & (yte == 1)).sum()); fp = int((yhat & (yte == 0)).sum()); fn = int((~yhat & (yte == 1)).sum())
    r = dict(key=key, val_pr_auc=round(fast_ap(yva, p_va), 4), pr_auc=round(fast_ap(yte, p_te), 4),
             onset_pr_auc=round(fast_ap(yte[onset_te], p_te[onset_te]), 4),
             severe_pr_auc=round(fast_ap(yte[severe_rows], p_te[severe_rows]), 4),
             precision=round(tp / max(tp + fp, 1), 3), recall=round(tp / max(tp + fn, 1), 3),
             csi=round(tp / max(tp + fp + fn, 1), 3),
             onset_recall=round(float(yhat[onset_te & (yte == 1)].mean()), 3),
             severe_recall=round(float(yhat[(yte == 1) & (zt >= severe_cut)].mean()), 3), note=note)
    drive_ok()
    done = pd.concat([done[done.key != key] if len(done) else done, pd.DataFrame([r])], ignore_index=True)
    done.to_csv(res_path, index=False)
    preds[key] = p_te.astype("float32")
    preds.to_parquet(pred_path, index=False)
    progress(f"{key}: PR-AUC {r['pr_auc']}  onset {r['onset_pr_auc']}  severe {r['severe_pr_auc']}")


section("2. TRAIN AND SCORE")
if "mlp" in CLASSIFIERS:                     # 1-minute self-check of the neural network path
    _keep = MLP_SEEDS, MLP_EPOCHS
    MLP_SEEDS, MLP_EPOCHS = [0], 1
    _i = rng.choice(len(Xtr), 20000, replace=False)
    _pv, _pt, _ = fit_mlp(Xtr[_i], ytr[_i], False)
    assert len(_pt) == len(yte) and np.isfinite(_pt).all(), "MLP self-check failed"
    MLP_SEEDS, MLP_EPOCHS = _keep
    out("self-check: neural network path OK")
if not (len(done) and "persistence" in set(done.key)):
    record("persistence", va.dist_l0.values, te.dist_l0.values, "no model")
lgb_trees = None
for clf in CLASSIFIERS:
    for m in METHODS:
        key = f"{clf}/{m}"
        if len(done) and key in set(done.key):
            out(f"   {key}: already done, skipped")
            if key == "lgbm/none":
                lgb_trees = int(str(done.set_index("key").loc[key, "note"]).split()[0])
            continue
        progress(f"training {key}")
        X, y = training_set(m)
        if clf == "lgbm":
            if m == "weighted" and lgb_trees is None:
                lgb_trees = 300
            pv, pt, n = fit_lgbm(X, y, m == "weighted", lgb_trees)
            if m == "none":
                lgb_trees = int(n)
            note = f"{n} trees" + (" (fixed)" if m == "weighted" else "") + (f", +{len(y) - len(ytr):,} synthetic" if len(y) > len(ytr) else "")
        else:
            pv, pt, note = fit_mlp(X, y, m == "weighted")
            if len(y) > len(ytr):
                note += f", +{len(y) - len(ytr):,} synthetic"
        record(key, pv, pt, note)
        del X, y
save_summary()

# ================================================================ 3. intervals and paired tests
section(f"3. 95% INTERVALS ({BOOT} resamples of whole test months) AND PAIRED TESTS")
months = te.month.values
um, inv = np.unique(months, return_inverse=True)
keys = [k for k in preds.columns]
P = {k: preds[k].values for k in keys}
subsets = {"pr_auc": np.ones(len(yte), bool), "onset_pr_auc": onset_te, "severe_pr_auc": severe_rows}
order = {(k, s): np.argsort(-P[k][m_], kind="stable") for k in keys for s, m_ in subsets.items()}


def wap(y, w, o):
    """AP for a resample given as per-row multiplicities w (rows pre-sorted by score)."""
    y, w = y[o], w[o]
    tpw = np.cumsum(w * y)
    allw = np.cumsum(w)
    denom = tpw[-1]
    return float(np.sum((w * y)[w > 0] * (tpw / np.maximum(allw, 1e-12))[w > 0]) / max(denom, 1e-12))


boot = {(k, s): [] for k in keys for s in subsets}
for b in range(BOOT):
    cnt = np.bincount(rng.integers(0, len(um), len(um)), minlength=len(um)).astype("float64")
    w_all = cnt[inv]
    for s, m_ in subsets.items():
        ys, ws = yte[m_], w_all[m_]
        for k in keys:
            boot[(k, s)].append(wap(ys, ws, order[(k, s)]))
    if (b + 1) % 50 == 0:
        progress(f"bootstrap {b + 1}/{BOOT}")

ci_rows = []
for k in keys:
    r = {"key": k}
    for s in subsets:
        v = np.array(boot[(k, s)])
        r[s] = done.set_index("key").loc[k, s]
        r[f"{s}_ci"] = f"[{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]"
    ci_rows.append(r)
ci = pd.DataFrame(ci_rows)
ci.to_csv(f"{OUT}/confidence_intervals.csv", index=False)
out(ci.to_string(index=False))

pairs = []
for clf in CLASSIFIERS:
    for a, b in [("ddpm_evt", "ddpm_emp"), ("ddpm_evt_tail", "ddpm_emp"), ("ddpm_evt", "smote"),
                 ("ddpm_evt_tail", "smote"), ("ddpm_evt", "none"), ("ddpm_evt_tail", "none"),
                 ("ddpm_evt", "ctgan"), ("ddpm_emp", "ctgan"), ("smote", "none"), ("weighted", "none")]:
        ka, kb = f"{clf}/{a}", f"{clf}/{b}"
        if ka not in keys or kb not in keys:
            continue
        r = {"classifier": clf, "method": a, "vs": b}
        for s in subsets:
            d = np.array(boot[(ka, s)]) - np.array(boot[(kb, s)])
            r[f"{s}_diff"] = round(float(d.mean()), 4)
            r[f"{s}_share_better"] = round(float(np.mean(d > 0)), 2)
        pairs.append(r)
for clf in CLASSIFIERS:
    ka = f"{clf}/ddpm_evt"
    if ka in keys:
        r = {"classifier": clf, "method": "ddpm_evt", "vs": "persistence"}
        for s in subsets:
            d = np.array(boot[(ka, s)]) - np.array(boot[("persistence", s)])
            r[f"{s}_diff"] = round(float(d.mean()), 4)
            r[f"{s}_share_better"] = round(float(np.mean(d > 0)), 2)
        pairs.append(r)
pairs = pd.DataFrame(pairs)
pairs.to_csv(f"{OUT}/paired_tests.csv", index=False)
out("\nPaired comparisons (same resampled months). share_better >= 0.95 = convincing win, <= 0.05 = convincing loss:")
out(pairs.to_string(index=False))

section("4. ALL SCORES")
out(done.to_string(index=False))
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for ax, clf in zip(axs, CLASSIFIERS):
    for k in ["persistence"] + [f"{clf}/{m}" for m in METHODS]:
        if k not in P:
            continue
        pr, rc, _ = precision_recall_curve(yte, P[k])
        ax.plot(rc, pr, lw=2 if "evt" in k else 1, label=f"{k} ({done.set_index('key').loc[k, 'pr_auc']:.3f})")
    ax.axhline(yte.mean(), color="grey", ls="--", lw=0.8)
    ax.set_title(f"{clf}: test 2016-2025"); ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/pr_curves.png", dpi=120); plt.close()
out(f"\nfiles in {OUT}")
out(f"END  (took {(time.time() - T0) / 60:.1f} min)")
save_summary()
print(f"\nSummary saved to {BASE}/step8_summary.txt")
