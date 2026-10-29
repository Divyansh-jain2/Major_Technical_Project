# ============================================================================
# STEP 6: BUILD DATASET v3 (adds longer-history features)
# Paste into ONE Colab cell and run. CPU or GPU runtime, ~10-15 min.
#
# Same task as before: is day D+1 a seasonal heat extreme for this cell?
# (thresholds = 99th pct of Tmax per cell and IMD season, 1981-2010)
# New: features describing the last 7-30 days (the useful part of the "LSTM idea"),
# and per-row helpers so later steps can score the extreme region:
#   z       severity of tomorrow's extreme = (Tmax - threshold) / cell variability (extremes only)
#   onset   1 if TODAY is not already past tomorrow's threshold
# z and onset are NEVER used as classifier features.
#
# Reads (never changes): data/processed/himachal_daily_raw.nc, data/model_ready/{thresholds.nc,cells.csv},
#                        results/step3b/cell_scales.csv, raw t2m_mean files 1981-85 (gap refill)
# Writes ONLY to:        data/model_ready_v3/  (train = random half of 1981-2010, val, test, features.json)
# ============================================================================
!pip install -q xarray netCDF4 pyarrow

import os, re, glob, json, time, shutil
import numpy as np
import pandas as pd
import xarray as xr
from google.colab import drive

drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/himachal_project"
PROC = f"{BASE}/data/processed/himachal_daily_raw.nc"
RAW_DAILY = f"{BASE}/data/raw/era5land/daily"
MR = f"{BASE}/data/model_ready"
OUT = f"{BASE}/data/model_ready_v3"
LOCAL = "/content/local_data"

SPLITS = {"train": (1981, 2010), "val": (2011, 2015), "test": (2016, 2025)}   # by LABEL day
SEASONS = {"winter": [12, 1, 2], "pre_monsoon": [3, 4, 5], "monsoon": [6, 7, 8, 9], "post_monsoon": [10, 11]}
TRAIN_FRACTION = 0.5
SEED = 42
START = 30                                   # first usable day (needs 30 days of history)
SHORT_VARS = ["t2m_max", "t2m_min", "t2m_mean", "d2m", "rh", "sp", "ssrd", "swvl1", "tp"]
SNAMES = list(SEASONS)


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


T0 = time.time()
LINES = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def progress(msg):
    print(f"   ... [{(time.time() - T0) / 60:4.1f} min] {msg}", flush=True)


os.makedirs(LOCAL, exist_ok=True)
proc_l, thr_l, cells_l = copy_local(PROC), copy_local(f"{MR}/thresholds.nc"), copy_local(f"{MR}/cells.csv")
drive_ok()
os.makedirs(OUT, exist_ok=True)
out(f"STEP 6 (dataset v3)  {time.strftime('%Y-%m-%d %H:%M')}")

# ---------------------------------------------------------------- fields (days x cells)
ds = xr.open_dataset(proc_l)
mask = ds["hp_mask"].values.astype(bool)
dates = pd.DatetimeIndex(ds["time"].values)
T = len(dates)
A = {v: ds[v].values[:, mask].astype("float32") for v in SHORT_VARS}
NC = A["t2m_max"].shape[1]
cells = pd.read_csv(cells_l)
for y in sorted({d.year for d, bad in zip(dates, np.isnan(A["t2m_mean"]).all(axis=1)) if bad}):
    pat = re.compile(rf"^t2m_mean_{y}( \(\d+\))?\.nc$")
    fs = sorted(f for f in glob.glob(f"{RAW_DAILY}/t2m_mean_{y}*.nc") if pat.match(os.path.basename(f)))
    if not fs:
        continue
    r = xr.open_dataset(fs[0])
    r = r.rename({k: v for k, v in {"valid_time": "time", "latitude": "lat", "longitude": "lon"}.items()
                  if k in r.dims or k in r.coords})
    g = r[list(r.data_vars)[0]].reindex(lat=ds.lat.values, lon=ds.lon.values, method="nearest", tolerance=1e-3)
    vals = g.values[:, mask].astype("float32")
    vals = vals - 273.15 if np.nanmedian(vals) > 150 else vals
    idx = dates.get_indexer(pd.DatetimeIndex(g["time"].values).floor("D"))
    ok = idx >= 0
    A["t2m_mean"][idx[ok]] = np.where(np.isnan(A["t2m_mean"][idx[ok]]), vals[ok], A["t2m_mean"][idx[ok]])
a_, b_ = 17.625, 243.04
rh_calc = np.clip(100 * np.exp(a_ * A["d2m"] / (b_ + A["d2m"]) - a_ * A["t2m_mean"] / (b_ + A["t2m_mean"])), 0, 100)
A["rh"] = np.where(np.isfinite(A["rh"]), A["rh"], rh_calc).astype("float32")
out(f"{T} days x {NC} cells; missing: " + ", ".join(f"{v} {100 * np.isnan(A[v]).mean():.2f}%" for v in SHORT_VARS))

th = xr.open_dataset(thr_l)
thr_sea = th["thr_heat_season"].values                    # (season, cell)
clim_tx, clim_tn = th["clim_tmax"].values, th["clim_tmin"].values
doy = dates.dayofyear.values.copy()
doy[dates.is_leap_year & (dates.month > 2)] -= 1
doy = np.minimum(doy, 365)
m2s = {m: k for k, s in enumerate(SNAMES) for m in SEASONS[s]}
sidx = np.array([m2s[m] for m in dates.month])
sc = pd.read_csv(f"{BASE}/results/step3b/cell_scales.csv")
scale = sc.pivot(index="cell", columns="season_next", values="heat_scale").reindex(cells.cell).values.T  # (season, cell)
EXT = A["t2m_max"] > thr_sea[sidx]
ANOM_X = A["t2m_max"] - clim_tx[doy - 1]
ANOM_N = A["t2m_min"] - clim_tn[doy - 1]
progress("fields ready")

# ---------------------------------------------------------------- longer-history features (trailing, end on D)


def roll(a, w, how="mean"):
    r = pd.DataFrame(a).rolling(w, min_periods=max(1, w // 2))
    return (r.mean() if how == "mean" else r.sum()).values.astype("float32")


def run_length(b, cap=60):
    o, cur = np.zeros(b.shape, "float32"), np.zeros(b.shape[1], "float32")
    for t in range(b.shape[0]):
        cur = np.where(b[t], cur + 1, 0)
        o[t] = cur
    return np.minimum(o, cap)


def days_since(b, cap=365):
    o, cur = np.zeros(b.shape, "float32"), np.full(b.shape[1], cap, "float32")
    for t in range(b.shape[0]):
        cur = np.where(b[t], 0, np.minimum(cur + 1, cap))
        o[t] = cur
    return o


def lagdiff(a, k):
    d = np.full_like(a, np.nan)
    d[k:] = a[k:] - a[:-k]
    return d


LONG = {
    "tmax_anom_m7": roll(ANOM_X, 7), "tmax_anom_m14": roll(ANOM_X, 14), "tmax_anom_m30": roll(ANOM_X, 30),
    "tmin_anom_m7": roll(ANOM_N, 7), "tmin_anom_m30": roll(ANOM_N, 30),
    "warm_spell_days": run_length(ANOM_X > 0),
    "days_since_extreme": days_since(EXT),
    "extremes_last30": roll(EXT.astype("float32"), 30, "sum"),
    "tp_sum7": roll(A["tp"], 7, "sum"), "tp_sum30": roll(A["tp"], 30, "sum"),
    "swvl1_m14": roll(A["swvl1"], 14), "swvl1_m30": roll(A["swvl1"], 30), "swvl1_change14": lagdiff(A["swvl1"], 14),
    "d2m_m7": roll(A["d2m"], 7), "ssrd_m7": roll(A["ssrd"], 7), "sp_change3": lagdiff(A["sp"], 3),
}
progress("long features ready")

STATIC = ["lat", "lon", "elev_m", "slope_deg"]
BASE_FEATS = STATIC + [f"{v}_l{l}" for v in SHORT_VARS for l in (0, 1, 2)] + ["tmax_anom_l0", "tmin_anom_l0"] + list(LONG)
DERIVED = [f"dist_l{l}" for l in (0, 1, 2)] + [f"dist_std_l{l}" for l in (0, 1, 2)] + ["dist_trend", "tmax_change"]
FEATS = BASE_FEATS + DERIVED


def build(split):
    y0, y1 = SPLITS[split]
    D = np.arange(START, T - 1)
    D = D[(dates.year.values[D + 1] >= y0) & (dates.year.values[D + 1] <= y1)]
    tgt = D + 1
    cols = {"cell": np.tile(cells.cell.values.astype("int16"), len(D)),
            "season": np.repeat(sidx[tgt].astype("int8"), NC),
            "month": np.repeat((dates.year.values * 12 + dates.month.values)[tgt].astype("int32"), NC)}
    for c in STATIC:
        cols[c] = np.tile(cells[c].values.astype("float32"), len(D))
    for v in SHORT_VARS:
        for l in (0, 1, 2):
            cols[f"{v}_l{l}"] = A[v][D - l].ravel()
    cols["tmax_anom_l0"] = ANOM_X[D].ravel()
    cols["tmin_anom_l0"] = ANOM_N[D].ravel()
    for k, a in LONG.items():
        cols[k] = a[D].ravel()
    thr_t, sc_t = thr_sea[sidx[tgt]], scale[sidx[tgt]]
    cols["thr"] = thr_t.ravel()
    cols["scale"] = sc_t.ravel()
    for l in (0, 1, 2):
        d_ = A["t2m_max"][D - l] - thr_t
        cols[f"dist_l{l}"] = d_.ravel()
        cols[f"dist_std_l{l}"] = (d_ / sc_t).ravel()
    cols["dist_trend"] = cols["dist_l0"] - cols["dist_l1"]
    cols["tmax_change"] = cols["t2m_max_l0"] - cols["t2m_max_l1"]
    y = EXT[tgt]
    cols["y"] = y.ravel().astype("int8")
    cols["z"] = np.where(y, (A["t2m_max"][tgt] - thr_t) / sc_t, np.nan).ravel().astype("float32")
    cols["onset"] = (cols["dist_l0"] <= 0).astype("int8")
    df = pd.DataFrame(cols)
    if split == "train" and TRAIN_FRACTION < 1:
        df = df.sample(frac=TRAIN_FRACTION, random_state=SEED).reset_index(drop=True)
    return df


info = []
for split in SPLITS:
    df = build(split)
    p = f"{OUT}/{split}.parquet"
    drive_ok()
    df.to_parquet(p, index=False, compression="zstd")
    info.append(dict(split=split, rows=len(df), extremes=int(df.y.sum()), pct=round(100 * df.y.mean(), 3),
                     onset_extremes=int(((df.y == 1) & (df.onset == 1)).sum()),
                     missing_pct=round(100 * float(df[FEATS].isna().mean().max()), 3),
                     size_mb=round(os.path.getsize(p) / 1e6, 1)))
    progress(f"{split} written")
    del df
json.dump({"BASE_FEATS": BASE_FEATS, "DERIVED": DERIVED, "FEATS": FEATS, "LONG": list(LONG),
           "SEASONS": SNAMES, "SPLITS": SPLITS, "TRAIN_FRACTION": TRAIN_FRACTION},
          open(f"{OUT}/features.json", "w"), indent=1)
for f in ("thresholds.nc", "cells.csv"):
    shutil.copy(f"{LOCAL}/{f}", f"{OUT}/{f}")
out(pd.DataFrame(info).to_string(index=False))
out(f"{len(FEATS)} classifier features ({len(LONG)} new long-history ones): {list(LONG)}")
out(f"END  (took {(time.time() - T0) / 60:.1f} min)")
drive_ok()
for p in (f"{OUT}/step6_summary.txt", f"{BASE}/step6_summary.txt"):
    open(p, "w").write("\n".join(LINES) + "\n")
