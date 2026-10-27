# ============================================================================
# STEP 2 (v2): BUILD THE MODEL-READY DATASET (next-day temperature extremes)
# Paste into ONE Colab cell and run. Takes about 3-10 minutes.
#
# Reads (never changes):
#   data/processed/himachal_daily_raw.nc            (checked in Step 1)
#   data/raw/era5land/daily/t2m_mean_198X*.nc       (to refill Tmean 1981-1985)
# Writes ONLY to:  himachal_project/data/model_ready/   (replaces the v1 files there)
#   train.parquet  1981-2010 label days, all rows (natural imbalance)
#   val.parquet    2011-2015
#   test.parquet   2016-2025
#   thresholds.nc  all thresholds and climatology per cell
#   cells.csv      cell id, lat, lon, elevation, slope
#   step2_summary.txt   (also copied to himachal_project/step2_summary.txt)
#
# One row = one Himachal grid cell on day D.
#   Features: weather on D, D-1, D-2 + terrain + season.
#   Labels: is day D+1 an extreme for this cell? Three ways of comparing:
#     *_season  vs the same SEASON in other years      (IMD seasons, see SEASONS)
#     *_annual  vs the WHOLE YEAR in other years
#     *_doy     vs the same TIME OF YEAR (+/-7 days) in other years
#   y_heat_*  Tmax(D+1) above the 99th percentile
#   y_cold_*  Tmin(D+1) below the 1st percentile
#   Thresholds use 1981-2010 only, so nothing from val/test leaks into them.
# ============================================================================
!pip install -q xarray netCDF4 pyarrow

import glob, os, re, time
import numpy as np
import pandas as pd
import xarray as xr
from google.colab import drive

drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/himachal_project"
PROC = f"{BASE}/data/processed/himachal_daily_raw.nc"
RAW_DAILY = f"{BASE}/data/raw/era5land/daily"
OUTDIR = f"{BASE}/data/model_ready"

# ---------------------------------------------------------------- settings
SPLITS = {"train": (1981, 2010), "val": (2011, 2015), "test": (2016, 2025)}   # by LABEL day
BASE_YEARS = (1981, 2010)          # thresholds + climatology from these years only
SEASONS = {"winter": [12, 1, 2], "pre_monsoon": [3, 4, 5],          # IMD seasons
           "monsoon": [6, 7, 8, 9], "post_monsoon": [10, 11]}
WINDOW = 7                         # +/- days around each day of year (doy labels)
HOT_Q, COLD_Q = 99, 1
LAGS = (0, 1, 2)                   # D, D-1, D-2
# processed-file units (verified in Step 1): temps C, sp hPa, ssrd W m-2, tp mm/day
FEATURE_VARS = ["t2m_max", "t2m_min", "t2m_mean", "d2m", "rh", "sp", "ssrd", "swvl1", "tp"]
# ----------------------------------------------------------------

os.makedirs(OUTDIR, exist_ok=True)
LINES, T0 = [], time.time()
pd.set_option("display.width", 250)
SNAMES = list(SEASONS)


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def progress(msg):
    print(f"   ... [{(time.time() - T0) / 60:4.1f} min] {msg}", flush=True)


def save_summary():
    txt = "\n".join(LINES) + "\n"
    for p in (f"{OUTDIR}/step2_summary.txt", f"{BASE}/step2_summary.txt"):
        with open(p, "w") as fh:
            fh.write(txt)


def section(t):
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


out(f"STEP 2 v2  {time.strftime('%Y-%m-%d %H:%M')}")
out(f"splits by label day: {SPLITS} | thresholds from {BASE_YEARS}")
out(f"seasons: {SEASONS}")

# ---------------------------------------------------------------- 1. load
section("1. LOAD (Himachal cells only)")
ds = xr.open_dataset(PROC)
mask = ds["hp_mask"].values.astype(bool)
ii, jj = np.where(mask)
NC = len(ii)
dates = pd.DatetimeIndex(ds["time"].values)
T = len(dates)
A = {}
need = sorted(set(FEATURE_VARS) - {"rh"} | {"t2m_max", "t2m_min", "t2m_mean", "d2m"})
for v in need + ["rh"]:
    progress(f"loading {v}")
    A[v] = ds[v].values[:, mask].astype("float32")          # (days, cells)
cells = pd.DataFrame({"cell": np.arange(NC, dtype="int16"),
                      "lat": ds.lat.values[ii].round(2), "lon": ds.lon.values[jj].round(2),
                      "elev_m": ds["elev_mean"].values[mask].round(1),
                      "slope_deg": ds["slope_mean"].values[mask].round(2)})
out(f"{T} days ({dates[0].date()} to {dates[-1].date()}), {NC} cells")

# ---------------------------------------------------------------- 2. refill Tmean 1981-85, humidity
section("2. REFILL t2m_mean FROM RAW FILES, RECOMPUTE HUMIDITY")
gap_years = sorted({d.year for d, bad in zip(dates, np.isnan(A["t2m_mean"]).all(axis=1)) if bad})
out(f"years with empty t2m_mean: {gap_years}")
for y in gap_years:
    pat = re.compile(rf"^t2m_mean_{y}( \(\d+\))?\.nc$")
    fs = sorted(f for f in glob.glob(f"{RAW_DAILY}/t2m_mean_{y}*.nc") if pat.match(os.path.basename(f)))
    if not fs:
        out(f"   {y}: NO raw file, left empty")
        continue
    r = xr.open_dataset(fs[0])
    r = r.rename({k: v for k, v in {"valid_time": "time", "latitude": "lat", "longitude": "lon"}.items()
                  if k in r.dims or k in r.coords})
    rv = list(r.data_vars)[0]
    g = r[rv].reindex(lat=ds.lat.values, lon=ds.lon.values, method="nearest", tolerance=1e-3)
    vals = g.values[:, mask].astype("float32")
    if np.nanmedian(vals) > 150:                       # Kelvin -> Celsius
        vals = vals - 273.15
    idx = dates.get_indexer(pd.DatetimeIndex(g["time"].values).floor("D"))
    ok = idx >= 0
    tgt = A["t2m_mean"][idx[ok]]
    A["t2m_mean"][idx[ok]] = np.where(np.isnan(tgt), vals[ok], tgt)
    out(f"   {y}: filled from {os.path.basename(fs[0])}, {int(ok.sum())} days")


def magnus_rh(t, td):
    a, b = 17.625, 243.04
    return np.clip(100 * np.exp(a * td / (b + td) - a * t / (b + t)), 0, 100).astype("float32")


rh_calc = magnus_rh(A["t2m_mean"], A["d2m"])
both = np.isfinite(A["rh"]) & np.isfinite(rh_calc)
d = np.abs(A["rh"][both] - rh_calc[both])
out(f"humidity check where the old file has it: our formula vs stored rh, "
    f"median diff {np.median(d):.3f}, 99th pct {np.percentile(d, 99):.3f}, max {d.max():.3f} (% RH)")
A["rh"] = np.where(np.isfinite(A["rh"]), A["rh"], rh_calc).astype("float32")
for v in FEATURE_VARS:
    out(f"   {v:<9} missing after refill: {100 * np.isnan(A[v]).mean():.3f}%")
save_summary()

# ---------------------------------------------------------------- 3. thresholds
section(f"3. THRESHOLDS PER CELL ({BASE_YEARS[0]}-{BASE_YEARS[1]})")
doy = dates.dayofyear.values.copy()
doy[dates.is_leap_year & (dates.month > 2)] -= 1          # 1..365, 29 Feb shares 28 Feb's slot
doy = np.minimum(doy, 365)
month_to_season = {m: k for k, s in enumerate(SNAMES) for m in SEASONS[s]}
sidx = np.array([month_to_season[m] for m in dates.month], dtype="int8")   # season of each day
in_base = (dates.year >= BASE_YEARS[0]) & (dates.year <= BASE_YEARS[1])
TX, TN = A["t2m_max"], A["t2m_min"]

# (a) day of year, +/- WINDOW days
thr_hot_doy = np.empty((365, NC), "float32")
thr_cold_doy = np.empty((365, NC), "float32")
clim_tx = np.empty((365, NC), "float32")
clim_tn = np.empty((365, NC), "float32")
for k in range(1, 366):
    dist = np.abs(doy - k)
    dist = np.minimum(dist, 365 - dist)
    sel = in_base & (dist <= WINDOW)
    thr_hot_doy[k - 1] = np.nanpercentile(TX[sel], HOT_Q, axis=0)
    thr_cold_doy[k - 1] = np.nanpercentile(TN[sel], COLD_Q, axis=0)
    clim_tx[k - 1] = np.nanmean(TX[sel], axis=0)
    clim_tn[k - 1] = np.nanmean(TN[sel], axis=0)
# (b) season
thr_hot_sea = np.empty((len(SNAMES), NC), "float32")
thr_cold_sea = np.empty((len(SNAMES), NC), "float32")
for k in range(len(SNAMES)):
    sel = in_base & (sidx == k)
    thr_hot_sea[k] = np.nanpercentile(TX[sel], HOT_Q, axis=0)
    thr_cold_sea[k] = np.nanpercentile(TN[sel], COLD_Q, axis=0)
# (c) whole year
thr_hot_ann = np.nanpercentile(TX[in_base], HOT_Q, axis=0).astype("float32")
thr_cold_ann = np.nanpercentile(TN[in_base], COLD_Q, axis=0).astype("float32")
progress("thresholds done")

xr.Dataset({"thr_heat_doy": (("doy", "cell"), thr_hot_doy), "thr_cold_doy": (("doy", "cell"), thr_cold_doy),
            "clim_tmax": (("doy", "cell"), clim_tx), "clim_tmin": (("doy", "cell"), clim_tn),
            "thr_heat_season": (("season", "cell"), thr_hot_sea),
            "thr_cold_season": (("season", "cell"), thr_cold_sea),
            "thr_heat_annual": (("cell",), thr_hot_ann), "thr_cold_annual": (("cell",), thr_cold_ann)},
           coords={"doy": np.arange(1, 366), "cell": cells.cell.values, "season": SNAMES},
           attrs={"seasons": str(SEASONS), "base_years": str(BASE_YEARS),
                  "heat_q": HOT_Q, "cold_q": COLD_Q}).to_netcdf(f"{OUTDIR}/thresholds.nc")
cells.to_csv(f"{OUTDIR}/cells.csv", index=False)

bands = pd.cut(cells.elev_m, [0, 1000, 2000, 3000, 4000, 5000, 7000])
heat_tab = pd.DataFrame({s: thr_hot_sea[k] for k, s in enumerate(SNAMES)} | {"annual": thr_hot_ann})
heat_tab = heat_tab.groupby(bands, observed=True).mean().round(1)
heat_tab["cells"] = bands.value_counts().reindex(heat_tab.index).values
out("HEAT threshold (Tmax 99th pct, C), mean by elevation band:")
out(heat_tab.to_string())
cold_tab = pd.DataFrame({s: thr_cold_sea[k] for k, s in enumerate(SNAMES)} | {"annual": thr_cold_ann})
out("\nCOLD threshold (Tmin 1st pct, C), mean by elevation band:")
out(cold_tab.groupby(bands, observed=True).mean().round(1).to_string())
save_summary()

# ---------------------------------------------------------------- 4. labels
section("4. LABELS (extreme on a given day)")
THR = {"heat_doy": thr_hot_doy[doy - 1], "heat_season": thr_hot_sea[sidx],
       "heat_annual": np.broadcast_to(thr_hot_ann, TX.shape),
       "cold_doy": thr_cold_doy[doy - 1], "cold_season": thr_cold_sea[sidx],
       "cold_annual": np.broadcast_to(thr_cold_ann, TN.shape)}
LAB = {k: (TX > v) if k.startswith("heat") else (TN < v) for k, v in THR.items()}
yrs = dates.year

out("Share of cell-days that are extreme (%), whole period, by label:")
out(pd.Series({k: round(100 * v.mean(), 2) for k, v in LAB.items()}).to_string())

for kind in ("heat", "cold"):
    lab = LAB[f"{kind}_season"]
    t = pd.DataFrame({s: pd.Series(lab.mean(axis=1), index=yrs)[sidx == k].groupby(level=0).mean() * 100
                      for k, s in enumerate(SNAMES)})
    t["annual"] = pd.Series(LAB[f"{kind}_annual"].mean(axis=1), index=yrs).groupby(level=0).mean() * 100
    out(f"\n{kind.upper()}: % of cell-days extreme by year, each season vs the same season in other years"
        f" (plus whole-year label). {BASE_YEARS[0]}-{BASE_YEARS[1]} average ~1%:")
    out(t.round(2).to_string())

mon = pd.DataFrame({k: pd.Series(v.mean(axis=1), index=dates.month).groupby(level=0).mean() * 100
                    for k, v in LAB.items()}).round(2)
out("\nBy calendar month (%), all labels:")
out(mon.to_string())

for kind in ("heat", "cold"):
    out(f"\nMost widespread {kind} days per season (cells extreme out of {NC}), season label:")
    n = pd.Series(LAB[f"{kind}_season"].sum(axis=1), index=dates)
    for k, s in enumerate(SNAMES):
        top = n[sidx == k].sort_values(ascending=False).head(4)
        out(f"   {s:<13} " + ", ".join(f"{d.date()} ({c})" for d, c in top.items()))
    n = pd.Series(LAB[f"{kind}_annual"].sum(axis=1), index=dates)
    top = n.sort_values(ascending=False).head(4)
    out(f"   {'whole year':<13} " + ", ".join(f"{d.date()} ({c})" for d, c in top.items()))
save_summary()

# ---------------------------------------------------------------- 5. features + write
section("5. BUILD ROWS (features on D, D-1, D-2; labels on D+1) AND WRITE")
L = max(LAGS)
D = np.arange(L, T - 1)                     # feature day indices; label day = D + 1
anom_tx = TX - clim_tx[doy - 1]
anom_tn = TN - clim_tn[doy - 1]
label_year = dates.year.values[D + 1]
file_info = []
for name, (y0, y1) in SPLITS.items():
    sel = D[(label_year >= y0) & (label_year <= y1)]
    if len(sel) == 0:
        out(f"{name}: no days")
        continue
    progress(f"building {name} ({len(sel)} days x {NC} cells)")
    nx = sel + 1
    cols = {"date": np.repeat(dates.values[sel], NC),
            "cell": np.tile(cells.cell.values, len(sel)),
            "season_next": np.repeat(sidx[nx], NC)}
    for c in ("lat", "lon", "elev_m", "slope_deg"):
        cols[c] = np.tile(cells[c].values.astype("float32"), len(sel))
    ang = 2 * np.pi * (doy[nx] - 1) / 365.0
    cols["doy_sin"] = np.repeat(np.sin(ang).astype("float32"), NC)
    cols["doy_cos"] = np.repeat(np.cos(ang).astype("float32"), NC)
    for v in FEATURE_VARS:
        for lag in LAGS:
            cols[f"{v}_l{lag}"] = A[v][sel - lag].ravel()
    cols["tmax_anom_l0"] = anom_tx[sel].ravel()
    cols["tmin_anom_l0"] = anom_tn[sel].ravel()
    # targets on day D+1
    cols["tmax_next"] = TX[nx].ravel()
    cols["tmin_next"] = TN[nx].ravel()
    for k in THR:
        cols[f"y_{k}"] = LAB[k][nx].ravel().astype("int8")
    for k in THR:
        thr = np.asarray(THR[k])[nx]
        cols[f"thr_{k}_next"] = thr.ravel()
        if k.startswith("heat"):
            cols[f"excess_{k}_next"] = (TX[nx] - thr).ravel()       # > 0 on extreme days
        else:
            cols[f"excess_{k}_next"] = (thr - TN[nx]).ravel()       # > 0 on extreme days
    df = pd.DataFrame(cols)
    path = f"{OUTDIR}/{name}.parquet"
    df.to_parquet(path, index=False, compression="zstd")
    row = dict(split=name, label_years=f"{y0}-{y1}", rows=len(df))
    for k in THR:
        row[f"{k}_%"] = round(100 * df[f"y_{k}"].mean(), 2)
    row["missing_%"] = round(100 * float(df.isna().mean().max()), 3)
    row["size_mb"] = round(os.path.getsize(path) / 1e6, 1)
    file_info.append(row)
    feat_cols = [c for c in df.columns if c not in ("date", "cell", "season_next")
                 and not c.startswith(("y_", "thr_", "excess_", "tmax_next", "tmin_next"))]
    del df, cols
info = pd.DataFrame(file_info)
out(info.to_string(index=False))
out(f"\n{len(feat_cols)} feature columns: {feat_cols}")
out("label columns: " + ", ".join(f"y_{k}" for k in THR))
out("helper columns: season_next (" + ", ".join(f"{k}={s}" for k, s in enumerate(SNAMES))
    + "), tmax_next, tmin_next, thr_<label>_next, excess_<label>_next (how far past the threshold; EVT input)")
out(f"\nwritten to {OUTDIR}")
out(f"END  (took {(time.time() - T0) / 60:.1f} min)")
save_summary()
print(f"\nSummary saved to {BASE}/step2_summary.txt")
