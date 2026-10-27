# ============================================================================
# STEP 2: BUILD THE MODEL-READY DATASET (next-day temperature extremes)
# Paste into ONE Colab cell and run. Takes about 5-10 minutes.
#
# Reads (never changes):
#   data/processed/himachal_daily_raw.nc            (checked in Step 1)
#   data/raw/era5land/daily/t2m_mean_198X*.nc       (to refill Tmean 1981-1985)
# Writes ONLY to a new folder:  himachal_project/data/model_ready/
#   train.parquet  1981-2010 label days, all rows (natural ~1% imbalance)
#   val.parquet    2011-2015
#   test.parquet   2016-2025
#   thresholds.nc  per cell and day of year: 99th pct Tmax, 1st pct Tmin, climatology
#   cells.csv      cell id, lat, lon, elevation, slope
#   step2_summary.txt   (also copied to himachal_project/step2_summary.txt)
#
# One row = one Himachal grid cell on day D.
#   Features: weather on D, D-1, D-2 + terrain + season.
#   Labels:   is day D+1 an extreme for this cell?
#     y_heat        Tmax(D+1) > cell's 99th pct for that time of year  (main target)
#     y_heat_annual Tmax(D+1) > cell's 99th pct over the whole year    (alternative)
#     y_cold        Tmin(D+1) < cell's 1st pct for that time of year   (for later)
#   Thresholds use 1981-2010 only, so nothing from val/test leaks into them.
# ============================================================================
!pip install -q xarray netCDF4 pyarrow

import glob, os, re, shutil, time
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
WINDOW = 7                         # +/- days around each day of year for percentiles
HOT_Q, COLD_Q = 99, 1
LAGS = (0, 1, 2)                   # D, D-1, D-2
# processed-file units (verified in Step 1): temps C, sp hPa, ssrd W m-2, tp mm/day
FEATURE_VARS = ["t2m_max", "t2m_min", "t2m_mean", "d2m", "rh", "sp", "ssrd", "swvl1", "tp"]
# ----------------------------------------------------------------

os.makedirs(OUTDIR, exist_ok=True)
LINES, T0 = [], time.time()
pd.set_option("display.width", 250)


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


out(f"STEP 2  {time.strftime('%Y-%m-%d %H:%M')}")
out(f"splits by label day: {SPLITS} | thresholds from {BASE_YEARS} | window +/-{WINDOW} days")

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
section(f"3. THRESHOLDS PER CELL AND DAY OF YEAR ({BASE_YEARS[0]}-{BASE_YEARS[1]})")
doy = dates.dayofyear.values.copy()
leap_after_feb = dates.is_leap_year & (dates.month > 2)
doy[leap_after_feb] -= 1                     # 1..365, 29 Feb shares 28 Feb's slot
doy = np.minimum(doy, 365)
in_base = (dates.year >= BASE_YEARS[0]) & (dates.year <= BASE_YEARS[1])
thr_hot = np.empty((365, NC), "float32")
thr_cold = np.empty((365, NC), "float32")
clim_tx = np.empty((365, NC), "float32")
clim_tn = np.empty((365, NC), "float32")
for k in range(1, 366):
    dist = np.abs(doy - k)
    dist = np.minimum(dist, 365 - dist)
    sel = in_base & (dist <= WINDOW)
    tx, tn = A["t2m_max"][sel], A["t2m_min"][sel]
    thr_hot[k - 1] = np.nanpercentile(tx, HOT_Q, axis=0)
    thr_cold[k - 1] = np.nanpercentile(tn, COLD_Q, axis=0)
    clim_tx[k - 1] = np.nanmean(tx, axis=0)
    clim_tn[k - 1] = np.nanmean(tn, axis=0)
thr_hot_annual = np.nanpercentile(A["t2m_max"][in_base], HOT_Q, axis=0).astype("float32")
progress("thresholds done")
xr.Dataset({"thr_heat_p99": (("doy", "cell"), thr_hot), "thr_cold_p1": (("doy", "cell"), thr_cold),
            "clim_tmax": (("doy", "cell"), clim_tx), "clim_tmin": (("doy", "cell"), clim_tn),
            "thr_heat_annual_p99": (("cell",), thr_hot_annual)},
           coords={"doy": np.arange(1, 366), "cell": cells.cell.values}).to_netcdf(f"{OUTDIR}/thresholds.nc")
cells.to_csv(f"{OUTDIR}/cells.csv", index=False)
bands = pd.cut(cells.elev_m, [0, 1000, 2000, 3000, 4000, 5000, 7000])
tab = pd.DataFrame({"band": bands, "summer_thr": thr_hot[171], "winter_thr": thr_hot[14],
                    "annual_thr": thr_hot_annual}).groupby("band", observed=True).mean().round(1)
tab["cells"] = bands.value_counts().reindex(tab.index).values
out("mean heat threshold (Tmax 99th pct, C) by elevation band, for 15 Jan, 21 Jun and the whole year:")
out(tab.to_string())

# ---------------------------------------------------------------- 4. labels
section("4. NEXT-DAY LABELS")
TXT = thr_hot[doy - 1]                      # (days, cells) threshold for each day
TNT = thr_cold[doy - 1]
heat = A["t2m_max"] > TXT
heat_a = A["t2m_max"] > thr_hot_annual[None, :]
cold = A["t2m_min"] < TNT
yrs = dates.year
rate = pd.DataFrame({"heat_%": heat.mean(axis=1), "heat_annual_%": heat_a.mean(axis=1),
                     "cold_%": cold.mean(axis=1)}, index=yrs).groupby(level=0).mean().mul(100).round(2)
out(f"share of cell-days that are extreme, by year ({BASE_YEARS[0]}-{BASE_YEARS[1]} should average ~1%; later years show any trend):")
out(rate.to_string())
mon = pd.DataFrame({"heat_%": heat.mean(axis=1), "heat_annual_%": heat_a.mean(axis=1),
                    "cold_%": cold.mean(axis=1)}, index=dates.month).groupby(level=0).mean().mul(100).round(2)
out("\nby month (seasonal labels spread over the year; the annual label sits in May-June):")
out(mon.to_string())
ncell_heat = pd.Series(heat.sum(axis=1), index=dates)
out(f"\nmost widespread heat days (cells extreme out of {NC}), whole period:")
out(ncell_heat.sort_values(ascending=False).head(12).to_string())
ncell_cold = pd.Series(cold.sum(axis=1), index=dates)
out("\nmost widespread cold days:")
out(ncell_cold.sort_values(ascending=False).head(8).to_string())
save_summary()

# ---------------------------------------------------------------- 5. features + write
section("5. BUILD ROWS AND WRITE PARQUET FILES")
L = max(LAGS)
D = np.arange(L, T - 1)                     # feature day indices; label day = D + 1
anom_tx = A["t2m_max"] - clim_tx[doy - 1]
anom_tn = A["t2m_min"] - clim_tn[doy - 1]
label_year = dates.year.values[D + 1]
file_info = []
for name, (y0, y1) in SPLITS.items():
    sel = D[(label_year >= y0) & (label_year <= y1)]
    if len(sel) == 0:
        out(f"{name}: no days")
        continue
    progress(f"building {name} ({len(sel)} days x {NC} cells)")
    n = len(sel) * NC
    cols = {"date": np.repeat(dates.values[sel], NC),
            "cell": np.tile(cells.cell.values, len(sel))}
    for c in ("lat", "lon", "elev_m", "slope_deg"):
        cols[c] = np.tile(cells[c].values.astype("float32"), len(sel))
    ang = 2 * np.pi * (doy[sel + 1] - 1) / 365.0
    cols["doy_sin"] = np.repeat(np.sin(ang).astype("float32"), NC)
    cols["doy_cos"] = np.repeat(np.cos(ang).astype("float32"), NC)
    for v in FEATURE_VARS:
        for lag in LAGS:
            cols[f"{v}_l{lag}"] = A[v][sel - lag].ravel()
    cols["tmax_anom_l0"] = anom_tx[sel].ravel()
    cols["tmin_anom_l0"] = anom_tn[sel].ravel()
    # targets (day D+1)
    cols["y_heat"] = heat[sel + 1].ravel().astype("int8")
    cols["y_heat_annual"] = heat_a[sel + 1].ravel().astype("int8")
    cols["y_cold"] = cold[sel + 1].ravel().astype("int8")
    cols["tmax_next"] = A["t2m_max"][sel + 1].ravel()
    cols["thr_heat_next"] = TXT[sel + 1].ravel()
    cols["heat_excess_next"] = (A["t2m_max"][sel + 1] - TXT[sel + 1]).ravel()
    cols["tmin_next"] = A["t2m_min"][sel + 1].ravel()
    cols["thr_cold_next"] = TNT[sel + 1].ravel()
    df = pd.DataFrame(cols)
    path = f"{OUTDIR}/{name}.parquet"
    df.to_parquet(path, index=False, compression="zstd")
    file_info.append(dict(split=name, label_years=f"{y0}-{y1}", rows=len(df),
                          heat=int(df.y_heat.sum()), heat_pct=round(100 * df.y_heat.mean(), 2),
                          heat_annual_pct=round(100 * df.y_heat_annual.mean(), 2),
                          cold_pct=round(100 * df.y_cold.mean(), 2),
                          missing_pct=round(100 * float(df.isna().mean().max()), 3),
                          size_mb=round(os.path.getsize(path) / 1e6, 1)))
    feat_cols = [c for c in df.columns if c not in ("date", "cell") and not c.startswith(("y_", "tmax_next",
                 "thr_", "heat_excess", "tmin_next"))]
    del df, cols
info = pd.DataFrame(file_info)
out(info.to_string(index=False))
out(f"\n{len(feat_cols)} feature columns: {feat_cols}")
out("target and helper columns: y_heat (main), y_heat_annual, y_cold, tmax_next, thr_heat_next, "
    "heat_excess_next (for EVT), tmin_next, thr_cold_next")
out(f"\nwritten to {OUTDIR}")
out(f"END  (took {(time.time() - T0) / 60:.1f} min)")
save_summary()
print(f"\nSummary saved to {BASE}/step2_summary.txt")
