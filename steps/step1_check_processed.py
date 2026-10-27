# ============================================================================
# STEP 1: CHECK THE PROCESSED DATA AGAINST THE RAW FILES   (read-only)
# Paste into ONE Colab cell and run. Takes about 5-10 minutes.
#
# Answers four questions before we build anything on himachal_daily_raw.nc:
#   A. What is in the file (dates, variables, Himachal cells)?
#   B. How complete is each variable, year by year, inside Himachal?
#   C. Do its values equal the raw ERA5-Land files (units, day alignment)?
#   D. Why is t2m_mean (and rh) empty in some years?
#   E. Which cells have the 10 m snow placeholder (glaciers)?
#   F. Basic physical sanity (Tmin <= Tmax, value ranges).
#
# Only file written: MyDrive/himachal_project/step1_check.txt
# ============================================================================
!pip install -q xarray netCDF4

import glob, os, re, time, traceback
import numpy as np
import pandas as pd
import xarray as xr
from google.colab import drive

drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/himachal_project"
PROC = f"{BASE}/data/processed/himachal_daily_raw.nc"
RAW = f"{BASE}/data/raw/era5land"
OUT = f"{BASE}/step1_check.txt"

SAMPLE_YEARS = [1981, 1990, 1991, 2005, 2020, 2025]
COMPARE = [("daily", "t2m_max"), ("daily", "t2m_min"), ("daily", "t2m_mean"), ("daily", "d2m"),
           ("daily", "sp"), ("daily", "swvl1"), ("daily", "sd"), ("accum", "tp"), ("accum", "ssrd")]
LINES, T0 = [], time.time()
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def save():
    with open(OUT, "w") as fh:
        fh.write("\n".join(LINES) + "\n")


def progress(msg):
    print(f"   ... [{(time.time() - T0) / 60:4.1f} min] {msg}", flush=True)


def section(t):
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


def std(ds):
    ren = {k: v for k, v in {"valid_time": "time", "latitude": "lat", "longitude": "lon"}.items()
           if k in ds.dims or k in ds.coords}
    return ds.rename(ren)


def raw_files(folder, var, year):
    pat = re.compile(rf"^{re.escape(var)}_{year}( \(\d+\))?\.nc$")
    return sorted(f for f in glob.glob(f"{RAW}/{folder}/{var}_{year}*.nc") if pat.match(os.path.basename(f)))


out(f"STEP 1 CHECK  {time.strftime('%Y-%m-%d %H:%M')}")
out(f"processed file: {PROC}")
out(f"raw folder:     {RAW}")

# ---------------------------------------------------------------- A. overview
section("A. WHAT IS IN THE PROCESSED FILE")
ds = std(xr.open_dataset(PROC))
t = pd.DatetimeIndex(ds["time"].values)
out(f"dims: {dict(ds.sizes)}")
out(f"time: {t[0].date()} -> {t[-1].date()}, {len(t)} days "
    f"(a full 1981-2025 run would be {len(pd.date_range('1981-01-01', '2025-12-31'))})")
out(f"lat {float(ds.lat.min()):.2f}-{float(ds.lat.max()):.2f}, lon {float(ds.lon.min()):.2f}-{float(ds.lon.max()):.2f}")
vars3 = [v for v in ds.data_vars if ds[v].dims == ("time", "lat", "lon")]
vars2 = [v for v in ds.data_vars if v not in vars3]
out(f"daily variables ({len(vars3)}): {vars3}")
out(f"static variables: {vars2}")
for v in vars3 + vars2:
    u = ds[v].attrs.get("units", "")
    if u:
        out(f"   {v}: units = {u}")
if "hp_mask" in ds:
    mask = ds["hp_mask"].values.astype(bool)
else:
    mask = np.isfinite(ds["t2m_max"].isel(time=0).values)
out(f"cells inside Himachal: {int(mask.sum())} of {mask.size}")
save()

# ---------------------------------------------------------------- B. completeness
section("B. % OF HIMACHAL CELL-DAYS WITH A VALUE, PER YEAR")
years = t.year.values
comp, stats = {}, []
for v in vars3:
    progress(f"loading {v}")
    a = ds[v].values[:, mask]                       # (days, cells inside Himachal)
    ok = np.isfinite(a)
    comp[v] = pd.Series(ok.mean(axis=1), index=years).groupby(level=0).mean() * 100
    fin = a[ok]
    if fin.size:
        stats.append((v, fin.min(), np.median(fin), fin.max()))
    del a, ok
tab = pd.DataFrame(comp).round(1)
out(tab.to_string())
bad = {v: tab.index[tab[v] < 99.0].tolist() for v in tab.columns if (tab[v] < 99.0).any()}
out("\nvariables below 99% in some year: " + (str(bad) if bad else "none"))
out("\nvalue range inside Himachal (all years): variable, min, median, max")
for v, lo, md, hi in stats:
    out(f"   {v:<14} {lo:12.3f} {md:12.3f} {hi:12.3f}")
save()

# ---------------------------------------------------------------- C. compare with raw
section("C. PROCESSED vs RAW ERA5-LAND (5 cells x sample years)")
out("For each variable the check tries unit conversions and day shifts and reports the one that")
out("matches. Expected: temperatures = raw - 273.15, same day; tp/ssrd = raw, raw shifted one day.")
elev = ds["elev_mean"].values if "elev_mean" in ds else None
ii, jj = np.where(mask)
if elev is not None:
    order = np.argsort(elev[ii, jj])
    picks = [order[int(q * (len(order) - 1))] for q in (0.05, 0.3, 0.5, 0.7, 0.95)]
else:
    picks = list(np.linspace(0, len(ii) - 1, 5).astype(int))
cells = [(float(ds.lat[ii[p]]), float(ds.lon[jj[p]])) for p in picks]
out("cells (lat, lon, elevation m): " + ", ".join(
    f"({la:.1f}, {lo:.1f}, {elev[ii[p], jj[p]]:.0f})" if elev is not None else f"({la:.1f}, {lo:.1f})"
    for (la, lo), p in zip(cells, picks)))

TRANSFORMS = {"raw": lambda x: x, "raw-273.15": lambda x: x - 273.15,
              "raw*1000": lambda x: x * 1000, "raw/1000": lambda x: x / 1000}
rows = []
for folder, var in COMPARE:
    if var not in ds:
        out(f"   {var}: not in processed file, skipped")
        continue
    for year in SAMPLE_YEARS:
        fs = raw_files(folder, var, year)
        if not fs:
            rows.append(dict(var=var, year=year, result="NO RAW FILE"))
            continue
        progress(f"{os.path.basename(fs[0])}")
        try:
            r = std(xr.open_dataset(fs[0]))
            rv = [v for v in r.data_vars][0]
            best = None
            diffs_all = []
            for la, lo in cells:
                rs = r[rv].sel(lat=la, lon=lo, method="nearest")
                dist = max(abs(float(rs.lat) - la), abs(float(rs.lon) - lo))
                rser = pd.Series(rs.values, index=pd.DatetimeIndex(rs["time"].values).floor("D"))
                pser = ds[var].sel(lat=la, lon=lo).sel(time=str(year)).to_series()
                pser.index = pd.DatetimeIndex(pser.index).floor("D")
                diffs_all.append((rser, pser, dist))
            for tname, fn in TRANSFORMS.items():
                for shift in (-1, 0, 1):
                    errs = []
                    for rser, pser, _ in diffs_all:
                        rr = fn(rser.copy())
                        rr.index = rr.index - pd.Timedelta(days=shift)   # raw day D+shift -> D
                        j = pd.concat([pser, rr], axis=1, join="inner").dropna()
                        if len(j):
                            errs.append(np.abs(j.iloc[:, 0] - j.iloc[:, 1]).values)
                    if errs:
                        e = np.concatenate(errs)
                        cand = (np.median(e), e.max(), tname, shift, len(e))
                        if best is None or cand[0] < best[0]:
                            best = cand
            maxdist = max(d for _, _, d in diffs_all)
            if all(p.isna().all() for _, p, _ in diffs_all):
                rows.append(dict(var=var, year=year, file=os.path.basename(fs[0]),
                                 result="processed is EMPTY this year (raw file exists)"))
            elif best is None:
                rows.append(dict(var=var, year=year, result="no overlapping days"))
            else:
                med, mx, tname, shift, n = best
                verdict = "MATCH" if mx < 1e-3 * max(1.0, abs(float(np.nanmedian(diffs_all[0][1])))) else (
                    "close" if med < 0.05 else "DIFFERENT")
                rows.append(dict(var=var, year=year, file=os.path.basename(fs[0]), transform=tname,
                                 day_shift=shift, median_abs_diff=round(float(med), 5),
                                 max_abs_diff=round(float(mx), 5), n=n, grid_offset=round(maxdist, 4),
                                 result=verdict))
        except Exception as e:
            rows.append(dict(var=var, year=year, result=f"error: {e.__class__.__name__}: {str(e)[:80]}"))
out(pd.DataFrame(rows).to_string(index=False))
out("day_shift = +1 means processed day D equals raw day D+1 (the expected fix for tp and ssrd).")
save()

# ---------------------------------------------------------------- D. t2m_mean gap
section("D. WHY IS t2m_mean EMPTY IN SOME YEARS? (raw files, every year)")
out("For each year: file name(s), grid size, how far the raw grid is from the processed grid,")
out("and how much of the raw file is missing. t2m_max is shown alongside for contrast.")
plat, plon = ds.lat.values, ds.lon.values
drows = []
for var in ("t2m_mean", "t2m_max"):
    for year in range(t[0].year, t[-1].year + 1):
        fs = raw_files("daily", var, year)
        row = dict(var=var, year=year, files=len(fs), names="; ".join(os.path.basename(f) for f in fs))
        if fs:
            try:
                r = std(xr.open_dataset(fs[0]))
                rv = [v for v in r.data_vars][0]
                rl, ro = r.lat.values, r.lon.values
                row["grid"] = f"{len(rl)}x{len(ro)}"
                row["lat_offset"] = round(float(np.max(np.min(np.abs(plat[:, None] - rl[None, :]), axis=1))), 4)
                row["lon_offset"] = round(float(np.max(np.min(np.abs(plon[:, None] - ro[None, :]), axis=1))), 4)
                row["days"] = r.sizes.get("time", 0)
                x = r[rv].values
                row["raw_nan_pct"] = round(100 * float(np.mean(~np.isfinite(x))), 1)
            except Exception as e:
                row["error"] = f"{e.__class__.__name__}: {str(e)[:60]}"
        if var in tab.columns and year in tab.index:
            row["processed_pct"] = tab.loc[year, var]
        drows.append(row)
    progress(f"checked {var} raw files")
dtab = pd.DataFrame(drows)
out(dtab.to_string(index=False))
save()

# ---------------------------------------------------------------- E. snow placeholder
section("E. SNOW PLACEHOLDER CELLS (sd stuck at ~10 m water equivalent)")
if "sd" in ds:
    sd = ds["sd"].values[:, mask]
    frac = np.nanmean(sd >= 9.99, axis=0)
    glac = frac > 0.5
    out(f"cells with sd >= 9.99 m on more than half the days: {int(glac.sum())} of {mask.sum()}")
    if glac.any() and elev is not None:
        e = elev[mask][glac]
        out(f"their elevation: {e.min():.0f}-{e.max():.0f} m (median {np.median(e):.0f} m)")
        la = ds.lat.values[ii][glac]
        lo = ds.lon.values[jj][glac]
        out(f"their area: lat {la.min():.1f}-{la.max():.1f}, lon {lo.min():.1f}-{lo.max():.1f}")
    del sd
else:
    out("sd not in processed file")
save()

# ---------------------------------------------------------------- F. sanity
section("F. PHYSICAL SANITY")
if {"t2m_max", "t2m_min"} <= set(ds.data_vars):
    tx = ds["t2m_max"].values[:, mask]
    tn = ds["t2m_min"].values[:, mask]
    both = np.isfinite(tx) & np.isfinite(tn)
    viol = (tn > tx + 1e-6) & both
    out(f"cell-days with Tmin > Tmax: {int(viol.sum())} of {int(both.sum())}")
    out(f"Tmax range {np.nanmin(tx):.1f} to {np.nanmax(tx):.1f} C; Tmin range {np.nanmin(tn):.1f} to {np.nanmax(tn):.1f} C")
    if "t2m_mean" in ds:
        tm = ds["t2m_mean"].values[:, mask]
        ok = np.isfinite(tm) & both
        out(f"cell-days with Tmean outside [Tmin, Tmax] by > 0.5 C: "
            f"{int((((tm < tn - 0.5) | (tm > tx + 0.5)) & ok).sum())} of {int(ok.sum())}")
        del tm
    # share of days a cell is at its own 99th percentile (expected ~1%)
    p99 = np.nanpercentile(tx, 99, axis=0)
    out(f"check of labelling idea: share of cell-days above each cell's own 99th pct Tmax = "
        f"{100 * np.nanmean(tx > p99):.2f}%")
    del tx, tn
out(f"\nEND  (took {(time.time() - T0) / 60:.1f} min)")
save()
print(f"\nSaved to {OUT}")
