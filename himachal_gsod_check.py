"""
Find NOAA GSOD weather stations in/around Himachal Pradesh and check how
complete their daily temperature records are. Optionally saves the daily
Tmax/Tmin/Tmean (in °C) for each station so you can use them as ground truth.

Usage:
    pip install pandas requests
    python himachal_gsod_check.py                    # 2000-2025, summary only
    python himachal_gsod_check.py --start 1990 --end 2025 --save-daily

Outputs (in ./gsod_himachal/):
    stations_summary.csv   one row per station: elevation, years, % complete
    yearly_completeness.csv  station x year table of valid Tmax/Tmin days
    daily/<id>_<name>.csv  daily data in °C (only with --save-daily)

Notes on GSOD:
  * Temperatures are in °F in the source files; 9999.9 means missing.
  * A '*' flag on MAX/MIN means it was derived from the hourly reports rather
    than reported directly — usable, but slightly less trustworthy.
  * GSOD is built from WMO synoptic messages, so values are station
    observations (not interpolated), but they are not IMD-quality-controlled.
"""

import argparse
import io
import os
import time

import pandas as pd
import requests

ISD_HISTORY = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
GSOD_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{sid}.csv"

# Bounding box around Himachal Pradesh (slightly padded). It will also catch a
# few stations just across the border (Punjab, J&K/Ladakh, Uttarakhand), which
# are flagged with in_hp_core=False so you can decide whether to keep them.
BBOX = dict(lat_min=30.3, lat_max=33.3, lon_min=75.5, lon_max=79.1)
HP_CORE = dict(lat_min=30.38, lat_max=33.22, lon_min=75.58, lon_max=79.0)

MISSING = 9999.9


def f_to_c(series):
    return ((series - 32) * 5 / 9).round(2)


def get_stations():
    df = pd.read_csv(ISD_HISTORY, dtype={"USAF": str, "WBAN": str})
    df = df[df["CTRY"] == "IN"].copy()
    df = df[
        df["LAT"].between(BBOX["lat_min"], BBOX["lat_max"])
        & df["LON"].between(BBOX["lon_min"], BBOX["lon_max"])
    ]
    df["sid"] = df["USAF"].str.zfill(6) + df["WBAN"].str.zfill(5)
    df["begin_year"] = df["BEGIN"].astype(str).str[:4].astype(int)
    df["end_year"] = df["END"].astype(str).str[:4].astype(int)
    df["in_hp_core"] = (
        df["LAT"].between(HP_CORE["lat_min"], HP_CORE["lat_max"])
        & df["LON"].between(HP_CORE["lon_min"], HP_CORE["lon_max"])
    )
    return df.sort_values("LAT").reset_index(drop=True)


def fetch_year(session, sid, year):
    r = session.get(GSOD_URL.format(year=year, sid=sid), timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--save-daily", action="store_true")
    ap.add_argument("--out", default="gsod_himachal")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.save_daily:
        os.makedirs(os.path.join(args.out, "daily"), exist_ok=True)

    stations = get_stations()
    print(f"Found {len(stations)} station entries in the Himachal box:\n")
    print(stations[["sid", "STATION NAME", "LAT", "LON", "ELEV(M)",
                    "BEGIN", "END", "in_hp_core"]].to_string(index=False))
    print()

    session = requests.Session()
    yearly_rows, summary_rows = [], []

    for _, st in stations.iterrows():
        y0 = max(args.start, st["begin_year"])
        y1 = min(args.end, st["end_year"])
        daily_parts = []
        print(f"Checking {st['STATION NAME']} ({st['sid']}) {y0}-{y1} ...")

        for year in range(y0, y1 + 1):
            try:
                d = fetch_year(session, st["sid"], year)
            except Exception as e:  # network hiccup: record and move on
                print(f"   {year}: error {e}")
                d = None
            time.sleep(0.2)  # be polite to NOAA's server

            if d is None or d.empty:
                yearly_rows.append(dict(sid=st["sid"], year=year,
                                        days=0, tmax_ok=0, tmin_ok=0))
                continue

            tmax_ok = (d["MAX"] != MISSING)
            tmin_ok = (d["MIN"] != MISSING)
            yearly_rows.append(dict(
                sid=st["sid"], year=year, days=len(d),
                tmax_ok=int(tmax_ok.sum()), tmin_ok=int(tmin_ok.sum()),
                tmax_derived=int((d["MAX_ATTRIBUTES"].astype(str).str.strip() == "*").sum()),
            ))

            if args.save_daily:
                out = pd.DataFrame({
                    "date": pd.to_datetime(d["DATE"]),
                    "tmean_c": f_to_c(d["TEMP"].where(d["TEMP"] != MISSING)),
                    "tmax_c": f_to_c(d["MAX"].where(tmax_ok)),
                    "tmin_c": f_to_c(d["MIN"].where(tmin_ok)),
                    "tmax_derived": d["MAX_ATTRIBUTES"].astype(str).str.strip() == "*",
                    "tmin_derived": d["MIN_ATTRIBUTES"].astype(str).str.strip() == "*",
                })
                daily_parts.append(out)

        st_years = [r for r in yearly_rows if r["sid"] == st["sid"]]
        possible = sum(366 if pd.Timestamp(year=r["year"], month=1, day=1).is_leap_year
                       else 365 for r in st_years) or 1
        tmax_total = sum(r["tmax_ok"] for r in st_years)
        tmin_total = sum(r["tmin_ok"] for r in st_years)
        summary_rows.append(dict(
            sid=st["sid"], name=st["STATION NAME"], lat=st["LAT"], lon=st["LON"],
            elev_m=st["ELEV(M)"], in_hp_core=st["in_hp_core"],
            years_with_data=sum(1 for r in st_years if r["days"] > 0),
            tmax_complete_pct=round(100 * tmax_total / possible, 1),
            tmin_complete_pct=round(100 * tmin_total / possible, 1),
            usable_years_80pct=sum(1 for r in st_years
                                   if min(r["tmax_ok"], r["tmin_ok"]) >= 292),
        ))

        if args.save_daily and daily_parts:
            safe = "".join(c if c.isalnum() else "_" for c in str(st["STATION NAME"]))
            pd.concat(daily_parts).to_csv(
                os.path.join(args.out, "daily", f"{st['sid']}_{safe}.csv"), index=False)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["usable_years_80pct", "tmax_complete_pct"], ascending=False)
    summary.to_csv(os.path.join(args.out, "stations_summary.csv"), index=False)

    yearly = pd.DataFrame(yearly_rows)
    yearly.to_csv(os.path.join(args.out, "yearly_completeness.csv"), index=False)

    print("\n=== Station summary (sorted by usable years) ===")
    print(summary.to_string(index=False))
    print(f"\n'usable_years_80pct' = years with >=80% of days having both Tmax and Tmin.")
    print(f"Files written to ./{args.out}/")


if __name__ == "__main__":
    main()
