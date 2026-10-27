"""
Coverage check for the master_himachal_daily_<year>.csv files.

Answers: for which grid cells (region) and which years is the data complete?

Run in Colab (Drive mounted):
    !python coverage_check.py "/content/drive/MyDrive/himachal_project_master_output"
or point it at the other folder:
    !python coverage_check.py "/content/drive/MyDrive/himachal_project/data/processed_master_outer"

Outputs (written next to the script, in ./coverage_report/):
    year_variable_completeness.csv   % of cell-days with data, per year x variable
    cell_completeness.csv            % of days with data, per grid cell x variable
    complete_cells_map.png           map of cells with 100% temperature data
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "/content/drive/MyDrive/himachal_project_master_output"
OUT = "coverage_report"
TEMP_VARS = ["t2m_max", "t2m_min", "t2m_mean"]   # what "usable for temperature" means
ID_COLS = {"lat", "lon", "date"}

os.makedirs(OUT, exist_ok=True)
files = sorted(glob.glob(os.path.join(FOLDER, "master_himachal_daily_*.csv")))
if not files:
    sys.exit(f"No master_himachal_daily_*.csv files in {FOLDER}")
print(f"{len(files)} files in {FOLDER}\n")

year_rows = []
cell_count = None      # running non-null counts per cell x variable
cell_days = None       # running number of days per cell

for f in files:
    df = pd.read_csv(f, parse_dates=["date"])
    year = int(os.path.basename(f).split("_")[-1].split(".")[0])
    df = df[df["date"].dt.year == year]          # drop spill-over days (e.g. 31 Dec of previous year)
    vars_ = [c for c in df.columns if c not in ID_COLS and not c.startswith("Unnamed")]

    row = {"year": year, "days": df["date"].nunique(), "cells": df.groupby(["lat", "lon"]).ngroups}
    row.update({v: round(100 * df[v].notna().mean(), 1) for v in vars_})
    year_rows.append(row)

    g = df.groupby(["lat", "lon"])
    cnt = g[vars_].count()
    days = g.size().rename("days")
    cell_count = cnt if cell_count is None else cell_count.add(cnt, fill_value=0)
    cell_days = days if cell_days is None else cell_days.add(days, fill_value=0)
    print(f"{year}: {row['days']} days, {row['cells']} cells, "
          + ", ".join(f"{v} {row.get(v, 0)}%" for v in TEMP_VARS if v in row))

yv = pd.DataFrame(year_rows).set_index("year").sort_index()
yv.to_csv(os.path.join(OUT, "year_variable_completeness.csv"))

cells = cell_count.div(cell_days, axis=0).mul(100).round(1)
cells["days"] = cell_days
cells.reset_index().to_csv(os.path.join(OUT, "cell_completeness.csv"), index=False)

# ---------------------------------------------------------------- summary
print("\n=== % of cell-days with data, per year (all variables) ===")
print(yv.to_string())

tv = [v for v in TEMP_VARS if v in cells.columns]
full = cells[(cells[tv] >= 99.9).all(axis=1)].reset_index()
some = cells[(cells[tv] > 0).any(axis=1)].reset_index()

print(f"\n=== Temperature ({', '.join(tv)}) ===")
print(f"cells in files:                  {len(cells)}")
print(f"cells with ANY temperature data: {len(some)}")
print(f"cells COMPLETE for all years:    {len(full)}")
if len(full):
    print(f"complete region: lat {full.lat.min()}-{full.lat.max()} N, "
          f"lon {full.lon.min()}-{full.lon.max()} E")

# years in which temperature is complete over the cells that have data at all
if len(some):
    ok_years = [y for y in yv.index if all(yv.loc[y, v] > 0 for v in tv)]
    print(f"years with temperature data:     {ok_years[0] if ok_years else '-'}"
          f" to {ok_years[-1] if ok_years else '-'} ({len(ok_years)} years)")
    missing = [y for y in yv.index if y not in ok_years]
    if missing:
        print(f"years with NO temperature data:  {missing}")

# ---------------------------------------------------------------- map
c = cells.reset_index()
score = c[tv].min(axis=1)                          # worst of the temperature variables
fig, ax = plt.subplots(figsize=(8, 6.5))
sc = ax.scatter(c.lon, c.lat, c=score, cmap="RdYlGn", vmin=0, vmax=100, marker="s", s=60)
plt.colorbar(sc, label="% of days with temperature data (worst of Tmax/Tmin/Tmean)")
ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
ax.set_title("Temperature data completeness per 0.1° cell")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "complete_cells_map.png"), dpi=130)
print(f"\nReport written to ./{OUT}/  (map: complete_cells_map.png)")
