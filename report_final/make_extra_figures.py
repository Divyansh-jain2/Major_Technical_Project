"""Extra figures for the final report: study-area map and the EVT tail-focus target.

Run: python3 report_final/make_extra_figures.py   (reads report_final/data/cells.csv)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
ORANGE, INK, INK2, MUTED, GRID, SURF = "#eb6834", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 11, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

# ---------------------------------------------------------------- study area
cells = pd.read_csv(os.path.join(HERE, "data", "cells.csv"))
lats = np.round(np.arange(cells.lat.min(), cells.lat.max() + 0.05, 0.1), 1)
lons = np.round(np.arange(cells.lon.min(), cells.lon.max() + 0.05, 0.1), 1)
Z = np.full((len(lats), len(lons)), np.nan)
for _, r in cells.iterrows():
    Z[np.where(lats == round(r.lat, 1))[0][0], np.where(lons == round(r.lon, 1))[0][0]] = r.elev_m / 1000
seq = LinearSegmentedColormap.from_list("elev", ["#e3eefb", "#8fbbed", "#2a78d6", "#14417a", "#0a2140"])
fig, ax = plt.subplots(figsize=(7.4, 6.0))
m = ax.pcolormesh(np.r_[lons - 0.05, lons[-1] + 0.05], np.r_[lats - 0.05, lats[-1] + 0.05], Z, cmap=seq,
                  vmin=0, vmax=5.7, edgecolors=SURF, linewidth=0.3)
cb = fig.colorbar(m, ax=ax, shrink=0.8, pad=0.02)
cb.set_label("Cell elevation (km)", color=INK2)
cb.outline.set_visible(False)
stations = [("Shimla", 31.100, 77.167), ("Sundernagar", 31.533, 76.900), ("Bhuntar", 31.833, 77.167)]
for nm, la, lo in stations:
    ax.plot(lo, la, marker="^", ms=10, color=ORANGE, mec="white", mew=1.4, zorder=4)
    ax.annotate(nm, (lo, la), xytext=(8, -3), textcoords="offset points", fontsize=9, color=INK,
                fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
ax.set_aspect(1 / np.cos(np.deg2rad(31.8)))
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.grid(color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.plot([], [], marker="^", ls="", ms=9, color=ORANGE, mec="white", label="GSOD weather stations")
ax.legend(loc="upper right", fontsize=8.5)
ax.set_title("Study area: 529 ERA5-Land grid cells (0.1°) over Himachal Pradesh", loc="left")
fig.savefig(os.path.join(FIG, "fig_study_area.png"), dpi=220, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- EVT tail-focus target
bins = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%\n(rarest, most severe)"]
real = [20, 20, 20, 20, 20]
tail = [10, 10, 10, 10, 60]
x = np.arange(len(bins))
fig, ax = plt.subplots(figsize=(7.4, 3.6))
w = 0.36
b1 = ax.bar(x - w / 2 - 0.01, real, w, color=MUTED, label="real extremes / plain sampling")
b2 = ax.bar(x + w / 2 + 0.01, tail, w, color=ORANGE, label="EVT tail focus")
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.0f}%", (r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
ax.set_xticks(x, bins, fontsize=9)
ax.set_xlabel("Tail position q of the extreme (GPD probability, within its season)")
ax.set_ylabel("Share of added extremes (%)")
ax.set_ylim(0, 70)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=8.5)
ax.set_title("EVT tail focus puts 60% of added extremes in the top 20% of the tail", loc="left")
fig.savefig(os.path.join(FIG, "fig_tail_focus.png"), dpi=220, bbox_inches="tight")
plt.close(fig)
print("written: fig_study_area.png, fig_tail_focus.png")
