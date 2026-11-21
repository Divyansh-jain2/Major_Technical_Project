"""Figures for the 'Weather and Climate of India' chapter.

Usage: python make_weather_figures.py <countries_ind.geojson> <admin1_states.geojson>
Boundaries: Natural Earth 10m, India point-of-view country outline + admin-1 states.
Writes figures/fig_india_regions.png, fig_india_climate.png, fig_hp_zones.png.
"""
import sys
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
from shapely.ops import unary_union

country_path, states_path = sys.argv[1], sys.argv[2]
OUT = "figures/"

ind = gpd.read_file(country_path)
ind = ind[ind.ADMIN == "India"].geometry.iloc[0]
st = gpd.read_file(states_path)
st = st[st.admin == "India"][["name", "geometry"]].reset_index(drop=True)

# IMD's four homogeneous rainfall regions, approximated by state.
REGION = {
    "Northwest India": ["Jammu and Kashmir", "Ladakh", "Himachal Pradesh", "Uttarakhand", "Punjab",
                        "Haryana", "Chandigarh", "Delhi", "Rajasthan", "Uttar Pradesh"],
    "Central India": ["Gujarat", "Madhya Pradesh", "Chhattisgarh", "Maharashtra", "Goa", "Odisha",
                      "Dadra and Nagar Haveli and Daman and Diu"],
    "South Peninsula": ["Andhra Pradesh", "Telangana", "Karnataka", "Kerala", "Tamil Nadu",
                        "Puducherry", "Lakshadweep", "Andaman and Nicobar"],
    "East & Northeast India": ["Bihar", "Jharkhand", "West Bengal", "Sikkim", "Assam", "Meghalaya",
                               "Arunachal Pradesh", "Nagaland", "Manipur", "Mizoram", "Tripura"],
}
COL = {"Northwest India": "#e8b04b", "Central India": "#7fb77e",
       "South Peninsula": "#6fa8dc", "East & Northeast India": "#b48ec7"}
LPA = {"Northwest India": 587.6, "Central India": 978.0,
       "South Peninsula": 716.2, "East & Northeast India": 1367.3}
lookup = {s: r for r, ss in REGION.items() for s in ss}
st["region"] = st.name.map(lookup)
assert st.region.notna().all(), st[st.region.isna()]

# Areas inside India's official outline that the state layer leaves out: give each piece the
# region of the nearest state.
rest = ind.difference(unary_union(st.geometry))
pieces = list(getattr(rest, "geoms", [rest]))
extra = []
for p in pieces:
    if p.area < 1e-4:
        continue
    near = st.geometry.distance(p.representative_point()).idxmin()
    extra.append({"name": "rest", "region": st.region[near], "geometry": p})
allg = pd.concat([st, gpd.GeoDataFrame(extra, crs=st.crs)], ignore_index=True)
reg = allg.dissolve(by="region")
hp = st[st.name == "Himachal Pradesh"].geometry.iloc[0]

def base(ax):
    ax.set_xlim(67.5, 98); ax.set_ylim(6, 37.6)
    ax.set_aspect(1 / 0.93)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.tick_params(labelsize=8)

# ---------- Figure 1: IMD regions ----------
fig, ax = plt.subplots(figsize=(6.6, 6.8))
for r, row in reg.iterrows():
    gpd.GeoSeries([row.geometry]).plot(ax=ax, color=COL[r], edgecolor="white", linewidth=0.3)
st.boundary.plot(ax=ax, color="white", linewidth=0.3)
gpd.GeoSeries([ind]).boundary.plot(ax=ax, color="#333333", linewidth=0.7)
gpd.GeoSeries([hp]).plot(ax=ax, facecolor="none", edgecolor="#c0392b", linewidth=1.6, hatch="////")
ax.annotate("Himachal Pradesh\n(study area)", xy=(77.2, 31.9), xytext=(69.0, 34.6), fontsize=8.5,
            color="#c0392b", arrowprops=dict(arrowstyle="->", color="#c0392b"))
handles = [Patch(facecolor=COL[r], label=f"{r}  ({LPA[r]:.0f} mm)") for r in REGION]
ax.legend(handles=handles, loc="upper right", fontsize=8,
          title="IMD region (June–Sept. normal rainfall)", title_fontsize=8, frameon=True)
base(ax)
fig.tight_layout(); fig.savefig(OUT + "fig_india_regions.png", dpi=220); plt.close(fig)

# ---------- Figure 2: climate types and the winds that drive them ----------
fig, ax = plt.subplots(figsize=(6.6, 6.8))
gpd.GeoSeries([ind]).plot(ax=ax, color="#f3efe6", edgecolor="#333333", linewidth=0.7)
st.boundary.plot(ax=ax, color="#bbbbbb", linewidth=0.3)
gpd.GeoSeries([hp]).plot(ax=ax, color="#e74c3c", alpha=0.85)

def lab(x, y, t, c="#222222", fs=7.8):
    ax.text(x, y, t, fontsize=fs, ha="center", va="center", color=c,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=c, lw=0.6, alpha=0.9))
lab(77.6, 35.0, "Cold desert / alpine\n(Ladakh, Lahaul–Spiti)", "#1f4e79")
lab(71.6, 27.0, "Hot desert\n(Thar)", "#a0522d")
lab(80.2, 26.9, "Humid subtropical\n(Indo-Gangetic plains)", "#6b8e23")
lab(78.2, 21.4, "Tropical wet and dry\n(central India)", "#8b6914")
lab(77.2, 16.2, "Semi-arid\n(Deccan rain shadow)", "#a0522d")
lab(77.4, 10.2, "Tropical wet\n(Western Ghats coast)", "#1e7b4f")
lab(93.6, 25.0, "Humid / very wet\n(Northeast)", "#1e7b4f")

def arrow(xy0, xy1, c, txt=None, txy=None, rad=0.0):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle="-|>", mutation_scale=16, lw=2.2, color=c,
                                 connectionstyle=f"arc3,rad={rad}"))
    if txt:
        ax.text(*txy, txt, fontsize=7.6, color=c, ha="center", fontweight="bold")
BLUE, GREEN, RED, PURPLE = "#1f77b4", "#2ca02c", "#d62728", "#7b3294"
arrow((67.5, 13.2), (73.2, 17.8), BLUE, "SW monsoon\n(Arabian Sea branch)\nJun–Sep", (68.4, 10.2), rad=-0.15)
arrow((86.0, 13.0), (89.5, 22.4), BLUE, "SW monsoon\n(Bay of Bengal\nbranch)", (93.2, 18.0), rad=0.2)
arrow((89.5, 24.3), (80.5, 29.0), BLUE, rad=0.15)
arrow((88.0, 17.5), (81.2, 12.6), PURPLE, "NE monsoon\nOct–Dec", (85.6, 10.6), rad=0.1)
arrow((67.8, 33.8), (74.0, 33.0), GREEN, "Western disturbances\n(winter snow and rain)", (68.4, 36.0), rad=-0.1)
arrow((70.6, 25.0), (76.4, 28.1), RED, "Loo: hot dry wind\n(Apr–Jun)", (73.0, 23.0), rad=0.1)
ax.text(79.6, 31.7, "HP", fontsize=8.5, color="#c0392b", fontweight="bold")
ax.text(82.5, 31.0, "Himalaya: barrier to cold\nCentral-Asian air; lifts monsoon air",
        fontsize=7.0, color="#555555", ha="left", rotation=-18)
base(ax); ax.set_xlim(65.5, 98)
fig.tight_layout(); fig.savefig(OUT + "fig_india_climate.png", dpi=220); plt.close(fig)

# ---------- Figure 3: Himachal Pradesh climatic zones from the project's 529 cells ----------
cells = pd.read_csv("data/cells.csv")
bins = [-1, 1000, 1500, 2500, 9000]
names = ["I  Sub-tropical low hills (< 1,000 m)", "II  Sub-temperate mid hills (1,000–1,500 m)",
         "III  Wet temperate high hills (1,500–2,500 m)", "IV  Dry temperate / cold desert (> 2,500 m)"]
zc = ["#f4a259", "#bfd36b", "#5aa9a2", "#3b5b92"]
cells["zone"] = pd.cut(cells.elev_m, bins, labels=range(4)).astype(int)
fig, ax = plt.subplots(figsize=(6.4, 6.4))
for z in range(4):
    c = cells[cells.zone == z]
    ax.scatter(c.lon, c.lat, s=46, marker="s", color=zc[z], edgecolor="white", linewidth=0.3,
               label=f"{names[z]}: {len(c)} cells")
gpd.GeoSeries([hp]).boundary.plot(ax=ax, color="#222222", linewidth=0.9)
for n, (x, y) in {"Una": (76.27, 31.47), "Mandi": (76.93, 31.71), "Shimla": (77.17, 31.10),
                  "Kullu": (77.11, 31.96), "Keylong": (77.03, 32.57), "Kaza": (78.07, 32.23),
                  "Dharamshala": (76.32, 32.22)}.items():
    ax.plot(x, y, "k^", ms=4); ax.text(x + 0.05, y + 0.05, n, fontsize=7.5)
ax.set_aspect(1 / 0.85); ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=7.5, frameon=False, markerscale=0.8)
ax.set_xlim(75.4, 79.2); ax.set_ylim(30.2, 33.4); ax.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig(OUT + "fig_hp_zones.png", dpi=220); plt.close(fig)
print(cells.zone.value_counts().sort_index().to_dict())
