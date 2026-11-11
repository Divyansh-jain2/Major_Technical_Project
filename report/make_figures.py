"""Report figures built from the numbers in the step summaries (Steps 2, 3b, 11, 12).

Run: python3 report/make_figures.py  ->  report/figures/fig_*.png
Per-seed Step 12 scores are read from report/data/step12_per_seed.csv (parsed from the Step 12 log).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# validated categorical slots (light surface) + text tokens
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titlesize": 11, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- thresholds by elevation (Step 2)
bands = ["<1", "1-2", "2-3", "3-4", "4-5", ">5"]
thr = {"Winter": [24.5, 20.0, 13.8, 4.4, -3.4, -8.1], "Pre-monsoon": [39.4, 32.0, 23.7, 13.2, 5.3, 1.0],
       "Monsoon": [39.3, 31.9, 24.2, 17.5, 13.6, 11.4], "Post-monsoon": [30.3, 24.4, 18.9, 12.3, 6.8, 2.9]}
fig, ax = plt.subplots(figsize=(7.2, 3.8))
x = np.arange(len(bands))
for (s, v), c in zip(thr.items(), (BLUE, ORANGE, AQUA, YELLOW)):
    ax.plot(x, v, color=c, lw=2, marker="o", ms=6, mec=SURF, mew=1.5)
    ax.annotate(s, (x[-1], v[-1]), xytext=(8, {"Post-monsoon": 4, "Pre-monsoon": -4}.get(s, 0)), textcoords="offset points", va="center", color=INK2, fontsize=9)
ax.set_xticks(x, [f"{b} km" for b in bands])
ax.set_xlim(-0.3, len(bands) - 0.2)
ax.set_xlabel("Cell elevation")
ax.set_ylabel("Heat threshold (99th pct of Tmax, °C)")
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_title("Extreme-heat thresholds fall about 6–7 °C per km of altitude", loc="left")
save(fig, "fig_thresholds.png")

# ---------------------------------------------------------------- GPD shape raw vs standardised (Step 3b)
grp = ["Whole year", "Monsoon", "Pre-monsoon", "Winter", "Post-monsoon"]
raw = [-0.153, -0.128, -0.165, -0.074, 0.262]
std = [-0.152, -0.115, -0.141, -0.096, 0.070]
lo = [-0.196, -0.185, -0.241, -0.176, -0.177]
hi = [-0.113, -0.040, -0.072, -0.032, 0.231]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
y = np.arange(len(grp))[::-1]
ax.axvline(0, color=INK2, lw=1)
ax.hlines(y, lo, hi, color=BLUE, lw=2)
ax.plot(std, y, "o", color=BLUE, ms=8, mec=SURF, mew=1.5, label="standardised excess (95% interval)")
ax.plot(raw, y + 0.18, "D", color=MUTED, ms=6, mec=SURF, mew=1, label="raw excess")
ax.set_yticks(y, grp)
ax.set_xlabel("GPD shape ξ   (< 0: bounded tail, > 0: heavy tail)")
ax.text(-0.27, -0.85, "bounded tail", color=INK2, fontsize=9)
ax.text(0.17, -0.85, "heavy tail", color=INK2, fontsize=9)
ax.set_ylim(-1.1, len(grp) - 0.5)
ax.legend(loc="upper right", fontsize=8.5)
ax.set_title("Standardising each cell removes the spurious heavy tail", loc="left")
save(fig, "fig_gpd_shape.png")

# ---------------------------------------------------------------- generator properties (Step 11)
gens = ["Gaussian\ncopula", "TVAE", "CTGAN", "CopulaGAN", "Cond.\nWGAN-GP"]
broke = [12.28, 74.69, 68.68, 71.48, 9.02]
top5 = [18.1, 16.7, 17.7, 12.3, 17.1]
mins = [1.8, 18.5, 23.5, 25.4, 3.4]
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, v, t, fmt in zip(axs, (broke, top5, mins),
                         ("Raw rows breaking physics (%)", "Rows judged in top 5% of tail (%; real = 5)",
                          "Training time on T4 (min; diffusion ≈ 30)"),
                         ("{:.0f}%", "{:.0f}%", "{:.0f}")):
    b = ax.bar(gens, v, color=BLUE, width=0.62)
    for r, val in zip(b, v):
        ax.annotate(fmt.format(val), (r.get_x() + r.get_width() / 2, val), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    ax.set_title(t, loc="left", fontsize=10)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(max(v), 30 if v is mins else 0) * 1.18)
axs[1].axhline(5, color=INK2, ls="--", lw=1)
axs[2].axhline(30, color=INK2, ls="--", lw=1)
fig.suptitle("Step 11: the five new generators before any classifier", x=0.01, ha="left", fontweight="bold", color=INK)
fig.tight_layout()
save(fig, "fig_generator_quality.png")

# ---------------------------------------------------------------- Step 12 per seed
d = pd.read_csv(os.path.join(HERE, "data", "step12_per_seed.csv"))
mean = d.groupby("method")[["pr_auc", "onset", "severe"]].mean()
PERS = {"pr_auc": 0.5864, "onset": 0.1907, "severe": 0.5651}
G = [("gaussian_copula", "Gaussian copula"), ("tvae", "TVAE"), ("ctgan", "CTGAN"),
     ("copulagan", "CopulaGAN"), ("cwgan", "Cond. WGAN-GP"), ("diffusion", "Diffusion")]
BASEL = [("none", "Real only"), ("smote", "SMOTE"), ("borderline_smote", "Borderline\nSMOTE"), ("adasyn", "ADASYN")]
fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
for ax, (s, lab) in zip(axs, [("severe", "Severe PR-AUC (10% most severe extremes)"),
                              ("pr_auc", "PR-AUC (all days)"), ("onset", "Onset PR-AUC (heatwave starts)")]):
    pos, ticks = 0, []
    for m, nm in BASEL:
        v = d[d.method == m][s].values
        ax.scatter(np.full(len(v), pos), v, s=18, color=MUTED, alpha=0.8, zorder=3)
        ax.hlines(v.mean(), pos - 0.32, pos + 0.32, color=INK2, lw=2.5, zorder=4)
        ticks.append((pos, nm))
        pos += 1
    pos += 0.5
    for g, nm in G:
        for k, (var, c) in enumerate((("plain", BLUE), ("tail", ORANGE))):
            v = d[d.method == f"{g}_{var}"][s].values
            xx = pos + (k - 0.5) * 0.42
            ax.scatter(np.full(len(v), xx), v, s=18, color=c, alpha=0.85, zorder=3)
            ax.hlines(v.mean(), xx - 0.18, xx + 0.18, color=c, lw=2.5, zorder=4)
        ticks.append((pos, nm))
        pos += 1.2
    ax.axhline(mean.loc["none", s], color=INK, ls="--", lw=1)
    ax.axhline(PERS[s], color=INK2, ls=":", lw=1.3)
    ax.set_title(lab, loc="left", fontsize=10)
    ax.grid(axis="x", visible=False)
axs[0].annotate("persistence", (pos - 0.9, PERS["severe"]), xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK2, ha="right")
axs[0].annotate("real only (mean)", (pos - 0.9, mean.loc["none", "severe"]), xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK2, ha="right")
axs[-1].set_xticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=9)
from matplotlib.lines import Line2D
axs[0].legend(handles=[Line2D([], [], marker="o", ls="", color=MUTED, label="no generator / interpolation"),
                       Line2D([], [], marker="o", ls="", color=BLUE, label="generator, as it is (plain)"),
                       Line2D([], [], marker="o", ls="", color=ORANGE, label="generator + EVT tail focus")],
              loc="lower left", bbox_to_anchor=(0, 1.08), fontsize=8.5, ncol=3)
fig.suptitle("Step 12: LightGBM test scores, 5 training seeds per method (dots), mean (bar)",
             x=0.01, y=1.03, ha="left", fontweight="bold", color=INK)
fig.tight_layout()
save(fig, "fig_step12_by_seed.png")

# ---------------------------------------------------------------- EVT effect per generator
fig, axs = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
names = [nm for _, nm in G]
for ax, (s, lab) in zip(axs, [("pr_auc", "PR-AUC"), ("onset", "Onset PR-AUC"), ("severe", "Severe PR-AUC")]):
    diffs, wins = [], []
    for g, _ in G:
        a = d[d.method == f"{g}_tail"].sort_values("seed")[s].values
        b = d[d.method == f"{g}_plain"].sort_values("seed")[s].values
        diffs.append((a - b).mean())
        wins.append(int(((a - b) > 0).sum()))
    yy = np.arange(len(G))[::-1]
    ax.barh(yy, diffs, color=ORANGE, height=0.6)
    for yv, dv, w in zip(yy, diffs, wins):
        ax.annotate(f"+{dv:.3f}  ({w}/5)" if dv >= 0 else f"{dv:.3f}  ({w}/5)", (dv, yv), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(yy, names)
    ax.set_title(f"{lab}: EVT tail − plain", loc="left", fontsize=10)
    ax.set_xlim(min(0, min(diffs)) - 0.003, max(diffs) * 1.75)
    ax.grid(axis="y", visible=False)
fig.suptitle("EVT tail focus improves every generator (mean same-seed gain; seeds better out of 5)",
             x=0.01, ha="left", fontweight="bold", color=INK)
fig.tight_layout()
save(fig, "fig_evt_gain.png")

# ---------------------------------------------------------------- realism vs usefulness (Step 12 ensembles)
q = pd.DataFrame([
    ("SMOTE", "interp", 0.81, 0.5071), ("Borderline-SMOTE", "interp", 0.88, 0.4533), ("ADASYN", "interp", 0.96, 0.4498),
    ("Gaussian copula", "plain", 5.70, 0.5553), ("Gaussian copula", "tail", 5.39, 0.5582),
    ("TVAE", "plain", 5.47, 0.5271), ("TVAE", "tail", 5.49, 0.6012),
    ("CTGAN", "plain", 6.65, 0.5950), ("CTGAN", "tail", 6.60, 0.6014),
    ("CopulaGAN", "plain", 7.22, 0.5439), ("CopulaGAN", "tail", 7.07, 0.5909),
    ("Cond. WGAN-GP", "plain", 3.76, 0.5375), ("Cond. WGAN-GP", "tail", 3.89, 0.5794),
    ("Diffusion", "plain", 2.04, 0.5305), ("Diffusion", "tail", 2.09, 0.5792)],
    columns=["gen", "kind", "close", "severe"])
fig, ax = plt.subplots(figsize=(7.6, 4.6))
col = {"interp": MUTED, "plain": BLUE, "tail": ORANGE}
for _, r in q.iterrows():
    ax.scatter(r.close, r.severe, s=60, color=col[r.kind], edgecolor=SURF, linewidth=1.5, zorder=3)
for g in q.gen.unique():
    sub = q[q.gen == g]
    if len(sub) == 2:
        ax.plot(sub.close, sub.severe, color=GRID, lw=1.5, zorder=2)
    r = sub.iloc[-1]
    off = {"CTGAN": (6, 4), "ADASYN": (6, -10), "Borderline-SMOTE": (6, 3)}.get(g, (6, -3))
    ax.annotate(g, (r.close, r.severe), xytext=off, textcoords="offset points", fontsize=8, color=INK2)
ax.axhline(0.4407, color=INK, ls="--", lw=1)
ax.annotate("real only", (4.2, 0.4407), xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK2, ha="right")
ax.axhline(0.5651, color=INK2, ls=":", lw=1.3)
ax.annotate("persistence", (7.4, 0.5651), xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK2, ha="right")
ax.axvline(1, color=GRID, lw=1)
ax.set_xlabel("Distance to nearest real extreme, relative to real data (1 = as close as real days are)")
ax.set_ylabel("Severe PR-AUC, 5-seed model (test)")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=MUTED, label="interpolation (SMOTE family)"),
                   Line2D([], [], marker="o", ls="", color=BLUE, label="generator, plain"),
                   Line2D([], [], marker="o", ls="", color=ORANGE, label="generator + EVT tail")],
          loc="lower left", bbox_to_anchor=(0.25, 0.1), fontsize=8)
ax.set_title("More realistic synthetic days did not help more", loc="left")
save(fig, "fig_realism.png")
print("figures written to", FIG)

# ---------------------------------------------------------------- Step 13: EVT emphasis with and without a generator
d13 = pd.read_csv(os.path.join(HERE, "data", "step13_per_seed.csv"))
both = pd.concat([d[d.method.isin(["none", "ctgan_tail", "tvae_tail", "gaussian_copula_tail", "diffusion_tail"])], d13])
M13 = [("none", "Real\nonly", MUTED), ("real_dup_plain", "Random\ncopies", AQUA), ("real_dup_tail", "EVT-tail\ncopies", AQUA),
       ("real_weight_tail", "EVT-tail\nweights", AQUA), ("gaussian_copula_tail", "Gaussian\ncopula", ORANGE),
       ("diffusion_tail", "Diffusion", ORANGE), ("ctgan_tail", "CTGAN", ORANGE), ("tvae_tail", "TVAE", ORANGE)]
fig, axs = plt.subplots(3, 1, figsize=(9.5, 9), sharex=True)
for ax, (s, lab) in zip(axs, [("severe", "Severe PR-AUC (10% most severe extremes)"), ("pr_auc", "PR-AUC (all days)"),
                              ("onset", "Onset PR-AUC (heatwave starts)")]):
    for i, (m, nm, c) in enumerate(M13):
        v = both[both.method == m][s].values
        ax.scatter(np.full(len(v), i), v, s=18, color=c, alpha=0.85, zorder=3)
        ax.hlines(v.mean(), i - 0.3, i + 0.3, color=c if c != MUTED else INK2, lw=2.5, zorder=4)
    ax.axhline(PERS[s], color=INK2, ls=":", lw=1.3)
    ax.axvline(3.5, color=GRID, lw=1.2)
    ax.set_title(lab, loc="left", fontsize=10)
    ax.grid(axis="x", visible=False)
axs[-1].set_xticks(range(len(M13)), [m[1] for m in M13], fontsize=9)
axs[0].annotate("persistence", (3.4, PERS["severe"]), xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK2, ha="right")
axs[0].text(1.5, 0.385, "no generator", ha="center", fontsize=8.5, color=INK2)
axs[0].text(5.5, 0.385, "generator + EVT tail focus", ha="center", fontsize=8.5, color=INK2)
axs[0].legend(handles=[Line2D([], [], marker="o", ls="", color=MUTED, label="real data only"),
                       Line2D([], [], marker="o", ls="", color=AQUA, label="extra real extremes (copies or weights)"),
                       Line2D([], [], marker="o", ls="", color=ORANGE, label="extra synthetic extremes")],
              loc="lower left", bbox_to_anchor=(0, 1.1), fontsize=8.5, ncol=3)
fig.suptitle("Step 13: EVT tail emphasis with and without a generator (5 seeds; bar = mean)",
             x=0.01, y=1.0, ha="left", fontweight="bold", color=INK)
fig.tight_layout()
save(fig, "fig_step13.png")
print("step 13 figure written")
