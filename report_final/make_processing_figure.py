"""Yearly share of extreme heat cell-days (season-wise labels), from the Step 2 summary
(step2_summary.txt, section 4). Seasonal shares are weighted by the number of days in each season.
Writes figures/fig_yearly_extremes.png."""
import matplotlib.pyplot as plt
rows = """1981 0.00 0.93 0.53 0.00
1982 0.00 0.00 0.07 0.00
1983 0.19 0.00 0.82 5.86
1984 0.09 6.48 2.18 2.22
1985 1.37 0.03 0.10 2.67
1986 0.00 0.00 1.23 0.01
1987 1.02 0.00 2.00 2.85
1988 5.53 3.53 0.96 0.19
1989 0.27 0.09 0.22 0.76
1990 0.49 0.98 1.09 0.01
1991 1.54 0.00 0.03 0.00
1992 0.04 0.00 0.04 0.91
1993 3.23 0.75 1.10 1.61
1994 1.65 0.40 1.00 0.05
1995 0.00 0.30 5.65 0.82
1996 0.00 0.00 0.11 0.07
1997 0.01 0.00 2.18 0.00
1998 2.95 3.62 0.62 0.33
1999 1.56 0.23 0.22 0.00
2000 0.02 4.31 0.07 4.59
2001 0.17 1.41 1.18 1.75
2002 0.73 0.39 0.39 0.11
2003 0.39 0.00 1.41 0.00
2004 2.17 2.64 0.72 0.40
2005 0.09 0.00 1.46 0.03
2006 3.84 1.37 0.76 2.14
2007 0.44 1.12 1.58 0.19
2008 2.36 0.29 0.34 2.00
2009 0.14 0.02 1.89 1.22
2010 0.67 1.53 0.37 0.37
2011 2.38 0.07 0.03 2.53
2012 0.00 0.81 1.26 0.41
2013 0.24 1.47 1.13 6.28
2014 0.68 0.01 0.93 2.21
2015 1.17 0.00 0.18 1.21
2016 9.20 0.66 0.07 4.20
2017 1.19 0.04 0.71 6.35
2018 1.07 0.13 0.44 0.05
2019 0.00 0.20 0.54 0.00
2020 0.27 0.05 1.33 10.95
2021 1.58 0.00 1.41 6.15
2022 0.01 3.37 1.57 0.26
2023 2.97 0.00 1.06 3.38
2024 0.35 2.13 5.86 6.62
2025 4.40 2.91 2.97 0.67"""
W = [90.25, 92, 122, 61]
yr, val = [], []
for line in rows.splitlines():
    p = line.split(); y = int(p[0]); s = list(map(float, p[1:]))
    yr.append(y); val.append(sum(a * w for a, w in zip(s, W)) / sum(W))
col = ["#9db4cf" if y <= 2010 else "#e8b04b" if y <= 2015 else "#d9622b" for y in yr]
fig, ax = plt.subplots(figsize=(8.6, 3.6))
ax.bar(yr, val, color=col, width=0.8)
ax.axhline(1.0, color="#333333", lw=1, ls="--")
ax.text(1981, 1.12, "1% by construction (1981–2010 average)", fontsize=8)
for x, t, c in [(1995.5, "Train years 1981–2010", "#5a7fa8"), (2013, "Val.", "#b07f1a"), (2020.5, "Test 2016–2025", "#b8401a")]:
    ax.text(x, 3.75, t, ha="center", fontsize=8.5, color=c, fontweight="bold")
ax.set_ylabel("Extreme heat cell-days (%)"); ax.set_xlim(1980.3, 2025.7); ax.set_ylim(0, 4.1)
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig("figures/fig_yearly_extremes.png", dpi=220)
for y, v in zip(yr, val): print(y, round(v, 2))
