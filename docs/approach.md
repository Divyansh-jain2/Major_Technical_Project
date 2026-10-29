# Extreme Value-Guided Deep Generative Models for Imbalanced Classification
## Our first approach, from raw data to results (Himachal Pradesh temperature extremes)

*Status: Steps 1–3b complete; Step 4 (CTGAN vs EVT-guided CTGAN) running. Last updated 28 Oct 2026.*

---

## 1. What the project is about

**The problem.** In imbalanced classification the class we care about is rare (here, about 1% of days), so classifiers learn it poorly. The usual fixes (class weights, SMOTE) have no theory behind them and are weakest exactly where it matters most: in the extreme tail of the data. Deep generative models (GANs, diffusion models) can create extra minority examples, but nothing makes them respect how real extremes behave.

**The idea.** Treat the minority class as a *tail phenomenon*. Use **Extreme Value Theory (EVT)** to describe that tail mathematically, then use the EVT description to **guide a generative model**, so the synthetic extreme examples are statistically consistent with real extremes. Train a classifier on real + synthetic data and check that it learns the rare class better.

**The case study.** Next-day **extreme temperature days** in **Himachal Pradesh**, 1981–2025, on a 0.1° grid (about 9–11 km). The instructor assigned the region and asked for the temperature side.

**What "predict" means here.** For every grid cell and day D we ask: *will day D+1 be an extreme day for this cell?* The generator never predicts; it only creates extra training examples. The classifier (LightGBM, kept identical across experiments) makes every prediction, so any improvement comes from the data the generator supplies.

---

## 2. The data we have

All raw files are in `himachal_project/data/raw/` on Google Drive and are **never modified**.

| Source | What it is | Period | Grid | How we use it |
|---|---|---|---|---|
| ERA5-Land daily | Reanalysis: Tmax, Tmin, Tmean, dewpoint, pressure, wind, soil moisture, snow | 1981–2025 | 0.1° | **Labels and features** |
| ERA5-Land accumulated | Rainfall, solar radiation | 1981–2025 | 0.1° | Features |
| SRTM | 30 m elevation tiles | static | 30 m | Elevation and slope features |
| Boundary (GADM) | Himachal outline | static | — | Mask to the state |
| IMD Tmax/Tmin | Gridded station temperature | 1981–2025 | 1° | Regional check only (too coarse; 99.9 = missing) |
| IMD rain, CHIRPS, IMERG, MSWEP (no gauge) | Rainfall products | various | 0.05–0.25° | Not used in the basic version |
| MODIS LST, ERA5-Land skin temperature | Surface temperature | 2001–2015 | 0.1° | Not used in the basic version |
| GSOD stations (Shimla, Sundernagar, Bhuntar) | Real station observations | 2016–2025 | points | Planned check of ERA5 extremes |

**Why ERA5-Land for labels.** It is the only temperature source that is complete for 45 years at a resolution fine enough for mountains. It has absolute biases in steep terrain, but our labels compare each cell **with itself** (its own percentiles), so a constant bias moves the threshold, not which days count as extreme.

---

## 3. Pre-processing (the earlier pipeline notebook)

An earlier Colab pipeline (`himachal_pipeline_test.ipynb`) combined the raw files into one daily dataset:

1. **Common grid.** ERA5-Land's 0.1° grid over 30.3–33.3 °N, 75.5–79.1 °E (31 × 37 = 1,147 cells).
2. **Time alignment.** ERA5-Land rain and radiation are stamped one day late; they were shifted back one day.
3. **Other inputs on the grid.** CHIRPS averaged from 0.05° to 0.1°; SRTM averaged into `elev_mean`, `elev_std`, `slope_mean`.
4. **Derived variables.** Relative humidity (from Tmean and dewpoint), wind speed, daily temperature range.
5. **State mask.** Cells whose centre lies outside Himachal set to missing, leaving **529 cells**.
6. Output: `data/processed/himachal_daily_raw.nc` (395 MB, 16,436 days).

That pipeline also bias-corrected temperature against MODIS and validated against IMD. **We do not use those corrected columns**: the correction was never properly validated (its IMD comparison showed a −15 °C bias that is most likely IMD's 99.9 missing-value code being treated as data), and it made the Tmin correlation worse.

---

## 4. Step 1 — Checking the processed file against the raw files

*Script: `steps/step1_check_processed.py` (read-only).*

Before building anything on the processed file we compared it with the raw ERA5-Land files for 5 cells (522–5,279 m) and 6 sample years.

**Results**
- Tmax, Tmin, Tmean, dewpoint, soil moisture, snow and rain match the raw files **exactly** (zero difference) after unit conversion: temperatures in °C (the file's "K" label is wrong), rain in mm/day with the one-day shift correctly applied. Pressure is in hPa and radiation in W m⁻² (Pa ÷ 100, J m⁻² ÷ 86,400).
- All 45 years and 529 cells present; every variable 100% complete **except Tmean and humidity, empty for 1981–1985**. The raw Tmean files for those years are complete, so this was a pipeline hiccup, fixed in Step 2.
- No day with Tmin > Tmax; Tmean always between them.
- 6 glacier cells (5,122–5,552 m) carry ERA5-Land's 10 m snow placeholder; snow is not used as a feature.
- Labelling each cell's top 1% of Tmax gives exactly 1.00% positives.

**Conclusion:** the processed file is a faithful copy of ERA5-Land and safe to build on.

---

## 5. Step 2 — Building the labelled dataset

*Script: `steps/step2_build_dataset.py`. Output: `data/model_ready/`.*

### 5.1 Fixing the gap
Tmean for 1981–1985 refilled from the raw files; humidity recomputed with the Magnus formula (agrees with the stored humidity within 0.03 %RH where both exist). After this, no feature has missing values.

### 5.2 What counts as "extreme"
All thresholds are computed **per grid cell** from **1981–2010 only**, so nothing from the validation or test years leaks in. Heat = Tmax above the 99th percentile; cold = Tmin below the 1st percentile. Three ways of comparing a day with history:

| Label | A day is compared with… | Notes |
|---|---|---|
| `*_season` (**main**) | the same **IMD season** in all base years: winter (Dec–Feb), pre-monsoon (Mar–May), monsoon (Jun–Sep), post-monsoon (Oct–Nov) | requested approach: summer compared with summer, winter with winter |
| `*_annual` | the **whole year** | heat extremes fall in May–August |
| `*_doy` | the same **time of year**, ±7 days | standard climate-index style; kept for comparison |

Mean heat thresholds (Tmax 99th percentile, °C) by elevation:

| Elevation | Winter | Pre-monsoon | Monsoon | Post-monsoon | Whole year | Cells |
|---|---|---|---|---|---|---|
| < 1,000 m | 24.5 | 39.4 | 39.3 | 30.3 | 38.8 | 95 |
| 1–2 km | 20.0 | 32.0 | 31.9 | 24.4 | 31.3 | 91 |
| 2–3 km | 13.8 | 23.7 | 24.2 | 18.9 | 23.5 | 77 |
| 3–4 km | 4.4 | 13.2 | 17.5 | 12.3 | 16.7 | 65 |
| 4–5 km | −3.4 | 5.3 | 13.6 | 6.8 | 12.7 | 134 |
| > 5 km | −8.1 | 1.0 | 11.4 | 2.9 | 10.3 | 67 |

The thresholds fall by about 6–7 °C per km of height, the normal atmospheric lapse rate, a good sign the terrain is represented sensibly.

### 5.3 One row of data
One row = one cell on day D.
- **Features (35 + 8 added later):** Tmax, Tmin, Tmean, dewpoint, humidity, pressure, solar radiation, soil moisture and rain on D, D−1 and D−2; Tmax and Tmin anomalies; latitude, longitude, elevation, slope; season (sine/cosine of day of year).
- **Labels:** the six extreme labels for day **D+1**.
- **Helpers (never used as features):** tomorrow's Tmax/Tmin, the thresholds, and the **excess** (how far past the threshold tomorrow goes), which is what EVT models.

### 5.4 Splits (by year, never random)
Neighbouring days and cells are strongly correlated; one heatwave makes hundreds of cells extreme at once. Random splits would leak information, so we split by year:

| Split | Label years | Rows | Heat (season) | Heat (whole year) | Cold (season) |
|---|---|---|---|---|---|
| Train | 1981–2010 | 5,794,666 | 1.02% | 1.00% | 1.02% |
| Validation | 2011–2015 | 965,954 | 1.00% | 0.77% | 0.88% |
| Test | 2016–2025 | 1,932,437 | **1.94%** | 1.34% | 0.58% |

### 5.5 What the labels already tell us
- **Warming is visible.** Heat extremes doubled in the test decade (1.94% vs 1.02%), cold extremes fell (0.58%). 2024 had the most widespread heat (4.4% of all cell-days, whole-year label).
- **The strongest warming signal is in October–November.** Post-monsoon heat extremes reached 11.0% of days in 2020 and 6–7% in 2017, 2021 and 2024, against a 1% baseline.
- **Extremes come as events.** On 6 Oct 1983, 522 of 529 cells were extreme together. The 59,000 training positives are really a few hundred weather events.
- **Season-edge effect.** Within a season, extremes gather at its warm or cold end: pre-monsoon heat is almost all in **May**, post-monsoon heat in **early October**, monsoon cold in **late September**. For May–June this is genuine heatwave territory; for October heat and September cold the label partly reflects "the season has not turned yet". The day-of-year label is our comparison that has no such effect.

---

## 6. Step 3 — EVT tail model and first baselines

*Scripts: `steps/step3_evt_and_baselines.py`, then `steps/step3b_fix_evt_and_features.py`.*

### 6.1 The EVT model
For a high threshold *u*, the **Pickands–Balkema–de Haan theorem** says the excesses *Y = X − u* (given *X > u*) are approximately **Generalized Pareto** distributed:

  P(Y ≤ y) = 1 − (1 + ξ y / σ)^(−1/ξ)

- **ξ (shape)** decides the tail: ξ < 0 → bounded tail with a maximum excess of σ/|ξ| (expected for temperature); ξ = 0 → exponential; ξ > 0 → heavy tail.
- Fitted by maximum likelihood on 1981–2010 excesses, separately per season, whole year and day of year, for heat and cold.
- **95% intervals by resampling whole days** (block bootstrap), because extremes are spatially clustered.
- **Threshold stability:** refit above the 25th and 50th percentile of the excesses; ξ should barely change.
- **Return levels:** the excess expected once in 10, 50 and 100 years per cell.

### 6.2 First fits and the fix (Step 3 → Step 3b)
Pooling all 529 cells mixed calm low cells with jumpy high cells, which made several tails look heavy (ξ > 0) when the Q-Q plots showed them flattening, i.e. bounded. **Fix:** divide each excess by the cell's own day-to-day variability (std of its temperature anomaly in that season), giving a standardised severity *z*:

  z = (T_{D+1} − u_cell,season) / s_cell,season

| Group | ξ raw | ξ standardised [95% CI] | Fit after fix |
|---|---|---|---|
| Heat, whole year | −0.15 | **−0.15 [−0.20, −0.11]** | excellent |
| Heat, monsoon | −0.13 | −0.12 [−0.19, −0.04] | excellent |
| Heat, pre-monsoon | −0.17 | −0.14 [−0.24, −0.07] | good |
| Heat, winter | −0.07 | −0.10 [−0.18, −0.03] | good up to z ≈ 1.2 |
| Heat, post-monsoon | **+0.26** | **+0.07** [−0.18, 0.23] | much better |
| Cold, monsoon | **+0.16** | **−0.07** [−0.14, 0.02] | fixed |
| Cold, winter / whole year / day of year | +0.13 / +0.01 / +0.24 | +0.19 / +0.08 / +0.22 | far tail still not captured |

**Takeaways**
- After standardising, all heat tails are bounded or nearly so, as physics expects. The heat side is ready to guide a generator.
- Cold extremes mix different regimes (clear-night cooling, snow-covered high cells) and need further work.
- **Recent winter warm extremes are more intense, not just more frequent:** 21% of 2016–2025 winter heat extremes lie beyond the 1981–2010 tail's 90th percentile (10% expected). Summer heat intensity is unchanged (8–10%).

### 6.3 Baselines (target: next-day seasonal heat, `y_heat_season`)
All trained on 1981–2010; decision threshold chosen on 2011–2015 to maximise F1; scored on 2016–2025 (1.94% positives).

- **Persistence (no model):** score = how far today's Tmax is above tomorrow's threshold.
- **LightGBM plain / class-weighted / SMOTE** (SMOTE grows extremes to 10% of normal days, +514,294 synthetic rows).

| Model | Test PR-AUC, Step 3 | Test PR-AUC, Step 3b | Recall (3b) | Precision (3b) | CSI (3b) |
|---|---|---|---|---|---|
| Persistence | 0.586 | 0.586 | 0.60 | 0.55 | 0.41 |
| LightGBM plain | 0.434 | 0.531 | 0.41 | 0.59 | 0.32 |
| LightGBM, no calendar | — | 0.549 | 0.47 | 0.58 | 0.35 |
| LightGBM weighted | 0.446 | 0.525 | 0.49 | 0.60 | 0.37 |
| **LightGBM SMOTE** | 0.449 | **0.562** | 0.52 | 0.57 | 0.37 |

**What happened**
- In Step 3 the model leaned on the calendar (day of year = 39% of its decisions), which broke when the climate shifted in the test years.
- Step 3b added **distance-to-tomorrow's-threshold** features (today, yesterday, the day before, in °C and in units of the cell's variability). Every model gained 0.08–0.11 PR-AUC; the standardised distance now carries 78% of the decisions, and dropping calendar features helps.
- **SMOTE is the best classic rebalancing method**, the bar the GAN methods must clear.
- **Persistence is still slightly ahead overall**, because heat extremes persist (most extreme days follow another extreme day) and the models stop after 5–29 trees under the 1981–2010 → 2016–2025 shift. In the **monsoon season SMOTE already beats persistence** (0.481 vs 0.442).

---

## 7. Step 4 — Vanilla vs EVT-guided CTGAN (running)

*Script: `steps/step4_ctgan_evt.py`. Output: `results/step4/`.*

**Generator.** One **CTGAN** (conditional tabular GAN) trained on the real extreme days of a random half of 1981–2010 (~30,000 extremes): the 41 classifier features plus season and the standardised severity *z*. CTGAN handles mixed, multi-modal tabular columns with mode-specific normalisation and conditional sampling.

**Three synthetic sets from the same trained GAN** (so differences come from EVT, not the GAN), each adding extremes up to 10% of normal days, like SMOTE:
1. **Vanilla:** samples used as generated.
2. **Physical:** impossible samples removed (Tmin above Tmax, humidity outside 0–100%, negative rain, dewpoint above temperature, inconsistent signs, terrain outside the real range).
3. **EVT-guided (proposed):** physical filter, then **per season, resampled so the severity z follows the fitted GPD tail**: 20 bins of equal GPD probability, each sample weighted by (target share ÷ generated share), and nothing beyond the tail's upper bound σ/|ξ|.

*z* is used only to steer generation; it is **never a classifier feature** (it encodes tomorrow's answer).

**Fixed classifier for every experiment:** LightGBM, learning rate 0.05, up to 1,000 trees with early stopping, no calendar features, and a **monotone constraint** (a larger distance to the threshold can only raise the predicted risk), to hold up better under climate shift.

**Evaluation (test 2016–2025)**
- PR-AUC (main), precision, recall, F1, CSI, PR-AUC per season.
- **Onset scores:** only days where today is not yet extreme; persistence cannot see these coming.
- **Top-10% recall:** share of the 10% most severe test extremes that are caught; where EVT guidance should matter most.
- **Tail consistency of each synthetic set:** GPD refit on its *z*, quantiles, and KS distance to the fitted tail.

**Results (test 2016–2025, ~2.9 M training rows, 257,248 synthetic extremes per generator)**

| Training data | PR-AUC | Recall | Onset PR-AUC | Onset recall | Top-10% recall |
|---|---|---|---|---|---|
| Persistence (no model) | 0.586 | 0.60 | 0.191 | 0.03 | 0.84 |
| Real only | 0.576 | 0.51 | 0.228 | 0.09 | 0.81 |
| SMOTE | 0.581 | 0.52 | 0.221 | 0.03 | 0.80 |
| CTGAN, vanilla | 0.584 | 0.54 | 0.199 | 0.08 | 0.82 |
| CTGAN, physical filter | 0.582 | 0.51 | 0.193 | 0.07 | 0.81 |
| CTGAN, EVT-guided | 0.571 | 0.50 | 0.202 | 0.11 | 0.79 |

- **Only 15.9% of vanilla CTGAN rows were physically possible.**
- EVT resampling made the synthetic severity follow the fitted GPD (KS distance 0.16–0.24 → 0.017–0.023, as close as the real data), but the GAN never generated the far tail, so EVT could only re-weight what was there.
- Classification: all methods within ~0.015 PR-AUC of each other.

## 7b. Step 5 — Tail-conditioned CTGAN and confidence intervals

*Script: `steps/step5_tail_conditioned.py`.*

CTGAN was conditioned on season × GPD severity bin, physics enforced by construction (rows snapped to real cells, distances computed exactly), and all methods were given **95% intervals by resampling whole test months**.

- Intervals are about **±0.07 PR-AUC** and overlap almost completely: **no method (SMOTE, CTGAN, EVT, weights) is statistically better than real data alone, or than persistence.**
- **Conditioning failed:** 70% of generated rows had a severity outside the requested bin. CTGAN only guarantees the conditioned label, not the continuous columns.
- One small, significant effect: tilting towards the severe tail beat the same conditional GAN without EVT (+0.0045 PR-AUC, better in 98% of resamples).

## 7c. What the literature says, and the revised plan (Steps 6–8)

- *To SMOTE, or not to SMOTE?* (Elor & Averbuch-Elor, 2022): balancing does not help **strong** classifiers such as gradient boosting, but does help weaker ones. This explains Steps 4–5.
- Recent tabular work (AAAI 2025) finds **diffusion models generate better minority samples than GANs**; tail-aware losses in tabular diffusion help on fraud data (2026).
- EVT + generative models (ExGAN, evtGAN, heavy-tailed diffusion) exist, but almost none test whether the generated extremes improve a **classifier**. That is this project's gap.

Revised plan:
1. **Step 6:** dataset v3 with longer-history features (7/14/30-day anomalies, warm-spell length, days since last extreme, rain and soil moisture over weeks, pressure trend).
2. **Step 7:** our own conditional **tabular diffusion** model where severity is a direct input, so EVT chooses exactly which severities are generated, including beyond the largest observed event up to the GPD bound. Variants: observed severities (no EVT), GPD severities (EVT), GPD with tail emphasis; plus a CTGAN baseline. A "control check" verifies generated rows match the requested severity.
3. **Step 8:** LightGBM **and** a neural-network classifier, all rebalancing methods, scored on all days, onset days and the **10% most severe extremes**, with month-block intervals and paired tests (EVT vs no-EVT, same generator).

---

## 8. Limitations so far
- **Labels come from a reanalysis**, not stations. Station checks (GSOD: Shimla, Sundernagar, Bhuntar, 2016–2025) are planned but not yet run.
- **Climate shift**: the test decade has twice the training rate of heat extremes, which hurts any model trained on 1981–2010.
- **Season-edge effect** on post-monsoon heat and monsoon cold labels.
- **Cold tails** not yet well described by a single pooled GPD.
- **Compute**: free Colab only; training uses half of 1981–2010 for the GAN experiment.

## 9. Next steps
1. Read Step 4: does the EVT-guided set beat vanilla CTGAN and SMOTE, especially on onset and top-10% recall?
2. Put EVT inside training (a GPD-likelihood penalty in the generator loss) instead of only at sampling time.
3. A diffusion model (TabDDPM) with the same EVT guidance.
4. Cold extremes; station checks; one or two public imbalanced benchmarks for the "general-purpose" claim.

## 10. Reproducing it
Code: GitHub `Divyansh-jain2/Major_Technical_Project`, folder `steps/`. Each step is one Colab cell, reads only earlier outputs, writes only to its own folder, and saves a `stepN_summary.txt` on Drive.

| Step | Script | Output |
|---|---|---|
| 1 | `step1_check_processed.py` | `step1_check.txt` |
| 2 | `step2_build_dataset.py` | `data/model_ready/` |
| 3 | `step3_evt_and_baselines.py` | `results/step3/` |
| 3b | `step3b_fix_evt_and_features.py` | `results/step3b/` |
| 4 | `step4_ctgan_evt.py` | `results/step4/` |
