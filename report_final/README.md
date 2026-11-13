# Final report (IIT Mandi MTP format)

Compile `main.tex` with pdfLaTeX (twice). Put every image below in a folder called `figures/` next to `main.tex`.
Fill in the lines marked `TODO` at the top of `main.tex` (roll number, supervisor, school, course code).

| File | Where it is used |
|---|---|
| iitmandi_logo.png | Title page |
| fig_pipeline.png | Fig. 1, project pipeline |
| fig_study_area.png | Fig. 2, study area map |
| fig_thresholds.png | Fig. 3, thresholds by elevation |
| fig_gpd_shape.png | Fig. 4, GPD shape |
| evt_qq_std.png | Fig. 5, GPD Q-Q plots |
| fig_tail_focus.png | Fig. 6, EVT tail focus |
| pr_curves.png | Fig. 7, first comparison PR curves |
| ratio_sweep.png | Fig. 8, amount of synthetic data |
| seed_spread.png | Fig. 9, five seeds (diffusion) |
| fig_generator_quality.png | Fig. 10, generator properties |
| fig_step12_by_seed.png | Fig. 11, all methods x 5 seeds |
| fig_evt_gain.png | Fig. 12, gain of EVT tail focus |
| fig_realism.png | Fig. 13, realism vs usefulness |
| fig_step13.png | Fig. 14, control experiment |

Figures are regenerated with `python3 report/make_figures.py` and `python3 report_final/make_extra_figures.py`;
`pipeline.tex` is the source of `fig_pipeline.png`.
