# Major Technical Project

Temperature prediction for Himachal Pradesh using ERA5 / ERA5-Land as model
inputs and independent weather-station observations as ground truth.

## Why station data

The IMD gridded temperature product (e.g. from Google Earth Engine) is
interpolated from a sparse station network at 1° resolution, so in the hills it
is a reference product, not true ground truth. This project uses raw station
observations from NOAA's Global Summary of the Day (GSOD), which come from WMO
synoptic reports and are not interpolated.

## Station check

`himachal_gsod_check.py` finds GSOD stations in and around Himachal Pradesh,
scores how complete their daily Tmax/Tmin records are, and saves clean daily
data in °C.

It runs automatically on GitHub Actions (see the **Actions** tab →
*GSOD Himachal station check* → *Run workflow*). Results are committed to
`data/gsod_himachal/` and also attached to each run as a downloadable artifact.

To run it locally instead:

```bash
pip install -r requirements.txt
python himachal_gsod_check.py --start 2000 --end 2025 --save-daily
```

### Outputs

| File | Contents |
|---|---|
| `stations_summary.csv` | One row per station: location, elevation, % completeness, usable years |
| `yearly_completeness.csv` | Valid Tmax/Tmin days per station per year |
| `daily/<id>_<name>.csv` | Daily Tmean/Tmax/Tmin in °C with "derived" flags |

"Usable year" = a year in which at least 80% of days have both Tmax and Tmin.

### GSOD caveats

- Source values are in °F; `9999.9` means missing (the script converts and masks).
- A `*` flag on MAX/MIN means the value was derived from hourly reports.
- Data is not IMD quality-controlled; high-altitude coverage is sparse.

## Planned validation approach

1. ERA5 / ERA5-Land as predictors.
2. GSOD (and ISRO MOSDAC AWS, where available) station points as validation,
   with a lapse-rate correction for the grid-cell vs station elevation gap.
3. IMD gridded data as a secondary benchmark only.
4. Leave-one-station-out cross-validation.
