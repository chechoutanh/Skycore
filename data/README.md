# Observing logs

Real measurements from club observing events, used by `python -m skyscore.evaluation data/events.csv data/observations.csv`.

## `events.csv`: one row per event

| Column | Example | Notes |
|---|---|---|
| `event_id` | E01 | Unique, links to `observations.csv` |
| `date` | 2025-03-15 | Local date of the **evening** the event started |
| `club` | Club A | |
| `site` | Ba Vi | |
| `lat`, `lon` | 21.07, 105.37 | Decimal degrees (east/north positive) |
| `utc_offset` | 7 | Hours, Vietnam = 7 |
| `target` | M42 (Orion Nebula) | Must be one of the targets in the app (used for altitude/Moon) |
| `other_targets` | Jupiter; M45 | Free text |
| `equipment` | 8" Dobsonian, f=1200 mm, ZWO ASI585MC | Focal length + camera if seeing was measured from images |
| `participants` | 35 | |
| `recommended_window` | 21:00-23:00 | Local time SkyScore suggested |
| `prediction_saved` | yes / no | Was the recommendation saved **before** the event? |

## `observations.csv`: one row per observed hour

| Column | Example | Notes |
|---|---|---|
| `event_id` | E01 | |
| `time` | 21:00 | Local time; hours after midnight (e.g. 01:00) are counted in the same night |
| `seeing_value` | 2.4 | Number on the scale given in `seeing_method` |
| `seeing_method` | `fwhm_arcsec` | `fwhm_arcsec` (from images, lower = better), `pickering_1_10` (10 = best), `scale_1_5` (5 = best) |
| `cloud_observed` | 30 | % of sky covered (okta × 12.5 if you logged okta) |
| `quality_rating` | 4 | Overall observing quality, 1 (bad) to 5 (excellent) |
| `transparency` | 5.5 | Naked-eye limiting magnitude or SQM reading (optional) |
| `wind_feel` | calm / light / strong | Did wind shake the telescope? (optional) |
| `notes` | dew on finder | Optional |

FWHM to arcsec: `seeing = FWHM_px × 206.265 × pixel_size_um / focal_length_mm`.

Log bad hours too (clouds, poor seeing): the evaluation needs them. Weather does **not** need to be logged; it is downloaded from the Open-Meteo forecast archive for each site and hour.
