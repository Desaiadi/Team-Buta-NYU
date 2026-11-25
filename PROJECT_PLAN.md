# Project Plan: Weather Sensitivity of NYC 311 Complaints

Team BUTA · DS-GA 1007 · Aditya's sub-project (part of "The 100-Meter Inequality")

## The idea
Neighboring areas of NYC get **the same weather**. If their 311 complaints still react to it differently, the difference has to come from the place itself: its buildings, boilers and drains. Weather becomes a natural "stress test" that is applied to every block equally.

**Research questions**
1. Which 311 complaint types follow the weather (cold, rain, snow, heat)?
2. How big is the effect once seasonality is removed? For example, "+X% heat complaints per 10°F colder than a normal day in that month".
3. Does that sensitivity differ between areas, and are neighboring areas alike or different? (the 100-meter question)
4. Do building characteristics (age, size) explain who complains more?

## Data
| Dataset | Role | Source |
|---|---|---|
| NYC 311 Service Requests (2020 → now, ~23M rows) | Main data: what, where and when | NYC Open Data `erm2-nwe9` |
| Daily weather, Central Park | Second dataset: the "stress" | Open-Meteo archive (NOAA station `USW00094728` is the alternative) |
| PLUTO tax lots (~858k) | Building age and units (denominators) | NYC Open Data `64uk-42ks` |

## How we execute it: notebook pipeline
Each notebook does one job, saves its output, and the next notebook starts from that file. Reusable logic lives in `src/`, and the notebooks call it and explain each step.

| # | Notebook | Input → Output |
|---|---|---|
| 01 | `01_explore_311.ipynb`: explore the full 311 data (size, schema, missing values, trends, seasonality, data quality) and choose the scope | `data/raw/311_full/` → decisions and figures |
| 02 | `02_clean_311.ipynb`: trim to the study scope (8 types, 2023–2025, 17 columns) and clean step by step | `311_full/` → `data/interim/311_study_subset.parquet`, `data/processed/complaints_clean.parquet` |
| 03 | `03_explore_weather_pluto.ipynb`: explore the second datasets (weather and buildings) and how they link to 311 | `data/raw/weather_daily.csv`, `pluto.parquet` → findings |
| 04 | `04_clean_weather_pluto.ipynb`: clean them and engineer weather features (lags, heating degree days, rain flags) | → `weather_clean.parquet`, `pluto_clean.parquet` |
| 05 | `05_build_project.ipynb`: join everything into the analysis tables | → `panel_zip_daily.parquet` (ZIP × day × type), `building_heat_season.parquet` |
| 06 | `06_weather_sensitivity_analysis.ipynb`: models, ZIP sensitivities, neighbor permutation test, building-age model and figures | → `reports/figures/` |

The same steps also run as scripts: `python -m src.download`, then `src.preprocess`, then `src.build_features`.

## Methods
- **Effect size:** a quasi-Poisson regression of daily counts on weather (today and yesterday), with weekday, month and year controls. It gives % change per 10°F or per inch of rain, with 95% CIs.
- **Local sensitivity:** the same model fitted separately for every ZIP.
- **Neighbors vs random:** a nearest-neighbor gap test against 2,000 shuffled pairings (permutation test).
- **Buildings:** a Poisson model of complaints per apartment by building era, controlling for borough, size and season.
- **Next step toward "100 m":** repeat the neighbor test on a 200 m grid, or at building level.

## Timeline
| When | What |
|---|---|
| Nov 24 | Pipeline and first analysis done ✅ |
| Nov 25 | Notebooks 01–06 finalized; share with team; agree on slides |
| Before Thu | 15-min feedback Zoom with professor |
| Nov 26–29 | Polish figures, 200 m grid (if time) and Flask demo page (team) |
| Dec 1 | Submission |

## Caveats to state up front
- 311 records **reports**, not conditions.
- "Closed" does not mean "fixed".
- One weather point represents the whole city.
- ZIPs are coarser than 100 m.
