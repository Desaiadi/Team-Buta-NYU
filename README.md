# The 100-Meter Inequality: Weather Sensitivity of NYC 311 Complaints

Team BUTA · DS-GA 1007 Programming for Data Science

**Question:** neighboring areas get the same weather. Do their 311 complaints react to it differently?
For example, which buildings and ZIPs spike in HEAT/HOT WATER complaints during a cold snap, or in Sewer complaints after heavy rain?

## Project layout
```
Project/
├── src/
│   ├── config.py          # every path, date range, complaint type and URL
│   ├── download.py        # step 1: download 311, PLUTO and weather (cached, resumable)
│   ├── preprocess.py      # step 2: clean the data and write a cleaning report
│   └── build_features.py  # step 3: ZIP×day panel and building×season heat table
├── data/                  # created by the scripts (not committed)
│   ├── raw/               #   API downloads, untouched
│   └── processed/         #   clean parquet files
├── reports/cleaning_report.json
├── notebooks/             # EDA and models go here
├── Relevent Materials/    # proposal, brief, meeting notes, findings
└── requirements.txt
```

## 1. Set up the environment in VS Code (one time)

1. **Open the folder:** VS Code → *File → Open Folder…* → select `Project`.
2. **Install the extensions:** *Python* and *Jupyter*, both by Microsoft.
3. **Open the terminal:** *View → Terminal* (or press `` Ctrl+` ``). It opens in the project folder.
4. **Create the virtual environment.** Use Python 3.13, since every package has wheels for it:
   ```bash
   python3.13 -m venv .venv
   ```
5. **Activate it.** The prompt then starts with `(.venv)`:
   ```bash
   source .venv/bin/activate
   ```
6. **Install the packages:**
   ```bash
   pip install --upgrade pip && pip install -r requirements.txt
   ```
7. **Point VS Code at the environment:** `Cmd+Shift+P` → *Python: Select Interpreter* → choose `./.venv/bin/python`.
   - New terminals now activate `.venv` automatically.
   - In notebooks, choose the same `.venv` kernel from the top-right *Select Kernel* button.

Optional: get a free NYC Open Data app token (data.cityofnewyork.us → sign up → *Developer Settings*) for faster, unthrottled downloads:
```bash
export NYC_APP_TOKEN="your-token"
```

## 2. Run the pipeline
Run these from the project root with `.venv` active. They must run as modules (`-m`) so the `src` imports resolve.

```bash
python -m src.download          # ~20-40 min the first time; re-runs skip cached files
python -m src.preprocess        # ~1-2 min
python -m src.build_features    # ~1-2 min
```

- **Analysis:** open `notebooks/01_weather_sensitivity.ipynb` in VS Code, select the `.venv` kernel and click *Run All*. It takes about 30 seconds and saves 8 figures to `reports/figures/`. The model code lives in `src/analysis.py` and the plotting code in `src/plots.py`.
- If the download is interrupted, run it again. Finished months are cached in `data/raw/311/<type>/<YYYY-MM>.parquet`.
- To change the years or complaint types, edit `src/config.py`, then re-run all three steps.

## 3. What each step does
**Download**
- **311:** 8 complaint types, Jan 2023 – Dec 2025, 17 of the 44 columns. Pulled month by month from the Socrata API, using 4 parallel threads and retries.
- **PLUTO:** about 858k tax lots, with year built, residential units, floors and building class.
- **Weather:** daily Central Park data from Open-Meteo (°F, inches): temperature, rain, snow and wind.

**Preprocess** (every step is logged to `reports/cleaning_report.json`)
- Drops duplicate `unique_key`s, parses timestamps and keeps only the study window.
- Trims and unifies text, and turns placeholder values (`Unspecified`, `N/A`, …) into missing values.
- Validates locations:
  - ZIP: 5-digit NYC ZIPs only.
  - BBL: normalizes to 10 digits (PLUTO sends `3050910009.00000000`).
  - Coordinates: keeps only points inside the NYC bounding box.
  - Borough: fills missing values from the BBL or the community board.
- Fills missing ZIPs from PLUTO through the BBL.
- Computes closure hours and flags suspicious ones: closed before created, closed in 0 hours, or open for more than 1 year. Flagged rows are kept so each analysis can decide what to do with them.
- Adds calendar features and the heat-season label (Oct–May, e.g. `2023-24`).
- PLUTO: sets `yearbuilt = 0` to missing and adds an `is_residential` flag.
- Weather: checks for date gaps and that `tmin ≤ tmax`, then adds heating degree days, 1–2 day lags, a 3-day mean, the 24-hour temperature drop, and heavy-rain and snow-day flags.

**Build features**
- `panel_zip_daily.parquet`: every (ZIP × day × complaint type) with counts (zeros included), the day's weather and the ZIP's housing stock. Counts are also given per 10k residential units.
- `building_heat_season.parquet`: every residential building × each full heat season, with its HEAT/HOT WATER complaint count, units, year built and era.
