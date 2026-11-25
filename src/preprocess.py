"""Clean the raw 311, PLUTO and weather data.

Every cleaning step is a small documented function. A `CleaningReport`
records how many rows and values each step touched, and the totals are
saved to reports/cleaning_report.json for the presentation ("we started
with X rows, Y had bad coordinates, ...").

Usage (from the project root, after `python -m src.download`):
    python -m src.preprocess
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)

BOROUGH_CODES = {"1": "MANHATTAN", "2": "BRONX", "3": "BROOKLYN", "4": "QUEENS", "5": "STATEN ISLAND"}
PLUTO_BOROUGHS = {"MN": "MANHATTAN", "BX": "BRONX", "BK": "BROOKLYN", "QN": "QUEENS", "SI": "STATEN ISLAND"}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
@dataclass
class CleaningReport:
    """Collects one entry per cleaning step for one dataset."""

    dataset: str
    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, rows_before: int, rows_after: int, detail: str = "", **extra) -> None:
        entry = {"step": step, "rows_before": rows_before, "rows_after": rows_after,
                 "rows_removed": rows_before - rows_after, "detail": detail, **extra}
        self.steps.append(entry)
        log.info("[%s] %-28s %10s -> %10s  %s", self.dataset, step, f"{rows_before:,}", f"{rows_after:,}", detail)

    def to_dict(self) -> dict:
        return {"dataset": self.dataset, "steps": self.steps}


# --------------------------------------------------------------------------
# Generic helpers (shared across datasets, and reusable by teammates)
# --------------------------------------------------------------------------
def normalize_bbl(series: pd.Series) -> pd.Series:
    """Return BBLs as 10-character strings ('3050910009'), or NA if invalid.

    311 stores BBL as '3050910009'; PLUTO returns '3050910009.00000000'.
    A valid BBL has a borough digit 1-5 followed by a 5-digit block and a
    4-digit lot, and must not be all zeros after the borough.
    """
    s = series.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    valid = s.str.fullmatch(r"[1-5]\d{9}") & ~s.str.fullmatch(r"\d0{9}")
    return s.where(valid.fillna(False))


def clean_zip(series: pd.Series) -> pd.Series:
    """Keep the first 5 digits of a ZIP code, and only if it is an NYC ZIP."""
    z = series.astype("string").str.extract(r"(\d{5})", expand=False)
    in_nyc = z.str.fullmatch(r"1(0[0-4]\d\d|1[0-6]\d\d)")   # 10000-10499, 11000-11699
    return z.where(in_nyc.fillna(False))


def clean_coordinates(df: pd.DataFrame, lat: str = "latitude", lon: str = "longitude") -> pd.DataFrame:
    """Convert coordinates to float and blank out points outside NYC."""
    df[lat] = pd.to_numeric(df[lat], errors="coerce")
    df[lon] = pd.to_numeric(df[lon], errors="coerce")
    b = config.NYC_BBOX
    inside = df[lat].between(b["lat_min"], b["lat_max"]) & df[lon].between(b["lon_min"], b["lon_max"])
    df.loc[~inside, [lat, lon]] = np.nan
    return df


# --------------------------------------------------------------------------
# 311 cleaning steps
# --------------------------------------------------------------------------
def load_raw_311() -> pd.DataFrame:
    """Load the raw 311 study data.

    Prefers the subset that notebooks/00_transform_311 extracts from the full
    dataset; falls back to the per-type trimmed download.
    """
    if config.STUDY_SUBSET_FILE.exists():
        log.info("Loading study subset from %s", config.STUDY_SUBSET_FILE)
        df = pd.read_parquet(config.STUDY_SUBSET_FILE)
    else:
        files = sorted(config.RAW_311_DIR.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError("No 311 data. Run notebooks/00_transform_311 (after "
                                    "`python -m src.download --only 311full`) or `python -m src.download --only 311`.")
        log.info("Study subset not found; loading %d trimmed-download chunks", len(files))
        df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    missing = set(config.COLUMNS_311) - set(df.columns)
    if missing:
        raise ValueError(f"Raw 311 data is missing expected columns: {missing}")
    return df


def drop_duplicate_requests(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    """Remove repeated unique_keys (overlapping page boundaries or re-downloads)."""
    n = len(df)
    df = df.drop_duplicates(subset="unique_key", keep="last")
    rep.add("drop_duplicate_keys", n, len(df), "duplicate unique_key")
    return df


def parse_dates(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    """Parse timestamps, drop rows without a valid created_date, keep the study window."""
    n = len(df)
    for col in ("created_date", "closed_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce", format="ISO8601")
    df = df[df["created_date"].notna()]
    start, end = pd.Timestamp(config.START_DATE), pd.Timestamp(config.END_DATE) + pd.Timedelta(days=1)
    df = df[(df["created_date"] >= start) & (df["created_date"] < end)]
    rep.add("parse_dates", n, len(df), "unparseable or out-of-window created_date")
    return df


def standardize_text(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    """Trim whitespace, unify casing and map placeholder values to missing."""
    n = len(df)
    placeholders = {"", "UNSPECIFIED", "N/A", "NA", "NONE", "UNKNOWN"}
    text_cols = ["agency", "complaint_type", "descriptor", "location_type", "status", "incident_address",
                 "address_type", "borough", "community_board", "open_data_channel_type"]
    n_placeholder = 0
    for col in text_cols:
        s = df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        is_ph = s.str.upper().isin(placeholders)
        n_placeholder += int(is_ph.sum())
        df[col] = s.mask(is_ph)

    # Canonical complaint_type spelling from config (the API mixes casing over the years).
    canon = {c.upper(): c for c in config.COMPLAINT_TYPES}
    df["complaint_type"] = df["complaint_type"].str.upper().map(canon).astype("string")
    df["borough"] = df["borough"].str.upper()
    df["open_data_channel_type"] = df["open_data_channel_type"].str.upper()
    df["status"] = df["status"].str.title()
    rep.add("standardize_text", n, len(df), f"{n_placeholder:,} placeholder values set to missing",
            placeholders_replaced=n_placeholder)
    return df


def clean_locations(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    """Validate ZIP, BBL, borough and coordinates. Missing values stay missing (no rows dropped here)."""
    n = len(df)
    before = {c: int(df[c].notna().sum()) for c in ("incident_zip", "bbl", "borough", "latitude")}

    df["incident_zip"] = clean_zip(df["incident_zip"])
    df["bbl"] = normalize_bbl(df["bbl"])
    df = clean_coordinates(df)

    # Borough: drop invalid names, then fill from the BBL's first digit or the community board.
    df["borough"] = df["borough"].where(df["borough"].isin(config.VALID_BOROUGHS))
    from_bbl = df["bbl"].str[0].map(BOROUGH_CODES)
    from_cb = df["community_board"].str.extract(r"\d+\s+(.+)$", expand=False).str.upper()
    from_cb = from_cb.where(from_cb.isin(config.VALID_BOROUGHS))
    df["borough"] = df["borough"].fillna(from_bbl).fillna(from_cb)

    after = {c: int(df[c].notna().sum()) for c in before}
    detail = ", ".join(f"{c}: {before[c]:,}->{after[c]:,} valid" for c in before)
    rep.add("clean_locations", n, len(df), detail, valid_before=before, valid_after=after)
    return df


def impute_zip_from_pluto(df: pd.DataFrame, pluto: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    """Fill a missing incident_zip from PLUTO when the request has a valid BBL."""
    n = len(df)
    missing_before = int(df["incident_zip"].isna().sum())
    zip_by_bbl = pluto.set_index("bbl")["zipcode"]
    df["incident_zip"] = df["incident_zip"].fillna(df["bbl"].map(zip_by_bbl))
    filled = missing_before - int(df["incident_zip"].isna().sum())
    rep.add("impute_zip_from_pluto", n, len(df), f"{filled:,} ZIPs filled via BBL", zips_filled=filled)
    return df


def add_closure_features(df: pd.DataFrame, rep: CleaningReport) -> pd.DataFrame:
    """Compute closure time in hours and flag suspicious values (rows are kept; flags let analyses filter)."""
    n = len(df)
    hours = (df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600
    df["flag_closed_before_created"] = hours < 0
    df["flag_zero_hour_close"] = hours == 0
    df["flag_very_long_close"] = hours > config.MAX_CLOSURE_HOURS
    df["closure_hours"] = hours.mask(hours < 0)
    df["is_closed"] = df["status"].eq("Closed") & df["closed_date"].notna()
    counts = {k: int(df[k].sum()) for k in ("flag_closed_before_created", "flag_zero_hour_close", "flag_very_long_close")}
    rep.add("closure_features", n, len(df), ", ".join(f"{k}={v:,}" for k, v in counts.items()), **counts)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features used for joining to weather and for EDA."""
    ts = df["created_date"]
    df["date"] = ts.dt.normalize()
    df["year"] = ts.dt.year.astype("int16")
    df["month"] = ts.dt.month.astype("int8")
    df["day_of_week"] = ts.dt.day_of_week.astype("int8")
    df["hour"] = ts.dt.hour.astype("int8")
    df["is_heat_season"] = df["month"].isin(config.HEAT_SEASON_MONTHS)
    # Heat season label: Oct 2023 – May 2024 -> "2023-24"
    season_start = np.where(df["month"] >= 10, df["year"], df["year"] - 1)
    df["heat_season"] = pd.Series(season_start, index=df.index).astype(str) + "-" + \
        pd.Series((season_start + 1) % 100, index=df.index).astype(str).str.zfill(2)
    df.loc[~df["is_heat_season"], "heat_season"] = pd.NA
    return df


def clean_311(pluto: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Run every 311 cleaning step in order."""
    rep = CleaningReport("311")
    df = load_raw_311()
    rep.add("load_raw", len(df), len(df), f"{df['complaint_type'].nunique()} complaint types")
    df = drop_duplicate_requests(df, rep)
    df = parse_dates(df, rep)
    df = standardize_text(df, rep)
    df = clean_locations(df, rep)
    df = impute_zip_from_pluto(df, pluto, rep)
    df = add_closure_features(df, rep)
    df = add_time_features(df)
    df = df.sort_values("created_date").reset_index(drop=True)
    return df, rep


# --------------------------------------------------------------------------
# PLUTO
# --------------------------------------------------------------------------
def clean_pluto() -> tuple[pd.DataFrame, CleaningReport]:
    """Normalize BBLs, convert numerics and treat yearbuilt=0 as unknown."""
    rep = CleaningReport("pluto")
    df = pd.read_parquet(config.RAW_PLUTO_FILE)
    n = len(df)
    rep.add("load_raw", n, n)

    df["bbl"] = normalize_bbl(df["bbl"])
    df = df[df["bbl"].notna()].drop_duplicates("bbl")
    rep.add("valid_unique_bbl", n, len(df), "invalid or duplicate BBL")

    for col in ("yearbuilt", "unitsres", "unitstotal", "numfloors", "latitude", "longitude"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    unknown_year = int((df["yearbuilt"] == 0).sum())
    df["yearbuilt"] = df["yearbuilt"].mask(df["yearbuilt"] < 1800)
    df["unitsres"] = df["unitsres"].fillna(0)
    df["zipcode"] = clean_zip(df["zipcode"])
    df["borough"] = df["borough"].map(PLUTO_BOROUGHS)
    df = clean_coordinates(df)
    df["is_residential"] = df["unitsres"] > 0
    rep.add("convert_types", len(df), len(df), f"{unknown_year:,} lots with yearbuilt=0 set to missing",
            unknown_yearbuilt=unknown_year)
    return df.reset_index(drop=True), rep


# --------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------
def clean_weather() -> tuple[pd.DataFrame, CleaningReport]:
    """Rename columns, validate the daily series and derive weather features."""
    rep = CleaningReport("weather")
    df = pd.read_csv(config.RAW_WEATHER_FILE)
    n = len(df)
    df = df.rename(columns={
        "time": "date", "temperature_2m_max": "tmax", "temperature_2m_min": "tmin",
        "temperature_2m_mean": "tmean", "precipitation_sum": "precip", "rain_sum": "rain",
        "snowfall_sum": "snow", "wind_speed_10m_max": "wind_max",
    })
    df["date"] = pd.to_datetime(df["date"])

    expected = pd.date_range(config.START_DATE, config.END_DATE, freq="D")
    gaps = expected.difference(df["date"])
    if len(gaps):
        log.warning("Weather is missing %d days; interpolating", len(gaps))
    df = df.set_index("date").reindex(expected).rename_axis("date")
    df = df.interpolate(limit=3).reset_index()
    bad = int((df["tmin"] > df["tmax"]).sum())
    if bad:
        raise ValueError(f"{bad} weather days have tmin > tmax")

    # Features: heating degree days, lags, rolling means, event flags.
    df["hdd"] = (65 - df["tmean"]).clip(lower=0)
    for lag in (1, 2):
        df[f"tmin_lag{lag}"] = df["tmin"].shift(lag)
        df[f"precip_lag{lag}"] = df["precip"].shift(lag)
    df["tmin_3day_mean"] = df["tmin"].rolling(3, min_periods=1).mean()
    df["temp_drop_24h"] = df["tmin"].shift(1) - df["tmin"]          # positive = got colder
    df["heavy_rain"] = df["precip"] >= config.HEAVY_RAIN_INCHES
    df["snow_day"] = df["snow"] > 0
    rep.add("validate_and_features", n, len(df), f"{len(gaps)} missing days filled", missing_days=len(gaps))
    return df, rep


# --------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    weather, rep_w = clean_weather()
    pluto, rep_p = clean_pluto()
    complaints, rep_c = clean_311(pluto)

    weather.to_parquet(config.CLEAN_WEATHER_FILE, index=False)
    pluto.to_parquet(config.CLEAN_PLUTO_FILE, index=False)
    complaints.to_parquet(config.CLEAN_311_FILE, index=False)

    summary = {
        "reports": [r.to_dict() for r in (rep_c, rep_p, rep_w)],
        "complaints_by_type": complaints["complaint_type"].value_counts().to_dict(),
        "missing_share_after_cleaning": complaints[["incident_zip", "bbl", "borough", "latitude", "closed_date"]]
        .isna().mean().round(4).to_dict(),
    }
    config.CLEANING_REPORT_FILE.write_text(json.dumps(summary, indent=2, default=str))
    log.info("Saved clean files to %s and report to %s", config.PROCESSED_DIR, config.CLEANING_REPORT_FILE)


if __name__ == "__main__":
    main()
