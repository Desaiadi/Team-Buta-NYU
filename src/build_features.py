"""Build the analysis tables from the cleaned data.

Outputs:
  panel_zip_daily.parquet      one row per (ZIP, date, complaint type): the count
                               (zeros included), the day's weather and the ZIP's
                               housing stock from PLUTO. Used to estimate each
                               area's weather sensitivity.
  building_heat_season.parquet one row per (residential building, full heat season):
                               the number of HEAT/HOT WATER complaints, units and
                               year built. Used to ask whether older buildings
                               complain more in the cold.

Usage (after `python -m src.preprocess`):
    python -m src.build_features
"""

from __future__ import annotations

import logging

import pandas as pd

from src import config

log = logging.getLogger(__name__)

HEAT_TYPE = "HEAT/HOT WATER"


def zip_housing_profile(pluto: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PLUTO residential lots to ZIP level (exposure denominators)."""
    res = pluto[pluto["is_residential"] & pluto["zipcode"].notna()].copy()
    res["pre1940_units"] = res["unitsres"].where(res["yearbuilt"] < 1940, 0)
    prof = res.groupby("zipcode").agg(
        res_buildings=("bbl", "size"),
        res_units=("unitsres", "sum"),
        median_year_built=("yearbuilt", "median"),
        pre1940_units=("pre1940_units", "sum"),
    )
    prof["share_units_pre1940"] = (prof["pre1940_units"] / prof["res_units"]).round(4)
    return prof.drop(columns="pre1940_units").rename_axis("incident_zip").reset_index()


def build_zip_daily_panel(complaints: pd.DataFrame, weather: pd.DataFrame, pluto: pd.DataFrame,
                          min_res_units: int = 500) -> pd.DataFrame:
    """Return complaint counts per (ZIP, date, type) on a complete grid, joined to weather and housing."""
    profile = zip_housing_profile(pluto)
    zips = profile.loc[profile["res_units"] >= min_res_units, "incident_zip"]

    counts = (complaints[complaints["incident_zip"].isin(zips)]
              .groupby(["incident_zip", "date", "complaint_type"], observed=True).size().rename("n"))

    # Complete grid, so days with zero complaints are present (essential for regressions).
    grid = pd.MultiIndex.from_product(
        [sorted(zips), pd.date_range(config.START_DATE, config.END_DATE, freq="D"), config.COMPLAINT_TYPES],
        names=["incident_zip", "date", "complaint_type"])
    panel = counts.reindex(grid, fill_value=0).reset_index()

    panel = panel.merge(weather, on="date", how="left").merge(profile, on="incident_zip", how="left")
    panel["n_per_10k_units"] = panel["n"] / panel["res_units"] * 10_000
    panel["day_of_week"] = panel["date"].dt.day_of_week.astype("int8")
    panel["month"] = panel["date"].dt.month.astype("int8")
    panel["year"] = panel["date"].dt.year.astype("int16")
    panel["complaint_type"] = panel["complaint_type"].astype("category")
    log.info("ZIP panel: %d ZIPs x %d days x %d types = %s rows",
             len(zips), panel["date"].nunique(), len(config.COMPLAINT_TYPES), f"{len(panel):,}")
    return panel


def full_heat_seasons() -> list[str]:
    """Heat seasons (Oct 1 - May 31) fully inside the study window, e.g. ['2023-24', '2024-25']."""
    start, end = pd.Timestamp(config.START_DATE), pd.Timestamp(config.END_DATE)
    return [f"{y}-{(y + 1) % 100:02d}" for y in range(start.year - 1, end.year + 1)
            if pd.Timestamp(f"{y}-10-01") >= start and pd.Timestamp(f"{y + 1}-05-31") <= end]


def build_building_heat_table(complaints: pd.DataFrame, pluto: pd.DataFrame) -> pd.DataFrame:
    """Return HEAT/HOT WATER complaints per residential building per full heat season (zeros included)."""
    seasons = full_heat_seasons()
    if not seasons:
        raise ValueError("No complete heat season in the study window; widen START_DATE/END_DATE.")
    heat = complaints[(complaints["complaint_type"] == HEAT_TYPE) & complaints["heat_season"].isin(seasons)
                      & complaints["bbl"].notna()]
    counts = heat.groupby(["bbl", "heat_season"]).agg(
        heat_complaints=("unique_key", "size"),
        heat_complaint_days=("date", "nunique"))

    buildings = pluto.loc[pluto["is_residential"],
                          ["bbl", "borough", "zipcode", "yearbuilt", "unitsres", "numfloors", "bldgclass",
                           "latitude", "longitude"]]
    grid = pd.MultiIndex.from_product([buildings["bbl"], seasons], names=["bbl", "heat_season"])
    table = counts.reindex(grid, fill_value=0).reset_index().merge(buildings, on="bbl", how="left")
    table["complaints_per_100_units"] = table["heat_complaints"] / table["unitsres"] * 100
    table["building_era"] = pd.cut(table["yearbuilt"], [1799, 1919, 1939, 1959, 1979, 1999, 2100],
                                   labels=["<1920", "1920-39", "1940-59", "1960-79", "1980-99", "2000+"])
    unmatched = heat.loc[~heat["bbl"].isin(buildings["bbl"]), "bbl"].nunique()
    log.info("Building table: %s buildings x %d seasons; %d complaint BBLs not residential in PLUTO",
             f"{len(buildings):,}", len(seasons), unmatched)
    return table


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    complaints = pd.read_parquet(config.CLEAN_311_FILE)
    weather = pd.read_parquet(config.CLEAN_WEATHER_FILE)
    pluto = pd.read_parquet(config.CLEAN_PLUTO_FILE)

    build_zip_daily_panel(complaints, weather, pluto).to_parquet(config.PANEL_ZIP_DAILY_FILE, index=False)
    build_building_heat_table(complaints, pluto).to_parquet(config.BUILDING_HEAT_FILE, index=False)
    log.info("Saved %s and %s", config.PANEL_ZIP_DAILY_FILE.name, config.BUILDING_HEAT_FILE.name)


if __name__ == "__main__":
    main()
