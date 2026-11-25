"""Central configuration for the weather-sensitivity pipeline.

Every path, date range, complaint type and API endpoint lives here, so the
download and preprocessing scripts never hard-code them. Change a value
here and re-run; cached downloads that already exist are reused.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_311_DIR = RAW_DIR / "311"            # one parquet file per (type, month) – trimmed download
RAW_311_FULL_DIR = RAW_DIR / "311_full"  # one parquet file per month – every row and column
INTERIM_DIR = DATA_DIR / "interim"
STUDY_SUBSET_FILE = INTERIM_DIR / "311_study_subset.parquet"   # written by notebooks/00_transform_311
FULL_START_DATE = "2020-01-01"           # first month of the full 311 dataset
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_PLUTO_FILE = RAW_DIR / "pluto.parquet"
RAW_WEATHER_FILE = RAW_DIR / "weather_daily.csv"

CLEAN_311_FILE = PROCESSED_DIR / "complaints_clean.parquet"
CLEAN_PLUTO_FILE = PROCESSED_DIR / "pluto_clean.parquet"
CLEAN_WEATHER_FILE = PROCESSED_DIR / "weather_clean.parquet"
PANEL_ZIP_DAILY_FILE = PROCESSED_DIR / "panel_zip_daily.parquet"
BUILDING_HEAT_FILE = PROCESSED_DIR / "building_heat_season.parquet"
CLEANING_REPORT_FILE = REPORTS_DIR / "cleaning_report.json"

# --------------------------------------------------------------------------
# Study scope
# --------------------------------------------------------------------------
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"   # inclusive

# Weather-sensitive complaint types, plus Noise - Residential as a
# "control" category that showed no temperature relationship in exploration.
COMPLAINT_TYPES = [
    "HEAT/HOT WATER",
    "Sewer",
    "Water System",
    "Snow or Ice",
    "Noise - Street/Sidewalk",
    "Noise - Residential",
    "Rodent",
    "Dirty Condition",
]

# Columns pulled from the 311 API (the full table has 44; most are sparse).
COLUMNS_311 = [
    "unique_key",
    "created_date",
    "closed_date",
    "agency",
    "complaint_type",
    "descriptor",
    "location_type",
    "status",
    "incident_zip",
    "incident_address",
    "address_type",
    "borough",
    "community_board",
    "bbl",
    "latitude",
    "longitude",
    "open_data_channel_type",
]

COLUMNS_PLUTO = [
    "bbl", "borough", "zipcode", "cd", "yearbuilt", "unitsres", "unitstotal",
    "numfloors", "bldgclass", "landuse", "latitude", "longitude",
]

# --------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------
NYC_311_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
PLUTO_URL = "https://data.cityofnewyork.us/resource/64uk-42ks.csv"

# Open-Meteo historical archive (free, no key). Point = Central Park.
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_LAT, WEATHER_LON = 40.7812, -73.9665
WEATHER_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
]

# Socrata paging. An optional app token (free, from data.cityofnewyork.us)
# raises rate limits; set it as the environment variable NYC_APP_TOKEN.
PAGE_SIZE = 50_000
MAX_WORKERS = 4
REQUEST_TIMEOUT = 300   # seconds
MAX_RETRIES = 4

# --------------------------------------------------------------------------
# Cleaning rules
# --------------------------------------------------------------------------
# Rough bounding box around the five boroughs; points outside are bad geocodes.
NYC_BBOX = {"lat_min": 40.48, "lat_max": 40.93, "lon_min": -74.27, "lon_max": -73.68}
VALID_BOROUGHS = {"MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"}
MAX_CLOSURE_HOURS = 24 * 365          # longer closures get flagged as outliers
HEAVY_RAIN_INCHES = 0.5
HEAT_SEASON_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5]   # NYC heat season: Oct 1 – May 31
