"""Download the raw data for the project.

Downloads three sources:
  1. NYC 311 service requests, filtered to the study's complaint types and
     dates. Fetched month by month and cached as one parquet file per
     (complaint type, month), so an interrupted run resumes where it stopped.
  2. NYC PLUTO tax-lot data: building age, residential units and so on.
  3. Daily weather for Central Park from the Open-Meteo archive.

Usage (from the project root):
    python -m src.download              # everything
    python -m src.download --only 311   # just one source: 311 | 311full | pluto | weather
    python -m src.download --only 311full   # the complete 311 table (~23M rows, ~3 GB)
    python -m src.download --force      # ignore the cache and re-download
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from src import config

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------
def _session() -> requests.Session:
    """Return a requests session, adding the Socrata app token if one is set."""
    s = requests.Session()
    token = os.environ.get("NYC_APP_TOKEN")
    if token:
        s.headers["X-App-Token"] = token
    return s


def _get_with_retry(session: requests.Session, url: str, params: dict) -> requests.Response:
    """GET a URL, retrying with exponential backoff on errors and throttling."""
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp
            log.warning("HTTP %s (attempt %d): %s", resp.status_code, attempt, resp.text[:200])
        except requests.RequestException as exc:
            log.warning("Request failed (attempt %d): %s", attempt, exc)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Giving up on {url} after {config.MAX_RETRIES} attempts")


def socrata_query(session: requests.Session, url: str, select: list[str] | None,
                  where: str | None = None, order: str | None = None) -> pd.DataFrame:
    """Page through a Socrata (NYC Open Data) endpoint and return every row.

    All columns are read as strings; type conversion belongs in preprocessing,
    so the raw files stay exactly what the API returned. select=None returns
    every column.
    """
    frames, offset = [], 0
    while True:
        params = {"$limit": config.PAGE_SIZE, "$offset": offset}
        if select:
            params["$select"] = ",".join(select)
        if where:
            params["$where"] = where
        if order:
            params["$order"] = order
        resp = _get_with_retry(session, url, params)
        page = pd.read_csv(io.StringIO(resp.text), dtype=str)
        frames.append(page)
        if len(page) < config.PAGE_SIZE:
            break
        offset += config.PAGE_SIZE
    df = pd.concat(frames, ignore_index=True)
    # A column that is empty for the whole page is dropped by the API, so restore it.
    return df.reindex(columns=select) if select else df


# --------------------------------------------------------------------------
# 1. NYC 311
# --------------------------------------------------------------------------
def _month_starts(start: str, end: str) -> list[pd.Timestamp]:
    """First day of every month between start and end (inclusive)."""
    return list(pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"))


def _chunk_path(complaint_type: str, month: pd.Timestamp):
    """Cache-file path for one (complaint type, month) chunk."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in complaint_type).strip("_")
    return config.RAW_311_DIR / safe / f"{month:%Y-%m}.parquet"


def _download_311_chunk(complaint_type: str, month: pd.Timestamp, force: bool) -> tuple[str, int]:
    """Download one month of one complaint type unless it is already cached."""
    path = _chunk_path(complaint_type, month)
    if path.exists() and not force:
        return f"{complaint_type} {month:%Y-%m} (cached)", 0
    nxt = month + pd.offsets.MonthBegin(1)
    ct = complaint_type.replace("'", "''")
    where = (f"complaint_type = '{ct}' AND created_date >= '{month:%Y-%m-%d}T00:00:00' "
             f"AND created_date < '{nxt:%Y-%m-%d}T00:00:00'")
    df = socrata_query(_session(), config.NYC_311_URL, config.COLUMNS_311, where, order="unique_key")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return f"{complaint_type} {month:%Y-%m}", len(df)


def download_311(force: bool = False) -> None:
    """Download every (complaint type, month) chunk in parallel threads."""
    months = _month_starts(config.START_DATE, config.END_DATE)
    jobs = [(ct, m) for ct in config.COMPLAINT_TYPES for m in months]
    log.info("311: %d chunks (%d types x %d months)", len(jobs), len(config.COMPLAINT_TYPES), len(months))
    total = 0
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {pool.submit(_download_311_chunk, ct, m, force): (ct, m) for ct, m in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            label, n = fut.result()
            total += n
            log.info("[%d/%d] %s: %s rows", i, len(jobs), label, f"{n:,}")
    log.info("311 done: %s new rows downloaded", f"{total:,}")


def _download_311_full_month(month: pd.Timestamp, force: bool) -> tuple[str, int]:
    """Download every row and column created in one month, unless cached."""
    path = config.RAW_311_FULL_DIR / f"{month:%Y-%m}.parquet"
    if path.exists() and not force:
        return f"{month:%Y-%m} (cached)", 0
    nxt = month + pd.offsets.MonthBegin(1)
    where = f"created_date >= '{month:%Y-%m-%d}T00:00:00' AND created_date < '{nxt:%Y-%m-%d}T00:00:00'"
    df = socrata_query(_session(), config.NYC_311_URL, None, where, order="unique_key")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.rename(path)   # only a complete month ever gets the final name
    return f"{month:%Y-%m}", len(df)


def download_311_full(force: bool = False) -> None:
    """Download the complete 311 table (2020 to the current month), one file per month.

    The current, still-growing month is always re-downloaded so it stays fresh.
    """
    months = _month_starts(config.FULL_START_DATE, pd.Timestamp.today().normalize())
    current = pd.Timestamp.today().normalize().replace(day=1)
    log.info("311 full: %d months", len(months))
    total = 0
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = [pool.submit(_download_311_full_month, m, force or m == current) for m in months]
        for i, fut in enumerate(as_completed(futures), 1):
            label, n = fut.result()
            total += n
            log.info("[%d/%d] %s: %s rows", i, len(months), label, f"{n:,}")
    log.info("311 full done: %s new rows downloaded", f"{total:,}")


# --------------------------------------------------------------------------
# 2. PLUTO
# --------------------------------------------------------------------------
def download_pluto(force: bool = False) -> None:
    """Download the PLUTO columns needed for building-level denominators."""
    if config.RAW_PLUTO_FILE.exists() and not force:
        log.info("PLUTO: cached at %s", config.RAW_PLUTO_FILE)
        return
    df = socrata_query(_session(), config.PLUTO_URL, config.COLUMNS_PLUTO, order="bbl")
    config.RAW_PLUTO_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.RAW_PLUTO_FILE, index=False)
    log.info("PLUTO: %s tax lots saved", f"{len(df):,}")


# --------------------------------------------------------------------------
# 3. Weather
# --------------------------------------------------------------------------
def download_weather(force: bool = False) -> None:
    """Download daily Central Park weather (°F, inches, mph) from Open-Meteo."""
    if config.RAW_WEATHER_FILE.exists() and not force:
        log.info("Weather: cached at %s", config.RAW_WEATHER_FILE)
        return
    params = {
        "latitude": config.WEATHER_LAT,
        "longitude": config.WEATHER_LON,
        "start_date": config.START_DATE,
        "end_date": config.END_DATE,
        "daily": ",".join(config.WEATHER_DAILY_VARS),
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "wind_speed_unit": "mph",
        "timezone": "America/New_York",
    }
    resp = _get_with_retry(requests.Session(), config.WEATHER_URL, params)
    df = pd.DataFrame(resp.json()["daily"])
    config.RAW_WEATHER_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_WEATHER_FILE, index=False)
    log.info("Weather: %d days saved", len(df))


SOURCES = {"311": download_311, "311full": download_311_full, "pluto": download_pluto, "weather": download_weather}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=SOURCES, help="download a single source")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    # Weather and PLUTO are fast, so fetch them first.
    for name in (["weather", "pluto", "311"] if args.only is None else [args.only]):
        SOURCES[name](force=args.force)


if __name__ == "__main__":
    main()
