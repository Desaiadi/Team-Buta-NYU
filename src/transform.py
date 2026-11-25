"""Profile the full 311 dataset and extract the study subset from it.

The full table (~23M rows × ~44 columns) does not fit comfortably in 16 GB of
RAM as a pandas DataFrame, so everything here works **one month file at a
time** and reads only the columns it needs. Row counts and missing-value
counts come straight from parquet metadata, without loading any data.

Used by notebooks/00_transform_311.ipynb.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src import config


def month_files(directory: Path = config.RAW_311_FULL_DIR) -> list[Path]:
    """Sorted list of the monthly parquet files (YYYY-MM.parquet)."""
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No files in {directory}. Run `python -m src.download --only 311full`.")
    return files


def inventory(files: list[Path]) -> pd.DataFrame:
    """One row per month: rows, columns and size on disk (metadata only, instant)."""
    rows = []
    for f in files:
        meta = pq.ParquetFile(f).metadata
        rows.append({"month": f.stem, "rows": meta.num_rows, "columns": meta.num_columns,
                     "size_mb": f.stat().st_size / 1e6})
    return pd.DataFrame(rows)


def column_presence(files: list[Path]) -> pd.DataFrame:
    """Which columns appear in which months.

    The API omits a column from a page when it is empty for the whole page,
    so some months lack sparse columns. This shows the schema drift.
    """
    presence = {f.stem: set(pq.read_schema(f).names) for f in files}
    all_cols = sorted(set().union(*presence.values()))
    table = pd.DataFrame({m: [c in cols for c in all_cols] for m, cols in presence.items()}, index=all_cols)
    return table


def missing_counts(files: list[Path]) -> pd.DataFrame:
    """Missing (null) values per column across the whole dataset, from parquet statistics.

    Columns absent from a month count as fully missing for that month.
    """
    all_cols = sorted(set().union(*(pq.read_schema(f).names for f in files)))
    total_rows, nulls = 0, dict.fromkeys(all_cols, 0)
    for f in files:
        meta = pq.ParquetFile(f).metadata
        n = meta.num_rows
        total_rows += n
        names = [meta.schema.column(i).name for i in range(meta.num_columns)]
        file_nulls = dict.fromkeys(names, 0)
        for rg in range(meta.num_row_groups):
            group = meta.row_group(rg)
            for i, name in enumerate(names):
                stats = group.column(i).statistics
                file_nulls[name] += stats.null_count if stats is not None and stats.has_null_count else 0
        for name in all_cols:
            nulls[name] += file_nulls.get(name, n)   # a column absent from this month = all missing
    out = pd.DataFrame({"missing": nulls}).assign(total_rows=total_rows)
    out["missing_share"] = out["missing"] / total_rows
    return out.sort_values("missing_share")


def _read_columns(f: Path, columns: list[str]) -> pd.DataFrame:
    """Read the requested columns from one month, adding any that month lacks as empty."""
    available = set(pq.read_schema(f).names)
    df = pd.read_parquet(f, columns=[c for c in columns if c in available])
    return df.reindex(columns=columns)


def stream_aggregate(files: list[Path], columns: list[str], func, combine: str = "sum"):
    """Apply `func` to each month (reading only `columns`) and combine the results.

    `func` takes one month's DataFrame and returns a Series or DataFrame.
    combine="sum" adds the results together (for counts); "concat" stacks
    them (for samples or per-month rows). This is how the notebooks explore
    23M rows without ever loading them all at once.
    """
    results = [func(_read_columns(f, columns)) for f in files]
    if combine == "concat":
        return pd.concat(results)
    out = results[0]
    for r in results[1:]:
        out = out.add(r, fill_value=0)
    return out


def profile(files: list[Path]) -> dict[str, pd.DataFrame]:
    """Streamed counts over the full dataset: requests by year, type, agency and borough."""
    cols = ["created_date", "complaint_type", "agency", "borough", "status"]
    parts = {"year": [], "complaint_type": [], "agency": [], "borough": [], "status": []}
    for f in files:
        df = _read_columns(f, cols)
        df["year"] = df["created_date"].str[:4]
        for key in parts:
            parts[key].append(df[key].value_counts(dropna=False))
    return {key: pd.concat(v).groupby(level=0).sum().sort_values(ascending=False).rename("requests").to_frame()
            for key, v in parts.items()}


def extract_study_subset(files: list[Path], complaint_types=config.COMPLAINT_TYPES,
                         start: str = config.START_DATE, end: str = config.END_DATE,
                         columns=config.COLUMNS_311) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter the full data to the study scope: complaint types, date window and columns.

    Only month files that overlap the window are opened, only the needed
    columns are read, and the type filter is an exact match (like the API's
    `complaint_type = '...'`). Returns the subset and a per-month log.
    """
    first, last = pd.Timestamp(start).strftime("%Y-%m"), pd.Timestamp(end).strftime("%Y-%m")
    in_window = [f for f in files if first <= f.stem <= last]
    wanted = set(complaint_types)
    parts, log = [], []
    for f in in_window:
        df = _read_columns(f, list(columns))
        keep = df[df["complaint_type"].isin(wanted)]
        parts.append(keep)
        log.append({"month": f.stem, "rows_in_month": len(df), "rows_kept": len(keep)})
    subset = pd.concat(parts, ignore_index=True)
    return subset, pd.DataFrame(log)


def save_subset(subset: pd.DataFrame, path: Path = config.STUDY_SUBSET_FILE) -> Path:
    """Write the subset as one compressed parquet file (all columns kept as raw strings)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(path, index=False, compression="zstd")
    return path


def trimmed_download_rows() -> int:
    """Row count of the earlier per-type download (for cross-checking), 0 if absent."""
    return sum(pq.ParquetFile(f).metadata.num_rows for f in config.RAW_311_DIR.rglob("*.parquet"))
