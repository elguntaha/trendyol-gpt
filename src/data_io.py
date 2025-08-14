# src/data_io.py
from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc

from src.config import load_config
from src.schemas import get_schema, PATHS

# -------- Helpers --------
def _ensure_utc(dt: pd.Series) -> pd.Series:
    """Ensure pandas datetime64[ns, UTC]. Accepts tz-naive or tz-aware; coerces to UTC."""
    # Convert PyArrow backed Series to regular pandas first to avoid timezone issues
    if hasattr(dt.dtype, 'pyarrow_dtype'):
        dt = dt.astype('object')
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt, errors="coerce")
    
    # Handle timezone conversion
    try:
        if dt.dt.tz is None:
            # Naive datetime - assume UTC
            dt = dt.dt.tz_localize("UTC")
        else:
            # Already timezone-aware - convert to UTC
            dt = dt.dt.tz_convert("UTC")
    except Exception:
        # If all else fails, just convert to naive datetime
        dt = pd.to_datetime(dt, errors="coerce")
    
    return dt

def _maybe_to_category(s: pd.Series, max_ratio: float = 0.5, max_uniques: int = 1_000_000) -> pd.Series:
    """Convert string-like series to category if unique ratio is small."""
    if not pd.api.types.is_string_dtype(s): 
        return s
    n = len(s)
    if n == 0: 
        return s
    u = s.nunique(dropna=True)
    if u <= max_uniques and (u / n) <= max_ratio:
        return s.astype("category")
    return s

def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64", "int32"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df

def _apply_dtypes(df: pd.DataFrame, dtypes: Dict[str, str], make_categorical: bool) -> pd.DataFrame:
    for col, dtype in dtypes.items():
        if col not in df.columns:
            continue
        if dtype == "string":
            df[col] = df[col].astype("string")
            if make_categorical:
                df[col] = _maybe_to_category(df[col])
        else:
            try:
                df[col] = df[col].astype(dtype)  # type: ignore
            except Exception:
                # Fallback safe cast
                if "int" in dtype:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                elif "float" in dtype:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    return df

def _apply_datetime(df: pd.DataFrame, datetime_cols: Sequence[str]) -> pd.DataFrame:
    for col in datetime_cols:
        if col in df.columns:
            df[col] = _ensure_utc(df[col])
    return df

def memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(index=True, deep=True).sum()) / (1024 ** 2)

# -------- Core loader --------
def load_table(
    key: str,
    data_dir: Union[str, Path],
    columns: Optional[Sequence[str]] = None,
    row_filter: Optional[pc.Expression] = None,
    make_categorical: bool = False,
) -> pd.DataFrame:
    """
    Load a known dataset by key using pyarrow.dataset for efficient column/row filtering.
    - key: one of schemas.PATHS keys
    - columns: subset of columns to read (plus any datetime columns always added if requested)
    - row_filter: optional pyarrow expression (e.g., ds.field("date") >= pa.scalar(pd.Timestamp("2024-01-01")))
    - make_categorical: convert eligible string columns to pandas 'category'
    """
    rel_path, dtypes, dt_cols = get_schema(key)
    path = Path(data_dir) / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    dataset = ds.dataset(str(path))
    # Ensure datetime columns included if requested in columns
    read_cols = list(columns) if columns is not None else None

    table = dataset.to_table(columns=read_cols, filter=row_filter) if row_filter is not None else dataset.to_table(columns=read_cols)
    df = table.to_pandas(types_mapper=lambda t: pd.ArrowDtype(t))

    # Apply recommended dtypes & conversions
    df = _apply_datetime(df, dt_cols)
    df = _apply_dtypes(df, dtypes, make_categorical=make_categorical)
    df = _downcast_numeric(df)
    return df

# -------- Convenience wrappers for each dataset --------
def load_train_sessions(data_dir: Union[str, Path], columns: Optional[Sequence[str]] = None, **kwargs) -> pd.DataFrame:
    return load_table("train_sessions", data_dir, columns=columns, **kwargs)

def load_test_sessions(data_dir: Union[str, Path], columns: Optional[Sequence[str]] = None, **kwargs) -> pd.DataFrame:
    return load_table("test_sessions", data_dir, columns=columns, **kwargs)

def load_content_metadata(data_dir, columns=None, **kwargs): 
    return load_table("content_metadata", data_dir, columns, **kwargs)

def load_content_price_rate_review(data_dir, columns=None, **kwargs):
    return load_table("content_price_rate_review", data_dir, columns, **kwargs)

def load_content_search_log(data_dir, columns=None, **kwargs):
    return load_table("content_search_log", data_dir, columns, **kwargs)

def load_content_sitewide_log(data_dir, columns=None, **kwargs):
    return load_table("content_sitewide_log", data_dir, columns, **kwargs)

def load_content_top_terms_log(data_dir, columns=None, **kwargs):
    return load_table("content_top_terms_log", data_dir, columns, **kwargs)

def load_user_metadata(data_dir, columns=None, **kwargs): 
    return load_table("user_metadata", data_dir, columns, **kwargs)

def load_user_search_log(data_dir, columns=None, **kwargs):
    return load_table("user_search_log", data_dir, columns, **kwargs)

def load_user_sitewide_log(data_dir, columns=None, **kwargs):
    return load_table("user_sitewide_log", data_dir, columns, **kwargs)

def load_user_top_terms_log(data_dir, columns=None, **kwargs):
    return load_table("user_top_terms_log", data_dir, columns, **kwargs)

def load_user_fashion_search_log(data_dir, columns=None, **kwargs):
    return load_table("user_fashion_search_log", data_dir, columns, **kwargs)

def load_user_fashion_sitewide_log(data_dir, columns=None, **kwargs):
    return load_table("user_fashion_sitewide_log", data_dir, columns, **kwargs)

def load_term_search_log(data_dir, columns=None, **kwargs):
    return load_table("term_search_log", data_dir, columns, **kwargs)

# -------- Example row filter builders --------
def build_time_range_filter(column: str, start=None, end=None) -> Optional[pc.Expression]:
    """
    Build a pyarrow filter for a datetime column ('ts_hour' or 'date').
    start/end can be pandas.Timestamp, string, or None.
    """
    expr = None
    if start is not None:
        expr = (ds.field(column) >= pa.scalar(pd.to_datetime(start)))
    if end is not None:
        rhs = (ds.field(column) <= pa.scalar(pd.to_datetime(end)))
        expr = rhs if expr is None else (expr & rhs)
    return expr

def head(path: Union[str, Path], n: int = 5) -> pd.DataFrame:
    """Quick preview of a parquet file (no schema application)."""
    dataset = ds.dataset(str(path))
    tbl = dataset.to_table()
    df = tbl.to_pandas()
    return df.head(n)
