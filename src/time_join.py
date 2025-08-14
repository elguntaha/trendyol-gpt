# src/time_join.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- Utilities ----------
def to_utc_hour(s: pd.Series) -> pd.Series:
    """Normalize timestamps to timezone-aware UTC and floor to hour."""
    _s = pd.to_datetime(s, utc=True, errors="coerce")
    return _s.dt.floor("h")

def bayesian_ctr(clicks: pd.Series, imps: pd.Series, alpha: float = 10.0, beta: float = 10.0) -> pd.Series:
    """Compute smoothed CTR = (clicks+alpha)/(imps+alpha+beta)."""
    c = clicks.fillna(0).astype("float32")
    i = imps.fillna(0).astype("float32")
    return (c + alpha) / (i + alpha + beta)

# ---------- Cumulative log preparation ----------
def prepare_cumsum_log(
    log_df: pd.DataFrame,
    key_cols: Sequence[str],
    time_col: str,
    value_cols: Sequence[str],
    freq: str = "h",
) -> pd.DataFrame:
    """
    Aggregate a raw log to hourly/daily buckets per key and compute cumulative sums.
    Returns a table with columns: key..., time_col, <val>_sum, <val>_cumsum.
    
    Strictly enforces no future lookups: all cumulative sums use only events ≤ ts_hour.
    """
    assert all(c in log_df.columns for c in list(key_cols) + [time_col]), "Missing key/time columns"
    df = log_df.loc[:, list(key_cols) + [time_col] + list(value_cols)].copy()
    df[time_col] = to_utc_hour(df[time_col])
    # Aggregate to period
    grp = df.groupby(list(key_cols) + [time_col], observed=True).sum(numeric_only=True).reset_index()
    grp = grp.sort_values(list(key_cols) + [time_col]).reset_index(drop=True)
    # Cumulative sums per key
    for v in value_cols:
        grp[f"{v}_cumsum"] = grp.groupby(list(key_cols), observed=True)[v].cumsum()
    return grp

# ---------- As-of join helpers ----------
def asof_join_latest(
    left: pd.DataFrame,
    right: pd.DataFrame,
    by: Sequence[str],
    left_on: str,
    right_on: str,
    suffix: str,
) -> pd.DataFrame:
    """
    Per-key asof merge: for each left row timestamp, bring the last <= right timestamp.
    Both left[left_on] and right[right_on] must be sorted within groups.
    
    Enforces no future lookups: only uses right records where right_on ≤ left_on.
    """
    # Convert timestamp columns to ensure they're datetime
    left = left.copy()
    right = right.copy()
    left[left_on] = pd.to_datetime(left[left_on])
    right[right_on] = pd.to_datetime(right[right_on])
    
    # Sort by ALL by columns AND the time column for merge_asof
    left = left.sort_values(list(by) + [left_on]).reset_index(drop=True)
    right = right.sort_values(list(by) + [right_on]).reset_index(drop=True)
    
    merged = pd.merge_asof(
        left,
        right,
        left_on=left_on,
        right_on=right_on,
        by=list(by),
        direction="backward",
        allow_exact_matches=True,
    )
    return merged

def add_window_diff_features(
    sessions: pd.DataFrame,
    cumsum_table: pd.DataFrame,
    key_cols: Sequence[str],
    ts_col: str,
    time_col: str,
    value_cols: Sequence[str],
    windows_hours: Sequence[int],
    prefix: str,
) -> pd.DataFrame:
    """
    For each window w (in hours), compute sum over (ts - w, ts] by cumsum(ts) - cumsum(ts - w).
    Implementation: two asof joins per window (current and past).
    Returns sessions with added columns: {prefix}{val}_sum_{w}h for each val and window.
    
    Strictly enforces no future lookups: only uses events ≤ ts_hour for all windows.
    """
    assert ts_col in sessions.columns
    # Normalize session ts to UTC hour
    base = sessions.copy()
    base[ts_col] = to_utc_hour(base[ts_col])

    # 1) Current cumsums at ts
    cur_cols = [*key_cols, time_col] + [f"{v}_cumsum" for v in value_cols] + list(value_cols)
    cur = cumsum_table.loc[:, cur_cols].copy()
    cur = cur.rename(columns={time_col: f"{time_col}_right"})
    cur_join = asof_join_latest(
        left=base,
        right=cur,
        by=key_cols,
        left_on=ts_col,
        right_on=f"{time_col}_right",
        suffix="_cur",
    )

    # For each window, compute past ts and join past cumsums
    out = cur_join
    for w in windows_hours:
        past = base.copy()
        past["_past_ts"] = past[ts_col] - pd.to_timedelta(w, unit="h")
        past = past.loc[:, [*key_cols, ts_col, "_past_ts"]]

        past_right = cumsum_table.loc[:, [*key_cols, time_col] + [f"{v}_cumsum" for v in value_cols]].copy()
        past_right = past_right.rename(columns={time_col: "_past_right"})

        past_join = asof_join_latest(
            left=past,
            right=past_right,
            by=key_cols,
            left_on="_past_ts",
            right_on="_past_right",
            suffix="_past",
        )
        # Compute differences: sum(ts) - sum(ts - w)
        for v in value_cols:
            cur_cum = out.get(f"{v}_cumsum")
            if cur_cum is None:
                # merge the current cum columns from cur_join if not yet present (first iteration)
                out[f"{v}_cumsum"] = out[f"{v}_cumsum"]
            past_cum = past_join.get(f"{v}_cumsum")
            win_col = f"{prefix}{v}_sum_{w}h"
            out[win_col] = out[f"{v}_cumsum"].fillna(0) - past_join[f"{v}_cumsum"].fillna(0)
    return out

# ---------- High-level convenience wrappers ----------
def time_join_sum_windows(
    sessions: pd.DataFrame,
    log_df: pd.DataFrame,
    key_cols: Sequence[str],
    session_ts_col: str,
    log_time_col: str,
    value_cols: Sequence[str],
    windows_hours: Sequence[int],
    prefix: str,
) -> pd.DataFrame:
    """
    One-stop: cumsum prep + window diffs for a given log onto sessions.
    
    Strictly enforces no future lookups: all windowed features use only events ≤ ts_hour.
    """
    cumsum = prepare_cumsum_log(log_df, key_cols=key_cols, time_col=log_time_col, value_cols=value_cols)
    out = add_window_diff_features(
        sessions=sessions,
        cumsum_table=cumsum,
        key_cols=key_cols,
        ts_col=session_ts_col,
        time_col=log_time_col,
        value_cols=value_cols,
        windows_hours=windows_hours,
        prefix=prefix,
    )
    return out

# ---------- CTR builders ----------
def add_ctr_features(df: pd.DataFrame, click_col: str, imp_col: str, prefix: str, alpha: float = 10.0, beta: float = 10.0) -> pd.DataFrame:
    """
    Add smoothed CTR columns based on provided click & impression sums.
    Uses Bayesian smoothing to handle low-impression scenarios robustly.
    """
    df[f"{prefix}ctr"] = bayesian_ctr(df[click_col], df[imp_col], alpha=alpha, beta=beta)
    return df

# ---------- Dataset-specific wrappers ----------
def join_user_sitewide_windows(sessions, user_site_df, session_ts_col="ts_hour", windows_days=(1,7,30)):
    hours = [24*d for d in windows_days]
    key = ["user_id_hashed"]
    vals = ["total_click","total_cart","total_fav","total_order"]
    out = time_join_sum_windows(
        sessions, user_site_df, key_cols=key, session_ts_col=session_ts_col,
        log_time_col="ts_hour", value_cols=vals, windows_hours=hours, prefix="user_"
    )
    # Example CTRs over 7d window:
    if f"user_total_click_sum_{24*7}h" in out.columns and f"user_total_order_sum_{24*7}h" in out.columns:
        out = add_ctr_features(out, f"user_total_click_sum_{24*7}h", f"user_total_order_sum_{24*7}h", "user_click_over_order_")
    return out

def join_content_sitewide_windows(sessions, content_site_df, session_ts_col="ts_hour", windows_days=(1,7,30)):
    hours = [24*d for d in windows_days]
    key = ["content_id_hashed"]
    vals = ["total_click","total_cart","total_fav","total_order"]
    return time_join_sum_windows(
        sessions, content_site_df, key_cols=key, session_ts_col=session_ts_col,
        log_time_col="date", value_cols=vals, windows_hours=hours, prefix="content_"
    )

def join_term_search_windows(sessions, term_df, session_ts_col="ts_hour", windows_hours=(1,6,24, 24*7, 24*30)):
    key = ["search_term_normalized"]
    vals = ["total_search_impression","total_search_click"]
    out = time_join_sum_windows(
        sessions, term_df, key_cols=key, session_ts_col=session_ts_col,
        log_time_col="ts_hour", value_cols=vals, windows_hours=list(windows_hours), prefix="term_"
    )
    # Smoothed CTR over 24h and 7d if present
    for w in (24, 24*7):
        imp_col = f"term_total_search_impression_sum_{w}h"
        clk_col = f"term_total_search_click_sum_{w}h"
        if imp_col in out.columns and clk_col in out.columns:
            out = add_ctr_features(out, clk_col, imp_col, prefix=f"term_ctr_{w}h_")
    return out

def join_user_top_terms_windows(sessions, user_top_df, session_ts_col="ts_hour", windows_days=(1,7,30)):
    hours = [24*d for d in windows_days]
    key = ["user_id_hashed","search_term_normalized"]
    vals = ["total_search_impression","total_search_click"]
    return time_join_sum_windows(
        sessions, user_top_df, key_cols=key, session_ts_col=session_ts_col,
        log_time_col="ts_hour", value_cols=vals, windows_hours=hours, prefix="user_term_"
    )

def join_content_top_terms_windows(sessions, content_top_df, session_ts_col="ts_hour", windows_days=(1,7,30)):
    hours = [24*d for d in windows_days]
    key = ["content_id_hashed","search_term_normalized"]
    vals = ["total_search_impression","total_search_click"]
    return time_join_sum_windows(
        sessions, content_top_df, key_cols=key, session_ts_col=session_ts_col,
        log_time_col="date", value_cols=vals, windows_hours=hours, prefix="content_term_"
    )

def join_user_fashion_sitewide_windows(sessions, uf_site_df, session_ts_col="ts_hour", windows_days=(1,7,30)):
    hours = [24*d for d in windows_days]
    key = ["user_id_hashed","content_id_hashed"]
    vals = ["total_click","total_cart","total_fav","total_order"]
    return time_join_sum_windows(
        sessions, uf_site_df, key_cols=key, session_ts_col=session_ts_col,
        log_time_col="ts_hour", value_cols=vals, windows_hours=hours, prefix="uf_"
    )

def join_user_fashion_search_windows(sessions, uf_search_df, session_ts_col="ts_hour", windows_days=(1,7,30)):
    hours = [24*d for d in windows_days]
    key = ["user_id_hashed","content_id_hashed"]
    vals = ["total_search_impression","total_search_click"]
    return time_join_sum_windows(
        sessions, uf_search_df, key_cols=key, session_ts_col=session_ts_col,
        log_time_col="ts_hour", value_cols=vals, windows_hours=hours, prefix="ufs_"
    )
