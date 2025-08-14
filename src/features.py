# src/features.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np, pandas as pd

from src.config import load_config, Config
from src.data_io import (
    load_train_sessions, load_test_sessions,
    load_content_metadata, load_content_price_rate_review,
    load_user_metadata,
)
from src.time_join import (
    join_user_sitewide_windows, join_content_sitewide_windows, join_term_search_windows,
    asof_join_latest, to_utc_hour
)
from src.encoders import RareCategoryGrouper, KFoldTargetEncoder
from src.text_feats import hashed_bow, term_basic_features

# ---------- Base helpers ----------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = to_utc_hour(df["ts_hour"])
    df = df.copy()
    df["hour"] = ts.dt.hour.astype("int16")
    df["dow"] = ts.dt.dayofweek.astype("int16")
    df["is_weekend"] = df["dow"].isin([5,6]).astype("int8")
    return df

def merge_user_meta(df: pd.DataFrame, user_meta: pd.DataFrame) -> pd.DataFrame:
    if user_meta is None: return df
    out = df.merge(user_meta, on="user_id_hashed", how="left")
    # derive age (approx; year only)
    if "user_birth_year" in out.columns:
        out["user_age_approx"] = (pd.Timestamp("now", tz="UTC").year - out["user_birth_year"]).astype("float32")
    return out

def merge_content_meta(df: pd.DataFrame, content_meta: pd.DataFrame, rare_threshold: int) -> pd.DataFrame:
    if content_meta is None: return df
    out = df.merge(content_meta, on="content_id_hashed", how="left")
    # rare grouping for category columns
    for col in ["level1_category_name","level2_category_name","leaf_category_name"]:
        if col in out.columns:
            out[col] = RareCategoryGrouper(min_count=rare_threshold).fit_transform(out[col].astype("string"))
    return out

def asof_latest_content_stats(sessions: pd.DataFrame, prr: pd.DataFrame) -> pd.DataFrame:
    """Bring latest <= ts_hour price/rating snapshot per content."""
    if prr is None or "content_id_hashed" not in sessions.columns:
        return sessions
    
    # Simple approach: loop through each content and do asof join
    result_list = []
    
    for content_id in sessions["content_id_hashed"].unique():
        sess_subset = sessions[sessions["content_id_hashed"] == content_id].copy()
        prr_subset = prr[prr["content_id_hashed"] == content_id].copy()
        
        if len(prr_subset) == 0:
            # No price data for this content
            result_list.append(sess_subset)
            continue
            
        # Sort both by time
        sess_subset = sess_subset.sort_values("ts_hour")
        prr_subset = prr_subset.sort_values("update_date")
        
        # Drop content_id_hashed from right to avoid duplicate columns
        prr_subset_clean = prr_subset.drop(columns=["content_id_hashed"])
        
        # Use merge_asof without 'by' parameter since we're already filtering
        merged = pd.merge_asof(
            sess_subset, prr_subset_clean,
            left_on="ts_hour", right_on="update_date",
            direction="backward", allow_exact_matches=True
        )
        result_list.append(merged)
    
    if not result_list:
        return sessions
        
    combined = pd.concat(result_list, ignore_index=True)
    
    # Add 'latest_' prefix to price/rating columns
    cols = ["original_price","selling_price","discounted_price","content_rate_avg","content_rate_count"]
    existing_cols = [c for c in cols if c in combined.columns]
    
    for col in existing_cols:
        combined[f"latest_{col}"] = combined[col]
        combined.drop(columns=[col], inplace=True)
    
    # Drop update_date column
    if "update_date" in combined.columns:
        combined.drop(columns=["update_date"], inplace=True)
        
    return combined

# ---------- Target Encoding orchestrator ----------
def apply_target_encoding(
    df_train: pd.DataFrame,
    df_test: Optional[pd.DataFrame],
    folds: Optional[pd.Series],
    cols: Sequence[str],
    y_click: pd.Series,
    y_order: pd.Series,
    smoothing_alpha: float,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if folds is None or len(cols) == 0:
        return df_train, df_test
    # Encode separately for click and order, suffixing names
    for tgt_name, y in [("click", y_click), ("order", y_order)]:
        enc = KFoldTargetEncoder(cols=cols, kfold=int(folds.max()+1) if folds is not None else 5, smoothing_alpha=smoothing_alpha)
        oof = enc.fit_transform(df_train[cols], y, folds=folds)
        df_train[f"te_{tgt_name}_{'_'.join(cols)}"] = oof
        if df_test is not None:
            df_test[f"te_{tgt_name}_{'_'.join(cols)}"] = enc.transform(df_test[cols])
    return df_train, df_test

# ---------- Master builder ----------
@dataclass
class FeatureBuildResult:
    X: pd.DataFrame
    y_click: Optional[pd.Series] = None
    y_order: Optional[pd.Series] = None
    meta_cols: Optional[List[str]] = None

def build_feature_matrix(
    cfg: Config,
    is_train: bool = True,
    limit_rows: Optional[int] = None,
    folds: Optional[pd.Series] = None,
) -> FeatureBuildResult:
    # 0) Load core frames
    sessions = load_train_sessions(cfg.paths.data_dir) if is_train else load_test_sessions(cfg.paths.data_dir)
    if limit_rows: sessions = sessions.head(limit_rows).copy()
    sessions = add_time_features(sessions)
    # 1) Basic text features on search term
    txt = term_basic_features(sessions["search_term_normalized"])
    X = pd.concat([sessions, txt], axis=1)
    # 2) Join logs (time-safe windows)
    # NOTE: extend with user/content/term wrappers as needed; here are the key ones:
    try:
        from src.data_io import load_user_sitewide_log, load_content_sitewide_log, load_term_search_log
        user_site = load_user_sitewide_log(cfg.paths.data_dir, columns=["ts_hour","total_click","total_cart","total_fav","total_order","user_id_hashed"])
        X = join_user_sitewide_windows(X, user_site, windows_days=cfg.features.days_windows)
    except Exception as e:
        print("[WARN] user_sitewide join skipped:", e)
    try:
        content_site = load_content_sitewide_log(cfg.paths.data_dir, columns=["date","total_click","total_cart","total_fav","total_order","content_id_hashed"])
        X = join_content_sitewide_windows(X, content_site, windows_days=cfg.features.days_windows)
    except Exception as e:
        print("[WARN] content_sitewide join skipped:", e)
    try:
        term_log = load_term_search_log(cfg.paths.data_dir)
        hours = cfg.features.hours_windows + [24*d for d in cfg.features.days_windows]
        X = join_term_search_windows(X, term_log, windows_hours=hours)
    except Exception as e:
        print("[WARN] term_search join skipped:", e)
    # 3) Merge metadata & latest price/rating snapshot (≤ ts_hour)
    try:
        user_meta = load_user_metadata(cfg.paths.data_dir)
        X = merge_user_meta(X, user_meta)
    except Exception as e:
        print("[WARN] user_meta merge skipped:", e)
    try:
        content_meta = load_content_metadata(cfg.paths.data_dir)
        X = merge_content_meta(X, content_meta, cfg.features.rare_category_threshold)
    except Exception as e:
        print("[WARN] content_meta merge skipped:", e)
    try:
        prr = load_content_price_rate_review(cfg.paths.data_dir)
        X = asof_latest_content_stats(X, prr)
    except Exception as e:
        print("[WARN] price/rating asof skipped:", e)
    # 4) cv_tags hashing (optional, can be heavy)
    dim = int(cfg.features.text.cv_tags_hashing_dim)
    if "cv_tags" in X.columns and dim > 0:
        bow = hashed_bow(X["cv_tags"], dim=dim, name="cvhash")
        X = pd.concat([X, bow], axis=1)
    # 5) Labels (train only)
    y_click = y_order = None
    if is_train:
        y_click = X["clicked"].astype("int8") if "clicked" in X.columns else None
        y_order = X["ordered"].astype("int8") if "ordered" in X.columns else None
    # 6) Target encoding (OOF) on selected categorical columns
    if cfg.features.target_encoding.enabled:
        te_cols = [c for c in ["level1_category_name","level2_category_name","leaf_category_name","search_term_normalized"] if c in X.columns]
        if len(te_cols) > 0 and is_train and y_click is not None and y_order is not None:
            X, _ = apply_target_encoding(
                df_train=X, df_test=None, folds=folds, cols=te_cols,
                y_click=y_click, y_order=y_order, smoothing_alpha=cfg.features.target_encoding.smoothing_alpha
            )
    # 7) Collect meta/id columns & cast types
    available_meta_cols = [c for c in ["session_id","content_id_hashed","user_id_hashed","ts_hour"] if c in X.columns]
    meta_cols = available_meta_cols
    # Return only numeric + encoded features for modeling; keep meta & labels separately
    drop_cols = set(["clicked","ordered","added_to_cart","added_to_fav"])  # labels & auxiliaries
    # Keep string/categorical cols only if TE not enabled
    for c in list(X.columns):
        if c in meta_cols or c in drop_cols: continue
        if pd.api.types.is_string_dtype(X[c]) or str(X[c].dtype) == 'category':
            X.drop(columns=[c], inplace=True, errors="ignore")
    return FeatureBuildResult(X=X, y_click=y_click, y_order=y_order, meta_cols=meta_cols)
