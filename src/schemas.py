# src/schemas.py
"""
Canonical dataset schemas for Trendyol Datathon.
- Provides recommended pandas dtypes (memory-aware).
- Lists datetime columns that must be parsed/tz-normalized to UTC.
- Centralizes relative parquet paths (joined under cfg.paths.data_dir).
"""
from __future__ import annotations
from typing import Dict, List, Tuple

# ---- Relative paths (under data_dir) ----
PATHS: Dict[str, str] = {
    "train_sessions": "train_sessions.parquet",
    "test_sessions": "test_sessions.parquet",
    # content/*
    "content_metadata": "content/metadata.parquet",
    "content_price_rate_review": "content/price_rate_review_data.parquet",
    "content_search_log": "content/search_log.parquet",
    "content_sitewide_log": "content/sitewide_log.parquet",
    "content_top_terms_log": "content/top_terms_log.parquet",
    # user/*
    "user_metadata": "user/metadata.parquet",
    "user_search_log": "user/search_log.parquet",
    "user_sitewide_log": "user/sitewide_log.parquet",
    "user_top_terms_log": "user/top_terms_log.parquet",
    "user_fashion_search_log": "user/fashion_search_log.parquet",
    "user_fashion_sitewide_log": "user/fashion_sitewide_log.parquet",
    # term/*
    "term_search_log": "term/search_log.parquet",
}

# ---- Datetime columns per table (must parse to timezone-aware UTC) ----
DATETIME_COLS: Dict[str, List[str]] = {
    "train_sessions": ["ts_hour"],
    "test_sessions": ["ts_hour"],
    "content_metadata": ["content_creation_date"],
    "content_price_rate_review": ["update_date"],
    "content_search_log": ["date"],
    "content_sitewide_log": ["date"],
    "content_top_terms_log": ["date"],
    "user_metadata": [],  # no datetime in spec
    "user_search_log": ["ts_hour"],
    "user_sitewide_log": ["ts_hour"],
    "user_top_terms_log": ["ts_hour"],
    "user_fashion_search_log": ["ts_hour"],
    "user_fashion_sitewide_log": ["ts_hour"],
    "term_search_log": ["ts_hour"],
}

# ---- Recommended pandas dtypes (memory-conscious) ----
# Note: int flags (0/1) -> 'int8'; scaled floats -> 'float32'; categories as 'string' by default
# (switchable to 'category' at load-time if desired).
DTYPES: Dict[str, Dict[str, str]] = {
    "train_sessions": {
        "search_term_normalized": "string",
        "clicked": "int8",
        "ordered": "int8",
        "added_to_cart": "int8",
        "added_to_fav": "int8",
        "user_id_hashed": "string",
        "content_id_hashed": "string",
        "session_id": "string",
    },
    "test_sessions": {
        "search_term_normalized": "string",
        "user_id_hashed": "string",
        "content_id_hashed": "string",
        "session_id": "string",
    },
    "content_metadata": {
        "level1_category_name": "string",
        "level2_category_name": "string",
        "leaf_category_name": "string",
        "attribute_type_count": "float32",
        "total_attribute_option_count": "float32",
        "merchant_count": "float32",
        "filterable_label_count": "float32",
        "cv_tags": "string",
        "content_id_hashed": "string",
    },
    "content_price_rate_review": {
        "original_price": "float32",
        "selling_price": "float32",
        "discounted_price": "float32",
        "content_review_count": "float32",
        "content_review_wth_media_count": "float32",
        "content_rate_count": "float32",
        "content_rate_avg": "float32",
        "content_id_hashed": "string",
    },
    "content_search_log": {
        "total_search_impression": "float32",
        "total_search_click": "float32",
        "content_id_hashed": "string",
    },
    "content_sitewide_log": {
        "total_click": "float32",
        "total_cart": "float32",
        "total_fav": "float32",
        "total_order": "float32",
        "content_id_hashed": "string",
    },
    "content_top_terms_log": {
        "search_term_normalized": "string",
        "total_search_impression": "float32",
        "total_search_click": "float32",
        "content_id_hashed": "string",
    },
    "user_metadata": {
        "user_gender": "string",
        "user_birth_year": "float32",
        "user_tenure_in_days": "int32",
        "user_id_hashed": "string",
    },
    "user_search_log": {
        "total_search_impression": "float32",
        "total_search_click": "float32",
        "user_id_hashed": "string",
    },
    "user_sitewide_log": {
        "total_click": "float32",
        "total_cart": "float32",
        "total_fav": "float32",
        "total_order": "float32",
        "user_id_hashed": "string",
    },
    "user_top_terms_log": {
        "search_term_normalized": "string",
        "total_search_impression": "float32",
        "total_search_click": "float32",
        "user_id_hashed": "string",
    },
    "user_fashion_search_log": {
        "total_search_impression": "float32",
        "total_search_click": "float32",
        "user_id_hashed": "string",
        "content_id_hashed": "string",
    },
    "user_fashion_sitewide_log": {
        "total_click": "float32",
        "total_cart": "float32",
        "total_fav": "float32",
        "total_order": "float32",
        "user_id_hashed": "string",
        "content_id_hashed": "string",
    },
    "term_search_log": {
        "search_term_normalized": "string",
        "total_search_impression": "float32",
        "total_search_click": "float32",
    },
}

def get_schema(key: str) -> Tuple[str, Dict[str, str], list[str]]:
    """Return (relative_path, dtypes, datetime_cols) for a known dataset key."""
    if key not in PATHS:
        raise KeyError(f"Unknown dataset key: {key}")
    return PATHS[key], DTYPES.get(key, {}), DATETIME_COLS.get(key, [])
