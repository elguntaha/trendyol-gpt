from __future__ import annotations
import math
from pathlib import Path
import pandas as pd
from src.config import load_config
from src.data_io import (
    load_train_sessions, load_test_sessions,
    load_user_sitewide_log, load_content_sitewide_log, load_term_search_log,
)
from src.time_join import (
    join_user_sitewide_windows, join_content_sitewide_windows, join_term_search_windows,
)

def _materialize_sessions(df: pd.DataFrame, name: str, out_dir: Path, cfg, chunk_rows: int = 1_000_000) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Sort by time for as-of joins
    df = df.sort_values("ts_hour").reset_index(drop=True)
    n = len(df)
    chunks = math.ceil(n / chunk_rows)
    parts = []
    for i in range(chunks):
        lo, hi = i*chunk_rows, min((i+1)*chunk_rows, n)
        part = df.iloc[lo:hi].copy()
        # Join a minimal but representative subset of features (extend later)
        # User sitewide
        if "user_id_hashed" in part.columns:
            try:
                user_site = load_user_sitewide_log(cfg.paths.data_dir, columns=["ts_hour","total_click","total_cart","total_fav","total_order","user_id_hashed"])
                part = join_user_sitewide_windows(part, user_site, windows_days=(1,7,30))
            except Exception as e:
                print("[SKIP] user sitewide join:", e)
        # Content sitewide
        if "content_id_hashed" in part.columns:
            try:
                content_site = load_content_sitewide_log(cfg.paths.data_dir, columns=["date","total_click","total_cart","total_fav","total_order","content_id_hashed"])
                part = join_content_sitewide_windows(part, content_site, windows_days=(1,7,30))
            except Exception as e:
                print("[SKIP] content sitewide join:", e)
        # Term search
        if "search_term_normalized" in part.columns:
            try:
                term = load_term_search_log(cfg.paths.data_dir)
                part = join_term_search_windows(part, term, windows_hours=(1,6,24, 24*7, 24*30))
            except Exception as e:
                print("[SKIP] term search join:", e)
        out_path = out_dir / f"{name}.part{i:03d}.parquet"
        part.to_parquet(out_path, index=False)
        parts.append(out_path)
        print(f"[OK] Wrote {out_path} rows={len(part)}")
    # Optionally: concat parts in downstream step

if __name__ == "__main__":
    cfg = load_config("configs/params.yaml")
    art_dir = Path(cfg.paths.artifacts_dir)
    # Train
    try:
        train = load_train_sessions(cfg.paths.data_dir)
        _materialize_sessions(train, "train_features", art_dir, cfg)
    except Exception as e:
        print("[WARN] Train features skipped:", e)
    # Test
    try:
        test = load_test_sessions(cfg.paths.data_dir)
        _materialize_sessions(test, "test_features", art_dir, cfg)
    except Exception as e:
        print("[WARN] Test features skipped:", e)
