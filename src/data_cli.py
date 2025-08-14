# src/data_cli.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.config import load_config
from src.data_io import load_table, build_time_range_filter, memory_mb

def main():
    ap = argparse.ArgumentParser(description="Ad-hoc data loader for Trendyol datasets")
    ap.add_argument("--key", required=True, help="dataset key, e.g. train_sessions, content_metadata, user_sitewide_log")
    ap.add_argument("--columns", nargs="*", default=None, help="subset of columns to read")
    ap.add_argument("--datecol", default=None, help="name of datetime column for filtering (e.g., ts_hour, date)")
    ap.add_argument("--start", default=None, help="start datetime (inclusive)")
    ap.add_argument("--end", default=None, help="end datetime (inclusive)")
    ap.add_argument("--categorical", action="store_true", help="attempt to convert strings to categorical")
    ap.add_argument("--config", default="configs/params.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    filt = build_time_range_filter(args.datecol, args.start, args.end) if args.datecol else None
    df = load_table(args.key, cfg.paths.data_dir, columns=args.columns, row_filter=filt, make_categorical=args.categorical)
    print(f"Rows: {len(df):,} | Cols: {len(df.columns)} | Memory: {memory_mb(df):.2f} MB")
    print(df.head(5))

if __name__ == "__main__":
    main()
