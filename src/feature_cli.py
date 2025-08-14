from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.config import load_config
from src.features import build_feature_matrix

def main():
    ap = argparse.ArgumentParser(description="Build feature matrices (train/test)")
    ap.add_argument("--config", default="configs/params.yaml")
    ap.add_argument("--train", action="store_true", help="build train matrix")
    ap.add_argument("--test", action="store_true", help="build test matrix")
    ap.add_argument("--limit-rows", type=int, default=None)
    ap.add_argument("--folds-csv", default=None, help="optional CSV with columns: session_id,fold")
    args = ap.parse_args()

    cfg = load_config(args.config)
    art = Path(cfg.paths.artifacts_dir); art.mkdir(parents=True, exist_ok=True)
    folds = None
    if args.folds_csv and Path(args.folds_csv).exists():
        ff = pd.read_csv(args.folds_csv)
        folds = ff.set_index("session_id")["fold"]
    if args.train:
        res = build_feature_matrix(cfg, is_train=True, limit_rows=args.limit_rows, folds=None if folds is None else folds.reindex(None))
        out = art / "train_matrix.parquet"
        res.X.to_parquet(out, index=False)
        print(f"[OK] Train matrix → {out} rows={len(res.X):,}")
    if args.test:
        res = build_feature_matrix(cfg, is_train=False, limit_rows=args.limit_rows, folds=None)
        out = art / "test_matrix.parquet"
        res.X.to_parquet(out, index=False)
        print(f"[OK] Test matrix  → {out} rows={len(res.X):,}")

if __name__ == "__main__":
    main()
