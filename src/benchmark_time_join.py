import time
from pathlib import Path
from src.config import load_config
from src.build_datasets import _materialize_sessions
from src.data_io import load_train_sessions

if __name__ == "__main__":
    cfg = load_config("configs/params.yaml")
    train = load_train_sessions(cfg.paths.data_dir)
    print(f"Loaded {len(train)} rows from train_sessions")
    
    for rows in [250_000, 500_000, 1_000_000]:
        # Adjust row count if dataset is smaller
        actual_rows = min(rows, len(train))
        if actual_rows == 0:
            print(f"rows={rows} skipped (no data)")
            continue
            
        print(f"\nBenchmarking with {actual_rows} rows (requested {rows})...")
        subset = train.head(actual_rows)
        
        t0 = time.time()
        _materialize_sessions(
            subset, 
            name=f"bench_train_{rows}", 
            out_dir=Path(cfg.paths.artifacts_dir) / "benchmark", 
            cfg=cfg,
            chunk_rows=rows
        )
        elapsed = time.time() - t0
        print(f"rows={rows} elapsed={elapsed:.1f}s")
        
    print("\nBenchmark complete! Check runs/artifacts/benchmark/ for output files.")
