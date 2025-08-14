from pathlib import Path
import pandas as pd
from src.config import load_config
from src.schemas import PATHS
from src.data_io import load_train_sessions, load_test_sessions, memory_mb

def main():
    try:
        cfg = load_config("configs/params.yaml")
        data_dir = Path(cfg.paths.data_dir)
        print(f"Data dir: {data_dir.resolve()}")
        # Presence
        for k, rel in PATHS.items():
            p = data_dir / rel
            print(f"{k:28s} | exists={p.exists()} | path={p}")
        # Minimal stats
        if (data_dir / PATHS["train_sessions"]).exists():
            try:
                # Load minimal set of columns that should exist
                train = load_train_sessions(data_dir, columns=["ts_hour","session_id","clicked"])
                print(f"train: shape={train.shape}, mem={memory_mb(train):.2f} MB, ts_hour[{train.ts_hour.min()}..{train.ts_hour.max()}]")
                sess = train.groupby("session_id").agg(any_click=("clicked","max")).reset_index()
                click_rate = train["clicked"].mean()
                print(f"sessions: total={len(sess):,}, with_click={sess.any_click.mean():.4f}, click_rate={click_rate:.4f}")
            except Exception as e:
                print(f"[ERROR] loading train_sessions: {e}")
        if (data_dir / PATHS["test_sessions"]).exists():
            try:
                test = load_test_sessions(data_dir, columns=["ts_hour","session_id"])
                print(f"test: shape={test.shape}, mem={memory_mb(test):.2f} MB, ts_hour[{test.ts_hour.min()}..{test.ts_hour.max()}]")
            except Exception as e:
                print(f"[ERROR] loading test_sessions: {e}")
    except Exception as e:
        print(f"[ERROR] in healthcheck: {e}")
        # Still exit with 0 as per requirements

if __name__ == "__main__":
    main()
