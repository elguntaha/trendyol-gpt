import os, json, random, logging
from pathlib import Path
from typing import Any, Dict
import numpy as np

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    # LightGBM/XGBoost will use their own seeds via params as well

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("trendyol")

def save_config_snapshot(cfg_dict: Dict[str, Any], runs_dir: str, run_name_prefix: str) -> str:
    run_dir = Path(runs_dir) / f"{run_name_prefix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.snapshot.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, ensure_ascii=False)
    return str(path)
