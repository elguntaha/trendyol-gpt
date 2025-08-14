from __future__ import annotations
import os, re, sys, argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml

# ---------- Dataclasses ----------
@dataclass
class PathsCfg:
    data_dir: str
    submissions_dir: str
    runs_dir: str
    artifacts_dir: str

@dataclass
class CVCfg:
    type: str
    n_splits: int
    purge_hours: int
    shuffle: bool
    group_by: str

@dataclass
class TargetEncodingCfg:
    enabled: bool
    kfold: int
    smoothing_alpha: float
    cols: List[str] = field(default_factory=list)

@dataclass
class TextCfg:
    cv_tags_hashing_dim: int
    keep_top_terms: int

@dataclass
class FeaturesCfg:
    hours_windows: List[int]
    days_windows: List[int]
    min_support: int
    rare_category_threshold: int
    target_encoding: TargetEncodingCfg
    text: TextCfg

@dataclass
class ModelCfg:
    framework: str
    params: Dict[str, Any]
    early_stopping_rounds: int
    class_weight: Optional[Union[str, Dict[str, float]]] = None
    use_p_click_feature: Optional[bool] = None  # only for order

@dataclass
class CalibrationCfg:
    enabled: bool
    method: str
    min_samples: int

@dataclass
class BlendingCfg:
    alpha: float
    sweep: List[float]

@dataclass
class InferenceCfg:
    batch_size: int
    num_threads: int
    output_csv: str

@dataclass
class LoggingCfg:
    level: str
    save_config_copy: bool
    run_name_prefix: str

@dataclass
class Config:
    seed: int
    paths: PathsCfg
    cv: CVCfg
    features: FeaturesCfg
    models: Dict[str, ModelCfg]   # keys: "click", "order"
    calibration: CalibrationCfg
    blending: BlendingCfg
    inference: InferenceCfg
    logging: LoggingCfg

# ---------- Utilities ----------
_BOOLS = {"true": True, "false": False}

def _auto_cast(value: str) -> Any:
    """Best-effort string -> python type casting for overrides."""
    v = value.strip()
    if v.lower() in _BOOLS: return _BOOLS[v.lower()]
    if re.fullmatch(r"-?\d+", v): return int(v)
    if re.fullmatch(r"-?\d+\.\d*", v): return float(v)
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner: return []
        return [_auto_cast(x) for x in inner.split(",")]
    return v

def _set_nested(d: Dict[str, Any], dotted_key: str, value: Any):
    """Set nested dict key given a dotted path e.g., 'cv.n_splits'."""
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def _merge_env_overrides(base: Dict[str, Any], prefix: str = "TRENDYOL_") -> None:
    # env style: TRENDYOL_cv__n_splits=7 (double underscore becomes dot)
    for k, v in os.environ.items():
        if not k.startswith(prefix): continue
        key = k[len(prefix):].replace("__", ".").lower()  # lowercase for case-insensitive
        _set_nested(base, key, _auto_cast(v))

def _merge_cli_overrides(base: Dict[str, Any], pairs: List[str]) -> None:
    # pairs like ["cv.n_splits=7", "blending.alpha=0.75"]
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid override (missing '='): {pair}")
        key, val = pair.split("=", 1)
        _set_nested(base, key.strip(), _auto_cast(val))

def _validate(cfg: Dict[str, Any]) -> None:
    # Minimal structural & range validation. Add more as needed.
    assert cfg["cv"]["n_splits"] >= 2
    assert cfg["cv"]["purge_hours"] >= 0
    assert 0.0 <= cfg["blending"]["alpha"] <= 1.0
    assert len(cfg["features"]["hours_windows"]) > 0
    assert len(cfg["features"]["days_windows"]) > 0
    for w in cfg["features"]["hours_windows"]:
        assert isinstance(w, int) and w > 0
    for w in cfg["features"]["days_windows"]:
        assert isinstance(w, int) and w > 0
    # models presence
    for key in ("click", "order"):
        assert key in cfg["models"], f"Missing model section: {key}"

def _ensure_dirs(cfg: Dict[str, Any]) -> None:
    for p in ("data_dir", "submissions_dir", "runs_dir", "artifacts_dir"):
        Path(cfg["paths"][p]).mkdir(parents=True, exist_ok=True)

def dict_to_dataclass(d: Dict[str, Any]) -> Config:
    return Config(
        seed=d["seed"],
        paths=PathsCfg(**d["paths"]),
        cv=CVCfg(**d["cv"]),
        features=FeaturesCfg(
            hours_windows=d["features"]["hours_windows"],
            days_windows=d["features"]["days_windows"],
            min_support=d["features"]["min_support"],
            rare_category_threshold=d["features"]["rare_category_threshold"],
            target_encoding=TargetEncodingCfg(**d["features"]["target_encoding"]),
            text=TextCfg(**d["features"]["text"]),
        ),
        models={k: ModelCfg(**v) for k, v in d["models"].items()},
        calibration=CalibrationCfg(**d["calibration"]),
        blending=BlendingCfg(**d["blending"]),
        inference=InferenceCfg(**d["inference"]),
        logging=LoggingCfg(**d["logging"]),
    )

def load_config(
    path: Union[str, Path] = "configs/params.yaml",
    overrides: Optional[List[str]] = None,
    apply_env: bool = True,
    ensure_dirs: bool = True,
) -> Config:
    """Load YAML, apply env/CLI overrides, validate, ensure dirs, return Config."""
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)
    if apply_env:
        _merge_env_overrides(raw)
    if overrides:
        _merge_cli_overrides(raw, overrides)
    _validate(raw)
    if ensure_dirs:
        _ensure_dirs(raw)
    return dict_to_dataclass(raw)

# Optional CLI to inspect:
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Config loader")
    ap.add_argument("--config", default="configs/params.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="Dot overrides: key=val")
    ap.add_argument("--print", action="store_true", help="Pretty-print config and exit")
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    cfg = load_config(args.config, overrides=args.set)
    if args.print:
        import json
        print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
