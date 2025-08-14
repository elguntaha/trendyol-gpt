# Trendyol Datathon — Kaggle Phase (Ranking Click & Order)

## What this repo contains
- Reproducible environment (`.venv`, `requirements.txt`)
- VSCode setup (`.vscode/settings.json`)
- Standard structure: `src/`, `configs/`, `notebooks/`, `data/`, `submissions/`, `tests/`, `runs/`

## Quickstart
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python -m ipykernel install --user --name trendyol
```

## Project layout
```bash
configs/        # params.yaml etc.
data/           # parquet files (ignored), place datasets here
notebooks/      # 00_eda.ipynb, 01_baseline.ipynb
runs/           # experiment logs (ignored)
src/            # code modules
submissions/    # generated CSVs (ignored)
tests/          # smoke & format tests
```

## Data loading (schema-aware)

```bash
# Preview 5 rows of any parquet (no schema):
python -c "from src.data_io import head; print(head('data/train_sessions.parquet'))"

# Load with schema, columns, and date filter:
python -m src.data_cli --key train_sessions \
  --columns ts_hour session_id clicked \
  --datecol ts_hour --start 2024-01-01 --end 2024-02-01
```

Programmatically:

```python
from src.config import load_config
from src.data_io import load_train_sessions, build_time_range_filter
cfg = load_config("configs/params.yaml")
filt = build_time_range_filter("ts_hour", "2024-01-01", "2024-02-01")
df = load_train_sessions(cfg.paths.data_dir, columns=["ts_hour","session_id","clicked"], row_filter=filt)
print(df.dtypes, df.head())
```

## EDA & Data Health

- Open the EDA notebook:
  - VSCode: **Run Task → EDA: Open 00_eda.ipynb**
- Healthcheck (console):
  ```bash
  python -m src.healthcheck
  ```
Figures from the notebook are saved under runs/eda/images/.

## Build time-safe features

```bash
# Materialize chunked features (train/test) under runs/artifacts/
python -m src.build_datasets
```

VSCode: **Run Task → Build: Time-safe features**
