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
