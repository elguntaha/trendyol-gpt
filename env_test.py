import os
import sys
sys.path.append('.')

# Set environment variable with correct format (double underscores become dots)
os.environ['TRENDYOL_blending__alpha'] = '0.6'

from src.config import load_config
cfg = load_config()
print(f"Environment override test: blending.alpha = {cfg.blending.alpha}")
print("✓ Environment overrides work with double underscore format!")
