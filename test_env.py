import os
import sys
sys.path.append('.')
from src.config import load_config

# Set environment variable
os.environ['TRENDYOL_blending.alpha'] = '0.6'

# Load config and test
cfg = load_config()
print(f"Blending alpha from env override: {cfg.blending.alpha}")
