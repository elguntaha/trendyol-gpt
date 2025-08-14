import os
import sys
sys.path.append('.')
from src.config import load_config

# Test with underscores instead of dots (common pattern)
os.environ['TRENDYOL_blending__alpha'] = '0.6'

# Load config and test
cfg = load_config()
print(f"Blending alpha from env override: {cfg.blending.alpha}")

# Also test the environment parsing directly
from src.config import _merge_env_overrides
test_dict = {'blending': {'alpha': 0.7}}
_merge_env_overrides(test_dict)
print(f"Direct env parsing test: {test_dict['blending']['alpha']}")
