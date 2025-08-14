import os
import sys
sys.path.append('.')

# Test the environment variable parsing
os.environ['TRENDYOL_blending.alpha'] = '0.6'

# Print all environment variables starting with TRENDYOL_
print("Environment variables:")
for k, v in os.environ.items():
    if k.startswith('TRENDYOL_'):
        print(f"  {k} = {v}")

from src.config import _merge_env_overrides
test_dict = {'blending': {'alpha': 0.7}}
print(f"Before: {test_dict}")
_merge_env_overrides(test_dict)
print(f"After: {test_dict}")

# Try with the corrected environment
from src.config import load_config
cfg = load_config()
print(f"Final config blending alpha: {cfg.blending.alpha}")
