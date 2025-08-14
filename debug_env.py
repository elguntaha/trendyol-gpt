import os
import sys
sys.path.append('.')

# Set environment variable
os.environ['TRENDYOL_blending__alpha'] = '0.6'

print("Environment variables set:")
for k, v in os.environ.items():
    if 'TRENDYOL' in k:
        print(f"  {k} = {v}")

# Test the merge function directly
from src.config import _merge_env_overrides
test_dict = {'blending': {'alpha': 0.7}}
print(f"\nBefore env merge: {test_dict}")
_merge_env_overrides(test_dict)
print(f"After env merge: {test_dict}")

# Load full config
from src.config import load_config
cfg = load_config()
print(f"\nFinal config blending.alpha: {cfg.blending.alpha}")
