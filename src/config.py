from pathlib import Path
import fastf1

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"

fastf1.Cache.enable_cache(CACHE_DIR)