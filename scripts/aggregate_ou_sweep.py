"""Compatibility wrapper for ``python -m noise.aggregate``."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noise.aggregate import main


if __name__ == "__main__":
    main()
