"""Compatibility entry point for the existing root reproduce_numerics.py script."""
from pathlib import Path
import runpy


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(root / "reproduce_numerics.py"), run_name="__main__")


if __name__ == "__main__":
    main()
