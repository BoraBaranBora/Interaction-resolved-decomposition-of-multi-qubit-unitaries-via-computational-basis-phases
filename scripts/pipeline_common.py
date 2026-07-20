from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]


def run(args: Iterable[str], *, cwd: Path = ROOT) -> None:
    command = [str(item) for item in args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run_python(*args: str) -> None:
    run([sys.executable, *args])


def run_module(module: str, *args: str) -> None:
    run_python("-m", module, *args)


def require(paths: Iterable[Path | str]) -> None:
    missing = []
    for value in paths:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
