#!/usr/bin/env python3
"""Local setup checks that do not require internet access."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REQUIRED_MODULES = ["fastapi", "uvicorn", "PIL", "numpy", "torch", "torchvision"]
REQUIRED_PATHS = [Path("app.py"), Path("src"), Path("assets/demo_images"), Path("assets/fallback")]


def main() -> int:
    missing_modules = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    missing_paths = [str(path) for path in REQUIRED_PATHS if not path.exists()]

    if missing_modules:
        print("Missing Python modules:", ", ".join(missing_modules))
    if missing_paths:
        print("Missing project paths:", ", ".join(missing_paths))

    if missing_modules or missing_paths:
        print("Setup check failed. Install requirements and confirm the project structure.")
        return 1

    print("Setup check passed for the FastAPI AlexNet demo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
