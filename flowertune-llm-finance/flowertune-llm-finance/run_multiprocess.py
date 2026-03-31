#!/usr/bin/env python3
"""Run the Finance Flower app with one process per client."""

from __future__ import annotations

from pathlib import Path
import sys


def _load_launcher() -> object:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "tools" / "flower_deployment_launcher.py"
        if candidate.exists():
            sys.path.insert(0, str(candidate.parent))
            from flower_deployment_launcher import main

            return main
    raise SystemExit("Could not locate tools/flower_deployment_launcher.py")


if __name__ == "__main__":
    launcher_main = _load_launcher()
    raise SystemExit(
        launcher_main(
            default_app_dir=Path(__file__).resolve().parent,
            default_num_clients=50,
        )
    )
