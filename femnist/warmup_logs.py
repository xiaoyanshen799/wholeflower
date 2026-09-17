"""Export readable CSVs from an existing warm-up run without training."""

import argparse
import fcntl
import json
from pathlib import Path

from warmup.logs import export_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    cfg = json.loads((root / "manifest.json").read_text())["config"]
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for stage in sorted((root / "stages").iterdir()):
            if stage.is_dir() and stage.name.startswith(("scan_", "validate_")):
                exported = export_stage(root, stage, cfg)
                if exported:
                    print(f"{exported['csv']}: {exported['rows']} rows; raw logs retained")


if __name__ == "__main__":
    main()
