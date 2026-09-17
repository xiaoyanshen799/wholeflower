"""External CPU warm-up controller. No experiment is started without --execute."""

import argparse
import json
import signal
from pathlib import Path

from warmup.config import prepare_config
from warmup.controller import Calibration, external_stage
from warmup.export import export_last
from warmup.simulation import simulated_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Run real local training under systemd CPU quotas")
    mode.add_argument("--simulate", action="store_true", help="Test with synthetic timings; never launches clients")
    mode.add_argument("--export-last", type=Path, metavar="OUTPUT_DIR",
                      help="Export the last complete validation batch without training or refitting")
    parser.add_argument("--resume", action="store_true", help="Reuse completed batches with identical config")
    args = parser.parse_args()
    if args.export_last and (args.config or args.resume):
        parser.error("--export-last uses the saved configuration; do not pass --config or --resume")
    if not args.export_last and not args.config:
        parser.error("--config is required unless using --export-last")
    if args.resume and not (args.execute or args.simulate):
        parser.error("--resume requires --execute or --simulate")
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("Calibration interrupted; completed batches can be resumed")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.export_last:
            rows = export_last(args.export_last)
            print(json.dumps({"output_dir": str(args.export_last.resolve()), "clients": rows}, indent=2))
            return 0 if all(row["converged"] for row in rows) else 2
        config = prepare_config(json.loads(args.config.read_text()), Path(__file__).parent, args.simulate)
        if args.simulate:
            config["output_dir"] += ".simulation"
        if not (args.execute or args.simulate):
            print(json.dumps({"mode": "dry-run", "config": config,
                              "scan_cpu_fractions": [0.3, 0.5, 0.7, 0.9],
                              "note": "No clients started. Use --execute only when ready."}, indent=2))
            return 0
        rows = Calibration(config, simulated_stage if args.simulate else external_stage).run(args.resume)
        print(json.dumps({"output_dir": config["output_dir"], "simulation": args.simulate,
                          "clients": rows}, indent=2))
        return 0 if all(row["converged"] for row in rows) else 2
    except KeyboardInterrupt as error:
        print(str(error))
        return 130
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Calibration failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
