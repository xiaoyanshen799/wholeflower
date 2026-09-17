"""Readable per-stage CSVs derived from retained JSONL files, never replacing them."""

import json
import math
from pathlib import Path

from .measurement import atomic_json


def export_stage(run_dir, stage, config):
    from .controller import write_csv

    run_dir, stage = Path(run_dir), Path(stage)
    complete_path = stage / "complete.json"
    if not complete_path.exists():
        return None
    complete = json.loads(complete_path.read_text())
    attempt = stage / complete["attempt"]
    if not attempt.resolve().is_relative_to(stage.resolve()):
        raise ValueError("Invalid stage attempt path")
    phase = "scan" if stage.name.startswith("scan_") else "validation"
    discard = complete.get("discard", config[f"{phase}_discard"])
    rounds = complete.get("rounds", config[f"{phase}_rounds"])
    rows, sources = [], []
    for cid in config["client_ids"]:
        source = attempt / f"client_{cid}.jsonl"
        records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        recorded_rounds = [record["round"] for record in records]
        expected_rounds = range(discard + 1, rounds + 1)
        if (any(type(r) is not int or r < 1 or r > rounds for r in recorded_rounds)
                or recorded_rounds != sorted(recorded_rounds)
                or len(recorded_rounds) != len(set(recorded_rounds))
                ):
            raise ValueError(f"Duplicate, unordered or out-of-range rounds in {source}")
        missing_retained = [r for r in expected_rounds if r not in recorded_rounds]
        if missing_retained:
            raise ValueError(f"Incomplete retained rounds in {source}: missing {missing_retained}")
        for record in records:
            if record["client_id"] != cid or record["stage_id"] != complete["stage_id"]:
                raise ValueError(f"Wrong client or stage in {source}")
            duration = record["train_time_s"]
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"Invalid duration in {source}")
            rows.append({"stage_id": complete["stage_id"], "client_id": cid,
                         "server_round": record["round"], "num_examples": record.get("num_examples", ""),
                         "cpu": record["cpu_requested"], "cpu_actual": record["cpu_actual"],
                         "cpu_affinity": ",".join(map(str, record["cpu_affinity"])),
                         "client_train_s": duration, "used_for_fit": record["round"] > discard,
                         "timing_definition": record.get("timing_definition", "legacy_v1"),
                         "source_jsonl": str(source)})
            if config.get("training_mode") == "steps":
                from .speed import validate_step_record
                validate_step_record(record, config["local_steps"], config["batch_size"], config["seed"])
                metrics = record["metrics"]
                rows[-1].update({key: metrics[key] for key in (
                    "training_mode", "local_steps_requested", "local_steps_used", "batch_size_used",
                    "processed_examples", "steps_per_second", "step_seed")})
        sources.append(str(source))
    destination = run_dir / "timing_exports"
    destination.mkdir(exist_ok=True)
    write_csv(destination / f"{stage.name}.csv", rows)
    index_path = destination / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index[stage.name] = {"csv": str(destination / f"{stage.name}.csv"), "rows": len(rows),
                         "retained_rows": sum(row["used_for_fit"] for row in rows), "sources": sources,
                         "process_logs": str(attempt / "process_<cid>.log"),
                         "training_logs": str(attempt / "training_<cid>.log")}
    atomic_json(index_path, index)
    return index[stage.name]
