#!/usr/bin/env python3
"""Export Pacer JSONL and client timing JSONL into readable CSV tables."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def flatten_pacer_client(client_dir):
    rows = []
    for path in sorted(Path(client_dir).glob("pacer_client_*.jsonl")):
        for record in read_jsonl(path):
            if record.get("event") != "client_feedback":
                continue
            rows.append({
                "round": record.get("pacer.round"),
                "client_id": record.get("pacer.client_id"),
                "feedback_valid": record.get("pacer.feedback_valid"),
                "feedback_reason": record.get("pacer.feedback_reason"),
                "gamma": record.get("pacer.gamma"),
                "fit_rmse": record.get("pacer.fit_rmse"),
                "n_samples": record.get("pacer.n_samples"),
                "sample_end_round": record.get("pacer.sample_end_round"),
                "resource_status": record.get("pacer.resource_status"),
            })
    return rows


def flatten_client_timings(client_log_dir, pacer_feedback):
    feedback = {(str(row["client_id"]), int(row["round"])): row for row in pacer_feedback}
    rows = []
    for path in sorted(Path(client_log_dir).glob("client_*.jsonl")):
        for record in read_jsonl(path):
            cid = str(record.get("client_id"))
            round_id = int(record.get("round"))
            metrics = record.get("metrics") or {}
            row = {
                "round": round_id,
                "client_id": cid,
                "partition_id": metrics.get("partition_id", cid),
                "train_time_s": record.get("train_time_s"),
                "num_examples": record.get("num_examples"),
                "cpu_requested": record.get("cpu_requested"),
                "cpu_actual": record.get("cpu_actual"),
                "quota_us": record.get("quota_us"),
                "period_us": record.get("period_us"),
                "cpu_affinity": ",".join(map(str, record.get("cpu_affinity", []))),
                "cpu_time_elapsed_s": metrics.get("cpu_time_elapsed_s"),
                "sched_runqueue_time_s": metrics.get("sched_runqueue_time_s"),
                "cgroup_cpu_usage_s": metrics.get("cgroup_cpu_usage_s"),
                "cgroup_throttled_s": metrics.get("cgroup_throttled_s"),
            }
            row.update({f"pacer_{key}": value for key, value in feedback.get((cid, round_id), {}).items()
                        if key not in ("round", "client_id")})
            rows.append(row)
    return sorted(rows, key=lambda row: (int(row["round"]), int(row["client_id"])))


def atest_like_rows(client_log_dir, pacer_feedback):
    feedback = {(str(row["client_id"]), int(row["round"])): row for row in pacer_feedback}
    rows = []
    for path in sorted(Path(client_log_dir).glob("client_*.jsonl")):
        for record in read_jsonl(path):
            cid = str(record.get("client_id"))
            round_id = int(record.get("round"))
            metrics = record.get("metrics") or {}
            resource_after = record.get("resource_after") or {}
            cpu_affinity = metrics.get("cpu_affinity")
            if cpu_affinity is None:
                cpu_affinity = ",".join(map(str, record.get("cpu_affinity", [])))
            row = {
                "server_round": round_id,
                "client_id": cid,
                "num_examples": record.get("num_examples"),
                "server_to_client_ms": metrics.get("server_to_client_ms", ""),
                "server_wait_ms": metrics.get("server_wait_time", ""),
                "client_train_s": record.get("train_time_s"),
                "edcode_s": metrics.get("edcode", ""),
                "client_to_server_ms": "",
                "server_receive_time": "",
                "config_time": metrics.get("config_time", ""),
                "cpu_freq_start_mhz": metrics.get("cpu_freq_start_mhz", ""),
                "cpu_freq_end_mhz": metrics.get("cpu_freq_end_mhz", ""),
                "cpu_time_start_s": metrics.get("cpu_time_start_s", ""),
                "cpu_time_end_s": metrics.get("cpu_time_end_s", ""),
                "cpu_time_elapsed_s": metrics.get("cpu_time_elapsed_s", ""),
                "sched_run_time_ns": metrics.get("sched_run_time_ns", ""),
                "sched_runqueue_time_ns": metrics.get("sched_runqueue_time_ns", ""),
                "sched_runqueue_time_s": metrics.get("sched_runqueue_time_s", ""),
                "sched_timeslices": metrics.get("sched_timeslices", ""),
                "partition_id": metrics.get("partition_id", cid),
                "training_run_id": metrics.get("training_run_id", ""),
                "timing_definition": metrics.get("timing_definition", record.get("timing_definition", "")),
                "local_epochs_used": metrics.get("local_epochs_used", record.get("epochs", "")),
                "batch_size_used": metrics.get("batch_size_used", record.get("batch_size", "")),
                "cpu_requested": metrics.get("cpu_requested", record.get("cpu_requested", "")),
                "cpu_actual": metrics.get("cpu_actual", record.get("cpu_actual", "")),
                "cpu_affinity": cpu_affinity,
                "quota_us": metrics.get("quota_us", record.get("quota_us", "")),
                "period_us": metrics.get("period_us", record.get("period_us", "")),
                "thread_count": metrics.get("thread_count", record.get("thread_count", "")),
                "resource_verified": metrics.get("resource_verified", ""),
                "cgroup_cpu_usage_s": metrics.get("cgroup_cpu_usage_s", ""),
                "cgroup_throttled_s": metrics.get("cgroup_throttled_s", ""),
                "cgroup_periods": metrics.get("cgroup_periods", ""),
                "cgroup_throttled_periods": metrics.get("cgroup_throttled_periods", ""),
                "bound_cpu_freq_mhz": metrics.get("bound_cpu_freq_mhz", ""),
                "cgroup": record.get("cgroup", resource_after.get("cgroup", "")),
            }
            row.update({f"pacer_{key}": value for key, value in feedback.get((cid, round_id), {}).items()
                        if key not in ("round", "client_id")})
            rows.append(row)
    return sorted(rows, key=lambda row: (int(row["server_round"]), int(row["client_id"])))


def round_summary(pacer_round_log):
    rows = []
    for record in read_jsonl(pacer_round_log):
        if record.get("event") != "round_observation":
            continue
        rows.append({
            "round": record.get("server_round"),
            "state": record.get("state"),
            "theta_target_s": record.get("theta_target_s"),
            "deadline_s": record.get("deadline_s"),
            "q": record.get("q"),
            "q_lower": record.get("q_lower"),
            "q_upper": record.get("q_upper"),
            "n_valid": record.get("n_valid"),
            "n_expected": record.get("n_expected"),
            "gamma_sum": record.get("gamma_sum"),
            "p_hat": record.get("p_hat"),
            "local_train_round_s": record.get("local_train_round_s"),
            "dispatch_barrier_elapsed_s": record.get("dispatch_barrier_elapsed_s"),
            "observed_deadline_met": record.get("observed_deadline_met"),
            "fresh_feedback": record.get("fresh_feedback"),
            "recalibration_requested": record.get("recalibration_requested"),
            "consecutive_checks": record.get("consecutive_checks"),
        })
    return rows


def percentile(values, pct):
    if not values:
        return ""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def round_train_times(client_times, rounds):
    round_meta = {int(row["round"]): row for row in rounds}
    grouped = {}
    for row in client_times:
        grouped.setdefault(int(row["round"]), []).append(float(row["train_time_s"]))
    rows = []
    for round_id in sorted(grouped):
        values = grouped[round_id]
        meta = round_meta.get(round_id, {})
        rows.append({
            "server_round": round_id,
            "n_clients": len(values),
            "client_train_min_s": min(values),
            "client_train_mean_s": statistics.fmean(values),
            "client_train_p50_s": percentile(values, 0.50),
            "client_train_p90_s": percentile(values, 0.90),
            "client_train_max_s": max(values),
            "local_train_round_s": meta.get("local_train_round_s", ""),
            "dispatch_barrier_elapsed_s": meta.get("dispatch_barrier_elapsed_s", ""),
            "theta_target_s": meta.get("theta_target_s", ""),
            "deadline_s": meta.get("deadline_s", ""),
            "p_hat": meta.get("p_hat", ""),
            "gamma_sum": meta.get("gamma_sum", ""),
            "state": meta.get("state", ""),
            "observed_deadline_met": meta.get("observed_deadline_met", ""),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pacer-round-log", required=True, type=Path)
    parser.add_argument("--client-log-dir", required=True, type=Path)
    parser.add_argument("--pacer-client-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    client_feedback = flatten_pacer_client(args.pacer_client_dir)
    client_times = flatten_client_timings(args.client_log_dir, client_feedback)
    atest_rows = atest_like_rows(args.client_log_dir, client_feedback)
    rounds = round_summary(args.pacer_round_log)
    round_times = round_train_times(client_times, rounds)

    write_csv(args.output_dir / "pacer_client_feedback.csv", client_feedback, [
        "round", "client_id", "feedback_valid", "feedback_reason", "gamma", "fit_rmse",
        "n_samples", "sample_end_round", "resource_status",
    ])
    write_csv(args.output_dir / "pacer_client_times.csv", client_times, [
        "round", "client_id", "partition_id", "train_time_s", "num_examples",
        "cpu_requested", "cpu_actual", "quota_us", "period_us", "cpu_affinity",
        "cpu_time_elapsed_s", "sched_runqueue_time_s", "cgroup_cpu_usage_s", "cgroup_throttled_s",
        "pacer_feedback_valid", "pacer_feedback_reason", "pacer_gamma", "pacer_fit_rmse",
        "pacer_n_samples", "pacer_sample_end_round", "pacer_resource_status",
    ])
    write_csv(args.output_dir / "pacer_atest_like.csv", atest_rows, [
        "server_round", "client_id", "num_examples", "server_to_client_ms",
        "server_wait_ms", "client_train_s", "edcode_s", "client_to_server_ms",
        "server_receive_time", "config_time", "cpu_freq_start_mhz", "cpu_freq_end_mhz",
        "cpu_time_start_s", "cpu_time_end_s", "cpu_time_elapsed_s", "sched_run_time_ns",
        "sched_runqueue_time_ns", "sched_runqueue_time_s", "sched_timeslices",
        "partition_id", "training_run_id", "timing_definition", "local_epochs_used",
        "batch_size_used", "cpu_requested", "cpu_actual", "cpu_affinity", "quota_us",
        "period_us", "thread_count", "resource_verified", "cgroup_cpu_usage_s",
        "cgroup_throttled_s", "cgroup_periods", "cgroup_throttled_periods",
        "bound_cpu_freq_mhz", "cgroup", "pacer_feedback_valid", "pacer_feedback_reason",
        "pacer_gamma", "pacer_fit_rmse", "pacer_n_samples", "pacer_sample_end_round",
        "pacer_resource_status",
    ])
    write_csv(args.output_dir / "pacer_round_summary.csv", rounds, [
        "round", "state", "theta_target_s", "deadline_s", "q", "q_lower", "q_upper",
        "n_valid", "n_expected", "gamma_sum", "p_hat", "local_train_round_s",
        "dispatch_barrier_elapsed_s", "observed_deadline_met", "fresh_feedback",
        "recalibration_requested", "consecutive_checks",
    ])
    write_csv(args.output_dir / "pacer_round_train_times.csv", round_times, [
        "server_round", "n_clients", "client_train_min_s", "client_train_mean_s",
        "client_train_p50_s", "client_train_p90_s", "client_train_max_s",
        "local_train_round_s", "dispatch_barrier_elapsed_s", "theta_target_s",
        "deadline_s", "p_hat", "gamma_sum", "state", "observed_deadline_met",
    ])
    print(args.output_dir / "pacer_round_summary.csv")
    print(args.output_dir / "pacer_client_times.csv")
    print(args.output_dir / "pacer_client_feedback.csv")
    print(args.output_dir / "pacer_atest_like.csv")
    print(args.output_dir / "pacer_round_train_times.csv")


if __name__ == "__main__":
    main()
