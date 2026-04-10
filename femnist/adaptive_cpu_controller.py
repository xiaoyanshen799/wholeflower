#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from recompute_theta_cpu_targets import fit_cpu_model, fit_theta_from_run, normalize_cpu_series


@dataclass
class LivePoint:
    cpu: float
    theta: float
    k: float
    source: str


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor a running comm-times CSV, fit theta from each client's latest "
            "window, and optionally update systemd CPUQuota online when theta drifts "
            "too far from the target."
        )
    )
    parser.add_argument("--run-csv", type=Path, required=True, help="Live comm-times CSV being appended to.")
    parser.add_argument(
        "--cpu-map-csv",
        type=Path,
        required=True,
        help="Current CPU map CSV used to launch clients.",
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=Path("/home/xiaoyan/wholeflower/femnist/mnist_cpu_theta.csv"),
        help="Historical cpu-theta CSV with columns id,theta,k,cpu.",
    )
    parser.add_argument("--target-theta", type=float, required=True, help="Theta target to hold online.")
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Use the latest N durations per client when fitting the current theta.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=8,
        help="Minimum latest-window sample count required before fitting a client.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Relative theta tolerance. 0.05 means +/-5%% around target_theta.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=20.0,
        help="Sleep interval between controller passes.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=120.0,
        help="Minimum time between quota updates for the same client.",
    )
    parser.add_argument("--cpu-min", type=float, default=0.30, help="Lower clamp for computed CPU share.")
    parser.add_argument("--cpu-max", type=float, default=1.00, help="Upper clamp for computed CPU share.")
    parser.add_argument(
        "--max-step",
        type=float,
        default=0.08,
        help="Maximum CPU share delta applied in one update. Use 0 to disable step capping.",
    )
    parser.add_argument(
        "--min-cpu-delta",
        type=float,
        default=0.01,
        help="Skip updates smaller than this absolute CPU-share delta.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=1000.0,
        help="Discard durations above this threshold before fitting.",
    )
    parser.add_argument(
        "--group-col",
        default="num_examples",
        help="Client identity column in run-csv.",
    )
    parser.add_argument(
        "--time-col",
        default="client_train_s",
        help="Duration column in run-csv.",
    )
    parser.add_argument(
        "--scope-prefix",
        default="fl_client_",
        help="systemd scope prefix used by launch_clients.sh.",
    )
    parser.add_argument(
        "--systemctl-prefix",
        default="sudo systemctl",
        help='Command prefix used to update CPUQuota, e.g. "sudo systemctl" or "systemctl".',
    )
    parser.add_argument(
        "--snapshot-csv",
        type=Path,
        default=Path("/home/xiaoyan/wholeflower/femnist/logs/adaptive_cpu_snapshot.csv"),
        help="Where to write the latest controller snapshot.",
    )
    parser.add_argument(
        "--adjust-log-csv",
        type=Path,
        default=Path("/home/xiaoyan/wholeflower/femnist/logs/adaptive_cpu_adjustments.csv"),
        help="Append-only log for quota update attempts.",
    )
    parser.add_argument(
        "--cpu-map-out",
        type=Path,
        default=Path("/home/xiaoyan/wholeflower/femnist/logs/adaptive_cpu_current_map.csv"),
        help="Write the current controller CPU map here after each pass.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually run systemctl set-property. Without this flag, the controller is observe-only.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one pass and exit.",
    )
    return parser


def load_history_points(history_csv: Path) -> pd.DataFrame:
    history_df = pd.read_csv(history_csv).copy()
    required_cols = {"id", "theta", "k", "cpu"}
    if not required_cols.issubset(history_df.columns):
        raise SystemExit(f"history-csv must contain columns: {sorted(required_cols)}")
    history_df["id"] = pd.to_numeric(history_df["id"], errors="raise").astype(int)
    history_df["theta"] = pd.to_numeric(history_df["theta"], errors="raise").astype(float)
    history_df["k"] = pd.to_numeric(history_df["k"], errors="raise").astype(float)
    history_df["cpu"] = normalize_cpu_series(pd.to_numeric(history_df["cpu"], errors="raise"))
    history_df["source"] = "history"
    return history_df[["id", "theta", "k", "cpu", "source"]]


def load_cpu_map(cpu_map_csv: Path) -> pd.DataFrame:
    cpu_map_df = pd.read_csv(cpu_map_csv).copy()
    required_cols = {"client_id", "num_examples_total", "num_examples_train", "cpu"}
    if not required_cols.issubset(cpu_map_df.columns):
        raise SystemExit(f"cpu-map-csv must contain columns: {sorted(required_cols)}")
    cpu_map_df["client_id"] = pd.to_numeric(cpu_map_df["client_id"], errors="raise").astype(int)
    cpu_map_df["num_examples_total"] = pd.to_numeric(
        cpu_map_df["num_examples_total"], errors="raise"
    ).astype(int)
    cpu_map_df["num_examples_train"] = pd.to_numeric(
        cpu_map_df["num_examples_train"], errors="raise"
    ).astype(int)
    cpu_map_df["cpu"] = normalize_cpu_series(pd.to_numeric(cpu_map_df["cpu"], errors="raise"))
    return cpu_map_df


def fit_latest_windows(
    run_csv: Path,
    *,
    group_col: str,
    time_col: str,
    window_size: int,
    min_samples: int,
    max_time: float,
) -> pd.DataFrame:
    if not run_csv.exists():
        return pd.DataFrame(columns=["id", "theta", "k", "n_samples", "time_min", "time_max", "time_mean"])
    run_df = pd.read_csv(run_csv).copy()
    if group_col not in run_df.columns:
        raise SystemExit(f"run-csv is missing group column: {group_col}")
    if time_col not in run_df.columns:
        raise SystemExit(f"run-csv is missing time column: {time_col}")

    filtered_parts: list[pd.DataFrame] = []
    for _, sub in run_df.groupby(group_col, sort=False):
        sub = sub.tail(window_size).copy()
        filtered_parts.append(sub)
    if not filtered_parts:
        return pd.DataFrame(columns=["id", "theta", "k", "n_samples", "time_min", "time_max", "time_mean"])
    latest_window_df = pd.concat(filtered_parts, ignore_index=True, sort=False)
    try:
        return fit_theta_from_run(
            latest_window_df,
            group_col=group_col,
            time_col=time_col,
            max_time=max_time,
            min_samples=min_samples,
        )
    except SystemExit:
        return pd.DataFrame(columns=["id", "theta", "k", "n_samples", "time_min", "time_max", "time_mean"])


def upsert_live_point(points: list[LivePoint], new_point: LivePoint) -> None:
    for idx, point in enumerate(points):
        if np.isclose(point.cpu, new_point.cpu, atol=1e-6):
            points[idx] = new_point
            return
    points.append(new_point)


def clip_cpu_step(current_cpu: float, desired_cpu: float, cpu_min: float, cpu_max: float, max_step: float) -> float:
    desired_cpu = float(np.clip(desired_cpu, cpu_min, cpu_max))
    if max_step <= 0:
        return desired_cpu
    delta = desired_cpu - current_cpu
    delta = float(np.clip(delta, -max_step, max_step))
    return float(np.clip(current_cpu + delta, cpu_min, cpu_max))


def run_systemctl_set_property(
    *,
    systemctl_prefix: str,
    scope_name: str,
    cpu_share: float,
) -> tuple[bool, str]:
    cmd = shlex.split(systemctl_prefix) + [
        "set-property",
        "--runtime",
        scope_name,
        f"CPUQuota={cpu_share * 100.0:.2f}%",
        "CPUQuotaPeriodSec=20ms",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = (result.stdout + "\n" + result.stderr).strip()
    return result.returncode == 0, output


def ensure_adjust_log(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ts",
                "num_examples_train",
                "client_id",
                "scope_name",
                "cpu_before",
                "cpu_after",
                "theta_now",
                "target_theta",
                "theta_rel_error",
                "applied",
                "ok",
                "message",
            ],
        )
        writer.writeheader()


def append_adjust_log(path: Path, row: dict[str, object]) -> None:
    ensure_adjust_log(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ts",
                "num_examples_train",
                "client_id",
                "scope_name",
                "cpu_before",
                "cpu_after",
                "theta_now",
                "target_theta",
                "theta_rel_error",
                "applied",
                "ok",
                "message",
            ],
        )
        writer.writerow(row)


def main() -> None:
    args = build_argument_parser().parse_args()

    history_df = load_history_points(args.history_csv)
    cpu_map_df = load_cpu_map(args.cpu_map_csv)
    current_cpu_by_num = {
        int(row["num_examples_train"]): float(row["cpu"])
        for _, row in cpu_map_df.iterrows()
    }
    client_id_by_num = {
        int(row["num_examples_train"]): int(row["client_id"])
        for _, row in cpu_map_df.iterrows()
    }
    total_by_num = {
        int(row["num_examples_train"]): int(row["num_examples_total"])
        for _, row in cpu_map_df.iterrows()
    }
    live_points: dict[int, list[LivePoint]] = {}
    last_adjust_ts: dict[int, float] = {}
    snapshot_columns = [
        "client_id",
        "num_examples_total",
        "num_examples_train",
        "cpu",
        "cpu_before",
        "theta_latest",
        "k_latest",
        "theta_rel_error",
        "n_samples",
        "cpu_proposed",
        "fit_alpha",
        "fit_beta",
        "fit_gamma",
        "point_count",
        "action",
        "apply_ok",
        "message",
    ]

    while True:
        cycle_start = time.time()
        latest_fit_df = fit_latest_windows(
            args.run_csv,
            group_col=args.group_col,
            time_col=args.time_col,
            window_size=args.window_size,
            min_samples=args.min_samples,
            max_time=args.max_time,
        )
        snapshot_rows: list[dict[str, object]] = []

        for _, fit_row in latest_fit_df.sort_values("id").iterrows():
            num_train = int(fit_row["id"])
            if num_train not in current_cpu_by_num or num_train not in client_id_by_num:
                continue

            current_cpu = float(current_cpu_by_num[num_train])
            theta_now = float(fit_row["theta"])
            k_now = float(fit_row["k"])
            rel_error = (theta_now - args.target_theta) / float(args.target_theta)

            points = live_points.setdefault(num_train, [])
            upsert_live_point(
                points,
                LivePoint(cpu=current_cpu, theta=theta_now, k=k_now, source=f"live:{args.run_csv.name}"),
            )

            points_rows = [
                {"id": num_train, "theta": point.theta, "k": point.k, "cpu": point.cpu, "source": point.source}
                for point in points
            ]
            client_history_df = history_df[history_df["id"] == num_train]
            points_df = pd.concat(
                [client_history_df, pd.DataFrame(points_rows)],
                ignore_index=True,
                sort=False,
            )
            points_df = (
                points_df.sort_values(["cpu", "source"])
                .drop_duplicates(subset=["id", "cpu"], keep="last")
                .reset_index(drop=True)
            )

            desired_cpu = np.nan
            fit_alpha = np.nan
            fit_beta = np.nan
            fit_gamma = np.nan
            point_count = int(len(points_df))
            if point_count >= 3:
                cpu_fit_df = fit_cpu_model(points_df[["id", "theta", "k", "cpu"]], args.target_theta)
                cpu_fit_row = cpu_fit_df[cpu_fit_df["id"] == num_train]
                if not cpu_fit_row.empty:
                    desired_cpu = float(cpu_fit_row.iloc[0]["cpu_required"])
                    fit_alpha = float(cpu_fit_row.iloc[0]["fit_alpha"])
                    fit_beta = float(cpu_fit_row.iloc[0]["fit_beta"])
                    fit_gamma = float(cpu_fit_row.iloc[0]["fit_gamma"])

            proposed_cpu = np.nan
            action = "hold"
            apply_ok = None
            message = ""
            if np.isfinite(desired_cpu):
                proposed_cpu = clip_cpu_step(
                    current_cpu=current_cpu,
                    desired_cpu=desired_cpu,
                    cpu_min=args.cpu_min,
                    cpu_max=args.cpu_max,
                    max_step=args.max_step,
                )

            should_adjust = (
                np.isfinite(proposed_cpu)
                and abs(rel_error) > args.tolerance
                and abs(proposed_cpu - current_cpu) >= args.min_cpu_delta
            )
            cooldown_ok = (time.monotonic() - last_adjust_ts.get(num_train, -1e9)) >= args.cooldown_seconds

            if should_adjust and cooldown_ok:
                scope_name = f"{args.scope_prefix}{client_id_by_num[num_train]}.scope"
                action = "adjust"
                if args.apply:
                    apply_ok, message = run_systemctl_set_property(
                        systemctl_prefix=args.systemctl_prefix,
                        scope_name=scope_name,
                        cpu_share=proposed_cpu,
                    )
                    if apply_ok:
                        current_cpu_by_num[num_train] = proposed_cpu
                        last_adjust_ts[num_train] = time.monotonic()
                else:
                    apply_ok = True
                    message = "dry-run"

                append_adjust_log(
                    args.adjust_log_csv,
                    {
                        "ts": f"{cycle_start:.3f}",
                        "num_examples_train": num_train,
                        "client_id": client_id_by_num[num_train],
                        "scope_name": scope_name,
                        "cpu_before": f"{current_cpu:.6f}",
                        "cpu_after": f"{proposed_cpu:.6f}",
                        "theta_now": f"{theta_now:.6f}",
                        "target_theta": f"{args.target_theta:.6f}",
                        "theta_rel_error": f"{rel_error:+.6f}",
                        "applied": int(args.apply),
                        "ok": int(bool(apply_ok)),
                        "message": message,
                    },
                )

            snapshot_rows.append(
                {
                    "client_id": client_id_by_num[num_train],
                    "num_examples_total": total_by_num[num_train],
                    "num_examples_train": num_train,
                    "cpu": current_cpu_by_num[num_train],
                    "cpu_before": current_cpu,
                    "theta_latest": theta_now,
                    "k_latest": k_now,
                    "theta_rel_error": rel_error,
                    "n_samples": int(fit_row["n_samples"]),
                    "cpu_proposed": proposed_cpu,
                    "fit_alpha": fit_alpha,
                    "fit_beta": fit_beta,
                    "fit_gamma": fit_gamma,
                    "point_count": point_count,
                    "action": action,
                    "apply_ok": apply_ok,
                    "message": message,
                }
            )

        snapshot_df = pd.DataFrame(snapshot_rows, columns=snapshot_columns)
        if not snapshot_df.empty:
            snapshot_df = snapshot_df.sort_values("num_examples_train").reset_index(drop=True)
        args.snapshot_csv.parent.mkdir(parents=True, exist_ok=True)
        snapshot_df.to_csv(args.snapshot_csv, index=False)
        args.cpu_map_out.parent.mkdir(parents=True, exist_ok=True)
        cpu_map_out_df = cpu_map_df[["client_id", "num_examples_total", "num_examples_train"]].copy()
        cpu_map_out_df["cpu"] = cpu_map_out_df["num_examples_train"].map(current_cpu_by_num).astype(float)
        cpu_map_out_df.to_csv(args.cpu_map_out, index=False)

        print(
            f"[controller] {time.strftime('%F %T')} rows={len(snapshot_df)} "
            f"apply={int(args.apply)} snapshot={args.snapshot_csv}"
        )
        if not snapshot_df.empty:
            display_cols = ["num_examples_train", "cpu_before", "cpu", "theta_latest", "theta_rel_error", "action"]
            print(snapshot_df[display_cols].to_string(index=False))

        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
