#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ANCHOR_CPU_VALUES = np.array([0.3, 0.6, 0.9], dtype=float)
ANCHOR_POINT_WEIGHT = 0.25


def logistic_cdf(t: np.ndarray, theta: float, k: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(t - theta) / k))


def power_shift(cpu: np.ndarray, alpha: float, beta: float, gamma: float) -> np.ndarray:
    return alpha * np.power(cpu, -beta) + gamma


def cpu_fit_sigma(cpu: np.ndarray) -> np.ndarray:
    weights = np.ones_like(cpu, dtype=float)
    anchor_mask = np.isclose(cpu[:, None], ANCHOR_CPU_VALUES[None, :], atol=1e-6).any(axis=1)
    weights[anchor_mask] = ANCHOR_POINT_WEIGHT
    return 1.0 / np.sqrt(weights)


def normalize_cpu_series(cpu: pd.Series) -> pd.Series:
    cpu = cpu.astype(float)
    if float(cpu.max()) > 1.5:
        return cpu / 100.0
    return cpu


def make_empirical_cdf(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(times.astype(float))
    y = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
    return x, y


def fit_theta_from_run(
    run_df: pd.DataFrame,
    *,
    group_col: str,
    time_col: str,
    max_time: float | None,
    min_samples: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for group_value, sub in run_df.groupby(group_col):
        times = pd.to_numeric(sub[time_col], errors="coerce").to_numpy(dtype=float)
        times = times[np.isfinite(times)]
        if max_time is not None:
            times = times[times <= max_time]
        if len(times) < min_samples:
            continue

        x_data, y_data = make_empirical_cdf(times)
        theta0 = float(np.median(x_data))
        k0 = float(max((np.percentile(x_data, 75) - np.percentile(x_data, 25)) / 4.0, 0.1))

        try:
            popt, _ = curve_fit(
                logistic_cdf,
                x_data,
                y_data,
                p0=[theta0, k0],
                bounds=([float(x_data.min()), 0.05], [float(x_data.max()), 1000.0]),
                maxfev=20000,
                loss="soft_l1",
            )
        except RuntimeError:
            continue

        theta_hat, k_hat = map(float, popt)
        rows.append(
            {
                "id": int(group_value),
                "theta": theta_hat,
                "k": k_hat,
                "n_samples": int(len(x_data)),
                "time_min": float(x_data.min()),
                "time_max": float(x_data.max()),
                "time_mean": float(x_data.mean()),
            }
        )

    if not rows:
        raise SystemExit("No per-client theta fits were produced from the run CSV.")

    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


def fit_cpu_model(points_df: pd.DataFrame, target_theta: float) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for cid, sub in points_df.groupby("id"):
        sub = sub.sort_values("cpu")
        cpu = sub["cpu"].to_numpy(dtype=float)
        theta = sub["theta"].to_numpy(dtype=float)

        if len(cpu) < 3:
            continue

        p0 = [float(max(theta.max() - theta.min(), 1.0)), 1.0, float(max(theta.min() * 0.5, 0.0))]
        bounds = ([0.0, 0.01, -np.inf], [np.inf, 10.0, np.inf])

        try:
            popt, _ = curve_fit(
                power_shift,
                cpu,
                theta,
                p0=p0,
                bounds=bounds,
                sigma=cpu_fit_sigma(cpu),
                absolute_sigma=False,
                maxfev=20000,
            )
        except RuntimeError:
            continue

        alpha, beta, gamma = map(float, popt)
        if alpha <= 0 or beta <= 0 or target_theta <= gamma:
            cpu_required = np.nan
        else:
            cpu_required = float((alpha / (target_theta - gamma)) ** (1.0 / beta))

        rows.append(
            {
                "id": int(cid),
                "fit_alpha": alpha,
                "fit_beta": beta,
                "fit_gamma": gamma,
                "cpu_required": cpu_required,
                "point_count": int(len(sub)),
                "cpu_points": ";".join(f"{v:.6f}" for v in cpu),
                "theta_points": ";".join(f"{v:.6f}" for v in theta),
            }
        )

    if not rows:
        raise SystemExit("No CPU recommendations were produced from the augmented points.")

    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit theta/k from a fresh run CSV, append the current (cpu, theta, k) "
            "point to historical cpu-theta data, and recompute target CPU per client."
        )
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=Path("/home/xiaoyan/wholeflower/femnist/mnist_cpu_theta.csv"),
        help="Historical cpu-theta CSV with columns id,theta,k,cpu.",
    )
    parser.add_argument(
        "--run-csv",
        type=Path,
        nargs="+",
        required=True,
        help="One or more run CSVs, e.g. speech_commom_target_85_0.csv speech_commom_target_85_1.csv.",
    )
    parser.add_argument(
        "--cpu-map-csv",
        type=Path,
        nargs="+",
        default=[Path("/home/xiaoyan/wholeflower/femnist/speech_comm/client_number_example.csv")],
        help=(
            "One or more CPU map CSVs with columns client_id,num_examples_total,"
            "num_examples_train,cpu. Pass one path to reuse it for every run-csv, "
            "or pass the same count as run-csv to pair them by position."
        ),
    )
    parser.add_argument(
        "--target-theta",
        type=float,
        required=True,
        help="Target theta to convert into a new CPU recommendation.",
    )
    parser.add_argument(
        "--group-col",
        default="num_examples",
        help="Column in run-csv used to identify each logical client.",
    )
    parser.add_argument(
        "--time-col",
        default="client_train_s",
        help="Column in run-csv used as the per-round duration for theta fitting.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=1000.0,
        help="Discard durations above this threshold before fitting.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum number of samples required to fit one client.",
    )
    parser.add_argument(
        "--latest-fit-out",
        type=Path,
        default=None,
        help="Output CSV for latest fitted theta/k values.",
    )
    parser.add_argument(
        "--augmented-history-out",
        type=Path,
        default=None,
        help="Output CSV for historical points plus the latest point.",
    )
    parser.add_argument(
        "--cpu-map-out",
        type=Path,
        default=None,
        help="Output CSV for the next launch. First four columns stay launch_clients-compatible.",
    )
    return parser


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            series = df[name]
            if isinstance(series, pd.DataFrame):
                return series.iloc[:, 0]
            return series
    raise KeyError(candidates[0])


def _normalize_cpu_map_df(cpu_map_df: pd.DataFrame) -> pd.DataFrame:
    required_cpu_map_cols = {"client_id", "num_examples_total", "num_examples_train", "cpu"}
    if not required_cpu_map_cols.issubset(cpu_map_df.columns):
        raise SystemExit(f"cpu-map-csv must contain columns: {sorted(required_cpu_map_cols)}")

    cpu_map_df = cpu_map_df.copy()
    cpu_map_df["client_id"] = pd.to_numeric(cpu_map_df["client_id"], errors="raise").astype(int)
    cpu_map_df["num_examples_total"] = pd.to_numeric(
        cpu_map_df["num_examples_total"], errors="raise"
    ).astype(int)
    cpu_map_df["num_examples_train"] = pd.to_numeric(
        cpu_map_df["num_examples_train"], errors="raise"
    ).astype(int)
    cpu_map_df["cpu"] = normalize_cpu_series(pd.to_numeric(cpu_map_df["cpu"], errors="raise"))
    drop_if_present = [
        "cpu_prev",
        "theta_latest",
        "k_latest",
        "n_samples",
        "fit_alpha",
        "fit_beta",
        "fit_gamma",
        "cpu_required",
        "point_count",
        "cpu_points",
        "theta_points",
    ]
    return cpu_map_df.drop(columns=[c for c in drop_if_present if c in cpu_map_df.columns])


def main() -> None:
    args = build_argument_parser().parse_args()

    history_df = pd.read_csv(args.history_csv).copy()
    run_csvs = list(args.run_csv)
    cpu_map_csvs = list(args.cpu_map_csv)

    required_history_cols = {"id", "theta", "k", "cpu"}
    if not required_history_cols.issubset(history_df.columns):
        raise SystemExit(f"history-csv must contain columns: {sorted(required_history_cols)}")
    if len(cpu_map_csvs) not in (1, len(run_csvs)):
        raise SystemExit("cpu-map-csv must be either one path or the same count as run-csv.")

    history_df["id"] = pd.to_numeric(history_df["id"], errors="raise").astype(int)
    history_df["theta"] = pd.to_numeric(history_df["theta"], errors="raise").astype(float)
    history_df["k"] = pd.to_numeric(history_df["k"], errors="raise").astype(float)
    history_df["cpu"] = normalize_cpu_series(pd.to_numeric(history_df["cpu"], errors="raise"))

    cpu_map_paths_for_runs = cpu_map_csvs if len(cpu_map_csvs) == len(run_csvs) else cpu_map_csvs * len(run_csvs)
    latest_fit_frames: list[pd.DataFrame] = []
    latest_point_frames: list[pd.DataFrame] = []
    last_cpu_map_df: pd.DataFrame | None = None

    for source_order, (run_csv, cpu_map_csv) in enumerate(zip(run_csvs, cpu_map_paths_for_runs), start=1):
        run_df = pd.read_csv(run_csv).copy()
        if args.group_col not in run_df.columns:
            raise SystemExit(f"run-csv is missing group column: {args.group_col}")
        if args.time_col not in run_df.columns:
            raise SystemExit(f"run-csv is missing time column: {args.time_col}")

        cpu_map_df = _normalize_cpu_map_df(pd.read_csv(cpu_map_csv))
        last_cpu_map_df = cpu_map_df

        latest_fit_df_single = fit_theta_from_run(
            run_df,
            group_col=args.group_col,
            time_col=args.time_col,
            max_time=args.max_time,
            min_samples=args.min_samples,
        )
        latest_fit_df_single = latest_fit_df_single.merge(
            cpu_map_df[["client_id", "num_examples_train", "cpu"]],
            left_on="id",
            right_on="num_examples_train",
            how="left",
        )
        latest_fit_df_single = latest_fit_df_single.rename(columns={"cpu": "cpu_current"})
        latest_fit_df_single["source_run"] = run_csv.name
        latest_fit_df_single["source_cpu_map"] = cpu_map_csv.name
        latest_fit_df_single["source_order"] = source_order
        missing_cpu = latest_fit_df_single["cpu_current"].isna()
        if missing_cpu.any():
            missing_ids = latest_fit_df_single.loc[missing_cpu, "id"].astype(str).tolist()
            raise SystemExit(
                "Latest theta fits could not be matched to cpu-map num_examples_train for ids: "
                + ", ".join(missing_ids)
            )

        latest_points_df_single = latest_fit_df_single[["id", "theta", "k", "cpu_current", "source_order"]].copy()
        latest_points_df_single = latest_points_df_single.rename(columns={"cpu_current": "cpu"})
        latest_points_df_single["source"] = f"latest:{run_csv.name}"
        latest_fit_frames.append(latest_fit_df_single)
        latest_point_frames.append(latest_points_df_single)

    latest_fit_df = pd.concat(latest_fit_frames, ignore_index=True, sort=False)
    latest_points_df = pd.concat(latest_point_frames, ignore_index=True, sort=False)
    if last_cpu_map_df is None:
        raise SystemExit("No cpu-map-csv was provided.")
    cpu_map_df = last_cpu_map_df
    latest_fit_for_output_df = (
        latest_fit_df.sort_values(["id", "source_order"])
        .drop_duplicates(subset=["id"], keep="last")
        .reset_index(drop=True)
    )

    history_with_source = history_df.copy()
    history_with_source["source"] = f"history:{args.history_csv.name}"
    history_with_source["source_order"] = 0
    augmented_history_df = pd.concat(
        [history_with_source, latest_points_df],
        ignore_index=True,
        sort=False,
    )
    augmented_history_df = (
        augmented_history_df.sort_values(["id", "cpu", "source_order", "source"])
        .drop_duplicates(subset=["id", "cpu"], keep="last")
        .reset_index(drop=True)
    )

    recommendation_df = fit_cpu_model(augmented_history_df[["id", "theta", "k", "cpu"]], args.target_theta)

    next_cpu_map_df = cpu_map_df.merge(
        latest_fit_for_output_df[["id", "theta", "k", "n_samples"]].rename(
            columns={"id": "num_examples_train", "theta": "theta_latest", "k": "k_latest"}
        ),
        on="num_examples_train",
        how="left",
    ).merge(
        recommendation_df.rename(columns={"id": "num_examples_train"}),
        on="num_examples_train",
        how="left",
        suffixes=("", "_new"),
    )
    next_cpu_map_df = next_cpu_map_df.rename(columns={"cpu": "cpu_prev"})
    cpu_required_series = _pick_column(next_cpu_map_df, ["cpu_required", "cpu_required_new"])
    next_cpu_map_df["cpu"] = cpu_required_series.combine_first(next_cpu_map_df["cpu_prev"])
    if "cpu_required" not in next_cpu_map_df.columns:
        next_cpu_map_df["cpu_required"] = cpu_required_series

    last_run_csv = run_csvs[-1]
    last_cpu_map_csv = cpu_map_paths_for_runs[-1]
    if len(run_csvs) == 1:
        run_stem = last_run_csv.stem
        latest_fit_out = args.latest_fit_out or last_run_csv.with_name(f"{run_stem}_theta_fit.csv")
        augmented_history_out = args.augmented_history_out or last_run_csv.with_name(
            f"{run_stem}_theta_cpu_augmented.csv"
        )
    else:
        multi_stem = f"{last_run_csv.stem}_multi_{len(run_csvs)}runs"
        latest_fit_out = args.latest_fit_out or last_run_csv.with_name(f"{multi_stem}_theta_fit.csv")
        augmented_history_out = args.augmented_history_out or last_run_csv.with_name(
            f"{multi_stem}_theta_cpu_augmented.csv"
        )
    cpu_map_out = args.cpu_map_out or last_cpu_map_csv.with_name(
        f"{last_cpu_map_csv.stem}_target_{str(args.target_theta).replace('.', '_')}.csv"
    )

    latest_fit_df.to_csv(latest_fit_out, index=False)
    augmented_history_df.to_csv(augmented_history_out, index=False)

    ordered_cols = [
        "client_id",
        "num_examples_total",
        "num_examples_train",
        "cpu",
        "cpu_prev",
        "theta_latest",
        "k_latest",
        "n_samples",
        "fit_alpha",
        "fit_beta",
        "fit_gamma",
        "cpu_required",
        "point_count",
        "cpu_points",
        "theta_points",
    ]
    existing_ordered_cols = [col for col in ordered_cols if col in next_cpu_map_df.columns]
    remaining_cols = [col for col in next_cpu_map_df.columns if col not in existing_ordered_cols]
    next_cpu_map_df[existing_ordered_cols + remaining_cols].to_csv(cpu_map_out, index=False)

    print(f"Wrote latest theta fits to {latest_fit_out}")
    print(f"Wrote augmented cpu-theta points to {augmented_history_out}")
    print(f"Wrote next CPU map to {cpu_map_out}")
    print()
    print("num_examples_train  cpu_prev  theta_latest  cpu_required")
    print("------------------  --------  ------------  ------------")
    for _, row in next_cpu_map_df.sort_values("num_examples_train").iterrows():
        theta_txt = "nan" if pd.isna(row.get("theta_latest")) else f"{float(row['theta_latest']):12.4f}"
        cpu_req_txt = "nan" if pd.isna(row.get("cpu_required")) else f"{float(row['cpu_required']):12.4f}"
        print(
            f"{int(row['num_examples_train']):18d}  "
            f"{float(row['cpu_prev']):8.4f}  "
            f"{theta_txt}  "
            f"{cpu_req_txt}"
        )


if __name__ == "__main__":
    main()
