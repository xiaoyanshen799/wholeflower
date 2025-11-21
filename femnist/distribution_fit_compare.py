import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from scipy.optimize import curve_fit, least_squares
from scipy.stats import lognorm, weibull_min, gompertz, genlogistic, logistic


# ----------------------------- 数据加载 -----------------------------
CSV_FILE = "/home/xiaoyan/wholeflower/femnist/logs/comm_times.csv"
TAIL_THRESHOLD = 0.9  # 统计尾部误差的下限
PLOT_TOP_N = 3        # 每个客户端展示拟合结果的数量
MAX_DURATION = 43

def read_client_durations(path: str) -> Dict[str, np.ndarray]:
    df = pd.read_csv(path)
    durations_by_client: Dict[str, List[float]] = defaultdict(list)

    for _, row in df.iterrows():
        cid = str(row["client_id"])
        duration = (
            row["client_train_s"]
            + row["server_to_client_ms"] / 1000.0
            + row["client_to_server_ms"] / 1000.0
        )
        if duration <= MAX_DURATION:
            durations_by_client[cid].append(duration)

    return {cid: np.sort(np.array(values, dtype=float))
            for cid, values in durations_by_client.items() if values}


@dataclass
class FitResult:
    model: str
    method: str
    params: Dict[str, float]
    sse: float
    tail_sse: float
    predictor: Callable[[np.ndarray], np.ndarray]


# ----------------------------- 模型定义 -----------------------------

def logistic_cdf(t: np.ndarray, theta: float, k: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(t - theta) / k))


def logistic_quantile_init(x: np.ndarray) -> Tuple[float, float]:
    theta = float(np.median(x))
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    k = max(iqr / 2.0, 0.05)
    return theta, k


def compute_empirical_cdf(x: np.ndarray) -> np.ndarray:
    n = len(x)
    ranks = np.arange(1, n + 1, dtype=float)
    return ranks / (n + 1.0)  # 使用 n+1 平滑尾部


def evaluate_fit(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    residual = y_pred - y_true
    sse = float(np.sum(residual ** 2))
    mask = y_true >= TAIL_THRESHOLD
    tail_sse = float(np.sum(residual[mask] ** 2)) if np.any(mask) else 0.0
    return sse, tail_sse


# ----------------------------- 拟合策略 -----------------------------

def fit_logistic_curve_fit(x: np.ndarray, y: np.ndarray, loss: str = "linear") -> FitResult:
    theta0, k0 = logistic_quantile_init(x)
    bounds = ([min(x), 0.05], [max(x), 500.0])

    popt, _ = curve_fit(
        logistic_cdf,
        x,
        y,
        p0=[theta0, k0],
        bounds=bounds,
        maxfev=20000,
        loss=loss,
    )

    def predictor(t: np.ndarray) -> np.ndarray:
        return logistic_cdf(t, *popt)

    y_pred = predictor(x)
    sse, tail_sse = evaluate_fit(y, y_pred)
    return FitResult(
        model="logistic",
        method=f"curve_fit(loss={loss})",
        params={"theta": float(popt[0]), "k": float(popt[1])},
        sse=sse,
        tail_sse=tail_sse,
        predictor=predictor,
    )


def fit_logistic_quantile(x: np.ndarray, y: np.ndarray) -> FitResult:
    theta, k = logistic_quantile_init(x)

    def predictor(t: np.ndarray) -> np.ndarray:
        return logistic_cdf(t, theta, k)

    y_pred = predictor(x)
    sse, tail_sse = evaluate_fit(y, y_pred)
    return FitResult(
        model="logistic",
        method="quantile-init",
        params={"theta": theta, "k": k},
        sse=sse,
        tail_sse=tail_sse,
        predictor=predictor,
    )


def fit_logistic_weighted(x: np.ndarray, y: np.ndarray) -> FitResult:
    theta0, k0 = logistic_quantile_init(x)
    bounds = ([min(x), 0.05], [max(x), 500.0])

    def residuals(params: np.ndarray) -> np.ndarray:
        theta, k = params
        model = logistic_cdf(x, theta, k)
        weights = 0.2 + 0.8 * np.power(y, 3.0)
        return (model - y) * np.sqrt(weights)

    result = least_squares(
        residuals,
        x0=np.array([theta0, k0]),
        bounds=(np.array(bounds[0]), np.array(bounds[1])),
        max_nfev=20000,
    )

    theta_hat, k_hat = result.x

    def predictor(t: np.ndarray) -> np.ndarray:
        return logistic_cdf(t, theta_hat, k_hat)

    y_pred = predictor(x)
    sse, tail_sse = evaluate_fit(y, y_pred)
    return FitResult(
        model="logistic",
        method="weighted-least-squares",
        params={"theta": float(theta_hat), "k": float(k_hat)},
        sse=sse,
        tail_sse=tail_sse,
        predictor=predictor,
    )


def fit_logistic_mle(x: np.ndarray, y: np.ndarray) -> FitResult:
    loc, scale = logistic.fit(x)

    def predictor(t: np.ndarray) -> np.ndarray:
        return logistic.cdf(t, loc=loc, scale=scale)

    y_pred = predictor(x)
    sse, tail_sse = evaluate_fit(y, y_pred)
    return FitResult(
        model="logistic",
        method="MLE",
        params={"theta": float(loc), "k": float(scale)},
        sse=sse,
        tail_sse=tail_sse,
        predictor=predictor,
    )


def fit_lognorm(x: np.ndarray, y: np.ndarray) -> FitResult:
    shape, loc, scale = lognorm.fit(x, floc=0.0)

    def predictor(t: np.ndarray) -> np.ndarray:
        return lognorm.cdf(t, shape, loc=loc, scale=scale)

    y_pred = predictor(x)
    sse, tail_sse = evaluate_fit(y, y_pred)
    return FitResult(
        model="lognorm",
        method="scipy-fit",
        params={"shape": float(shape), "loc": float(loc), "scale": float(scale)},
        sse=sse,
        tail_sse=tail_sse,
        predictor=predictor,
    )


def fit_weibull(x: np.ndarray, y: np.ndarray) -> FitResult:
    c, loc, scale = weibull_min.fit(x, floc=0.0)

    def predictor(t: np.ndarray) -> np.ndarray:
        return weibull_min.cdf(t, c, loc=loc, scale=scale)

    y_pred = predictor(x)
    sse, tail_sse = evaluate_fit(y, y_pred)
    return FitResult(
        model="weibull",
        method="scipy-fit",
        params={"shape": float(c), "loc": float(loc), "scale": float(scale)},
        sse=sse,
        tail_sse=tail_sse,
        predictor=predictor,
    )


def fit_weibull_curve_fit(x: np.ndarray, y: np.ndarray) -> FitResult:
    shape0 = 1.5
    scale0 = float(np.mean(x)) if np.mean(x) > 0 else 1.0

    def predictor_func(t: np.ndarray, shape: float, scale: float) -> np.ndarray:
        return weibull_min.cdf(t, shape, loc=0.0, scale=scale)

    bounds = ([0.2, 0.1], [10.0, 500.0])

    popt, _ = curve_fit(
        predictor_func,
        x,
        y,
        p0=[shape0, scale0],
        bounds=bounds,
        maxfev=20000,
    )
    shape_hat, scale_hat = popt

    def predictor(t: np.ndarray) -> np.ndarray:
        return predictor_func(t, shape_hat, scale_hat)

    y_pred = predictor(x)
    sse, tail_sse = evaluate_fit(y, y_pred)
    return FitResult(
        model="weibull",
        method="curve_fit",
        params={"shape": float(shape_hat), "scale": float(scale_hat)},
        sse=sse,
        tail_sse=tail_sse,
        predictor=predictor,
    )


def fit_gompertz(x: np.ndarray, y: np.ndarray) -> FitResult:
    c, loc, scale = gompertz.fit(x, floc=0.0)

    def predictor(t: np.ndarray) -> np.ndarray:
        return gompertz.cdf(t, c, loc=loc, scale=scale)

    y_pred = predictor(x)
    sse, tail_sse = evaluate_fit(y, y_pred)
    return FitResult(
        model="gompertz",
        method="scipy-fit",
        params={"shape": float(c), "loc": float(loc), "scale": float(scale)},
        sse=sse,
        tail_sse=tail_sse,
        predictor=predictor,
    )


def fit_genlogistic(x: np.ndarray, y: np.ndarray) -> FitResult:
    c, loc, scale = genlogistic.fit(x, floc=0.0)

    def predictor(t: np.ndarray) -> np.ndarray:
        return genlogistic.cdf(t, c, loc=loc, scale=scale)

    y_pred = predictor(x)
    sse, tail_sse = evaluate_fit(y, y_pred)
    return FitResult(
        model="genlogistic",
        method="scipy-fit",
        params={"shape": float(c), "loc": float(loc), "scale": float(scale)},
        sse=sse,
        tail_sse=tail_sse,
        predictor=predictor,
    )

FITTERS = {
    "logistic_curve_fit": lambda x, y: fit_logistic_curve_fit(x, y, loss="linear"),
    "logistic_curve_fit_soft_l1": lambda x, y: fit_logistic_curve_fit(x, y, loss="soft_l1"),
    "logistic_quantile": fit_logistic_quantile,
    "logistic_weighted": fit_logistic_weighted,
    "logistic_mle": fit_logistic_mle,
    "lognorm": fit_lognorm,
    "weibull_mle": fit_weibull,
    "weibull_curve_fit": fit_weibull_curve_fit,
    "gompertz": fit_gompertz,
    "genlogistic": fit_genlogistic,
}


# ----------------------------- 主流程 -----------------------------

def main() -> None:
    durations = read_client_durations(CSV_FILE)
    if not durations:
        raise SystemExit("No duration data found")

    plot_dir = Path(__file__).resolve().parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for idx, (cid, samples) in enumerate(sorted(durations.items())):
        y_emp = compute_empirical_cdf(samples)
        results: List[FitResult] = []

        for name, fitter in FITTERS.items():
            try:
                result = fitter(samples, y_emp)
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] client {cid}: {name} failed ({exc})")

        if not results:
            print(f"[WARN] client {cid}: no successful fits")
            continue

        results.sort(key=lambda r: (r.tail_sse, r.sse))
        best = results[0]
        summary_rows.append({
            "client": cid,
            "best_model": best.model,
            "best_method": best.method,
            "tail_sse": best.tail_sse,
            "sse": best.sse,
        })

        # 绘制当前客户端的拟合效果
        plt.figure(figsize=(8, 4))
        plt.scatter(samples, y_emp, s=12, alpha=0.6, label="empirical")
        colors = plt.cm.tab10(np.linspace(0, 1, min(PLOT_TOP_N, len(results))))
        for color, res in zip(colors, results[:PLOT_TOP_N]):
            grid = np.linspace(samples.min(), samples.max(), 200)
            plt.plot(grid, res.predictor(grid), label=f"{res.model}-{res.method}", color=color)
        plt.title(f"Client {cid} CDF fits")
        plt.xlabel("Time (s)")
        plt.ylabel("Cumulative probability")
        plt.ylim(0, 1.05)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_dir / f"client_{idx:03d}_{cid}_fits.png")
        plt.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_path = plot_dir / "fit_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved per-client comparison plots in {plot_dir}/ and summary to {summary_path}")


if __name__ == "__main__":
    main()
