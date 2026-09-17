"""Logistic timing fits and bounded per-client CPU corrections."""

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.stats import logistic


@dataclass(frozen=True)
class TimingFit:
    theta_s: float
    k_s: float
    n_samples: int
    ks_distance: float
    empirical_p90_s: float

    def to_dict(self):
        return asdict(self)


def fit_logistic(samples):
    """Fit the positive-support logistic with loc > 0 and scale > 0."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or len(values) < 3 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Logistic fitting needs at least three finite positive durations")
    if np.ptp(values) <= np.finfo(float).eps * max(1.0, float(np.mean(values))):
        raise ValueError("Constant timing samples cannot identify a logistic scale")
    theta, scale = logistic.fit(values)
    normalizer = float(np.median(values))
    normalized = values / normalizer

    def objective(params):
        loc, log_scale = params
        width = np.exp(log_scale)
        return float(-np.mean(logistic.logpdf(normalized, loc=loc, scale=width))
                     + logistic.logsf(0, loc=loc, scale=width))

    starts = [
        [max(theta / normalizer, 1e-6), math.log(scale / normalizer)],
        [1.0, math.log(max(float(np.std(normalized)) * math.sqrt(3) / math.pi, 1e-6))],
        [float(np.mean(normalized)), math.log(max(float(np.std(normalized)), 1e-6))],
    ]
    candidates = []
    for start in starts:
        for method in ("L-BFGS-B", "Powell"):
            result = minimize(
                objective,
                start,
                method=method,
                bounds=[(1e-8, None), (-25, 10)],
                options={"maxiter": 10000},
            )
            if result.success and np.all(np.isfinite(result.x)) and math.isfinite(float(result.fun)):
                candidates.append(result)
        result = minimize(objective, start, method="Nelder-Mead", options={"maxiter": 10000})
        if (result.success and np.all(np.isfinite(result.x)) and math.isfinite(float(result.fun))
                and result.x[0] > 1e-8 and -25 < result.x[1] < 10):
            candidates.append(result)
    if not candidates:
        raise ValueError("Truncated logistic fit failed for all optimizer starts")
    fitted = min(candidates, key=lambda result: float(result.fun))
    theta, scale = float(fitted.x[0] * normalizer), float(np.exp(fitted.x[1]) * normalizer)
    if fitted.x[0] <= 1.01e-8 or not -24.99 < fitted.x[1] < 9.99:
        raise ValueError("Logistic fit reached a numerical bound")
    ordered = np.sort(values)
    f0 = logistic.cdf(0, loc=theta, scale=scale)
    cdf = (logistic.cdf(ordered, loc=theta, scale=scale) - f0) / (1 - f0)
    n = len(values)
    ks = max(np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(n) / n))
    return TimingFit(theta, scale, n, float(ks), float(np.quantile(values, 0.90)))


@dataclass(frozen=True)
class CpuModel:
    mu: float
    beta: float
    theta_floor_s: float
    relative_rmse: float

    def theta(self, cpu):
        return self.mu * cpu ** -self.beta + self.theta_floor_s

    def inverse(self, theta):
        if theta <= self.theta_floor_s:
            return math.inf
        return (self.mu / (theta - self.theta_floor_s)) ** (1 / self.beta)

    def to_dict(self):
        return asdict(self)


def fit_cpu_model(points):
    cpu, theta = np.asarray(points, dtype=np.float64).T
    if len(cpu) < 4 or len(set(cpu)) < 4 or np.any(cpu <= 0) or np.any(theta <= 0):
        raise ValueError("CPU fitting needs four distinct positive CPU anchors")
    if not np.all(np.isfinite(cpu)) or not np.all(np.isfinite(theta)):
        raise ValueError("Nonfinite CPU profile")
    normalizer = float(np.median(theta))
    y = theta / normalizer

    def residual(params):
        mu, beta, floor = params
        return (mu * cpu ** -beta + floor - y) / y

    candidates = []
    for beta in (0.5, 1.0, 2.0):
        result = least_squares(residual, [float(np.min(y) * np.max(cpu) ** beta), beta, 0.0],
                               bounds=([1e-10, 0.05, 0], [np.inf, 5.0, float(np.min(y) * 0.999)]),
                               max_nfev=4000)
        if result.success and np.all(np.isfinite(result.x)):
            candidates.append(result)
    if not candidates:
        raise ValueError("Monotone CPU-to-theta fitting failed")
    best = min(candidates, key=lambda result: float(np.sum(result.fun ** 2)))
    mu, beta, floor = best.x
    return CpuModel(float(mu * normalizer), float(beta), float(floor * normalizer),
                    float(np.sqrt(np.mean(best.fun ** 2))))


def quantize_cpu(cpu, minimum, maximum, step):
    if math.isnan(cpu):
        raise ValueError("CPU proposal is NaN")
    low_tick = math.ceil((minimum - 1e-12) / step)
    high_tick = math.floor((maximum + 1e-12) / step)
    if high_tick < low_tick:
        raise ValueError("CPU bounds contain no executable quota")
    cpu = min(maximum, max(minimum, cpu))
    tick = min(high_tick, max(low_tick, round(cpu / step)))
    return round(tick * step, 10)


def adjust_cpu(model, current, measured_theta, target, history, minimum, maximum, step):
    """Correct only this client's allocation; keep the chosen target fixed."""
    too_slow = measured_theta > target
    scale = measured_theta / model.theta(current)
    proposed = model.inverse(target / scale)
    # A measured bracket is preferable to extrapolating through the wrong side.
    if too_slow:
        opposite = [cpu for cpu, theta in history if cpu > current and theta <= target]
        if opposite:
            upper = min(opposite)
            if not current < proposed < upper:
                proposed = (current + upper) / 2
        proposed = max(current + step, proposed)
    else:
        opposite = [cpu for cpu, theta in history if cpu < current and theta >= target]
        if opposite:
            lower = max(opposite)
            if not lower < proposed < current:
                proposed = (lower + current) / 2
        proposed = min(current - step, proposed)
    new_cpu = quantize_cpu(proposed, minimum, maximum, step)
    if (too_slow and new_cpu <= current) or (not too_slow and new_cpu >= current):
        raise ValueError(f"Target is not reachable within CPU bounds at cpu={current:.4f}, theta={measured_theta:.6f}")
    return new_cpu
