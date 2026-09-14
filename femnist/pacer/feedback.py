"""Fit the single power exponent from ordinary local-training timings."""

import warnings
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from .protocol import positive_number


@dataclass(frozen=True)
class GammaEstimate:
    gamma: float | None
    n_samples: int
    sample_end_round: int
    reason: str
    rmse: float | None = None


class GammaEstimator:
    def __init__(self):
        self._key = None
        self.samples = deque()
        self.last_round = 0

    def observe(self, train_time_s, server_round, settings, resource_signature="external", workload_signature=()):
        # Never combine measurements made under different reference/resource/work settings.
        key = (settings, resource_signature, workload_signature)
        if key != self._key:
            self._key = key
            self.samples = deque(maxlen=settings.gamma_window)
            self.last_round = 0
        if server_round <= self.last_round:
            return GammaEstimate(None, len(self.samples), self.last_round, "duplicate_or_old_round")
        self.last_round = server_round
        try:
            duration = positive_number(train_time_s, "train_time")
        except ValueError:
            self.samples.clear()
            return GammaEstimate(None, 0, server_round, "invalid_train_time")
        self.samples.append(duration)
        n = len(self.samples)
        if n < settings.gamma_min_samples:
            return GammaEstimate(None, n, server_round, "insufficient_samples")

        times = np.sort(np.asarray(self.samples, dtype=np.float64))
        if np.ptp(times) == 0:
            return GammaEstimate(None, n, server_round, "degenerate_samples")
        empirical = np.searchsorted(times, times, side="right") / float(n)
        z = (times - settings.theta_target_s) / (settings.ref_a * settings.theta_target_s)
        log_f = -np.logaddexp(0.0, -z)
        if not np.all(np.isfinite(log_f)):
            return GammaEstimate(None, n, server_round, "invalid_reference")

        def powered_cdf(_times, gamma):
            return np.exp(gamma * log_f)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                fitted, _ = curve_fit(
                    powered_cdf, times, empirical, p0=[1.0],
                    bounds=([1e-6], [1e6]), maxfev=2000,
                )
            gamma = float(fitted[0])
            sensitivity = np.linalg.norm(log_f * np.exp(gamma * log_f))
            if not np.isfinite(gamma) or gamma <= 1.01e-6 or gamma >= 0.99e6 or sensitivity < 1e-12:
                return GammaEstimate(None, n, server_round, "unidentifiable_or_bound_gamma")
            rmse = float(np.sqrt(np.mean((powered_cdf(times, gamma) - empirical) ** 2)))
            return GammaEstimate(gamma, n, server_round, "ok", rmse)
        except (RuntimeError, ValueError, OptimizeWarning, FloatingPointError):
            return GammaEstimate(None, n, server_round, "fit_failed")
