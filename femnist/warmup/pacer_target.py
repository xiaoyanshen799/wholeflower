"""Derive a rounded Pacer deadline and its matching common theta target."""

import math
from pathlib import Path

from pacer.protocol import ControlConfig


REF_A = 0.02
Q = 0.9
Q_LOWER = 0.85
Q_UPPER = 0.95
DEADLINE_QUANTUM_S = 0.5


def _quantile_logit(q, gamma_total):
    if type(gamma_total) is not int or gamma_total < 1:
        raise ValueError("gamma_total must be a positive integer")
    if type(q) not in (int, float) or not math.isfinite(q) or not 0 < q < 1:
        raise ValueError("q must be a finite probability in (0, 1)")
    log_client_cdf = math.log(float(q)) / gamma_total
    return log_client_cdf - math.log(-math.expm1(log_client_cdf))


def ceil_to_quantum(value, quantum=DEADLINE_QUANTUM_S):
    if not math.isfinite(value) or value <= 0 or not math.isfinite(quantum) or quantum <= 0:
        raise ValueError("value and quantum must be finite and positive")
    return math.ceil(value / quantum - 1e-12) * quantum


def derive_pacer_target(anchor_theta_s, client_count, ref_a=REF_A, q=Q,
                        deadline_quantum_s=DEADLINE_QUANTUM_S):
    """Use the round-level q quantile before and after rounding the deadline."""
    if (type(anchor_theta_s) not in (int, float) or not math.isfinite(anchor_theta_s)
            or anchor_theta_s <= 0):
        raise ValueError("anchor_theta_s must be finite and positive")
    if type(ref_a) not in (int, float) or not math.isfinite(ref_a) or ref_a <= 0:
        raise ValueError("ref_a must be finite and positive")
    logit = _quantile_logit(q, client_count)
    factor = 1.0 + float(ref_a) * logit
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Pacer parameters produce a nonpositive deadline factor")
    raw_deadline = float(anchor_theta_s) * factor
    deadline = ceil_to_quantum(raw_deadline, deadline_quantum_s)
    return {
        "anchor_theta_s": float(anchor_theta_s),
        "raw_deadline_s": raw_deadline,
        "deadline_s": deadline,
        "theta_target_s": deadline / factor,
        "ref_a": float(ref_a),
        "q": float(q),
        "assumed_gamma_total": client_count,
        "deadline_quantum_s": float(deadline_quantum_s),
    }


def build_pacer_control(output_dir, client_ids, target):
    control = ControlConfig(
        schema_version=1,
        run_id=f"{Path(output_dir).name}-pacer",
        config_version=1,
        effective_round=1,
        theta_target_s=target["theta_target_s"],
        deadline_s=target["deadline_s"],
        ref_a=target["ref_a"],
        q=target["q"],
        q_lower=Q_LOWER,
        q_upper=Q_UPPER,
        required_client_ids=tuple(client_ids),
        timing_basis="local_train_wall_s",
        gamma_window=50,
        gamma_min_samples=20,
        violation_patience=3,
        feedback_max_age_rounds=2,
    )
    return control.to_dict()
