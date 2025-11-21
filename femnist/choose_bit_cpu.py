import argparse
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

try:
    # SciPy gives a better fit for the power-law model
    from scipy.optimize import curve_fit  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    curve_fit = None  # type: ignore
    _HAVE_SCIPY = False


# ---------------------------- Models ----------------------------

def theta_comm_model(bits: np.ndarray, k: float, gamma: float) -> np.ndarray:
    """Linear model for compression cost: theta_comm(b) = k * b + gamma."""
    return k * bits + gamma


def theta_comp_model(cpu: np.ndarray, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Power law + constant: theta_comp(c) = alpha * c^(-beta) + gamma.

    Assumes alpha >= 0, beta >= 0, gamma >= 0.
    """
    return alpha * np.power(cpu, -beta) + gamma


def fit_comm_linear(df: pd.DataFrame) -> Tuple[float, float]:
    """Fit k & gamma from 3 (bit, theta) points using linear regression.

    Expects columns: 'cpu_size' (bits), 'theta'.
    Returns (k, gamma).
    """
    x = df["cpu_size"].to_numpy(dtype=float)
    y = df["theta"].to_numpy(dtype=float)
    # Use simple least squares (deg=1). Stable with 3 points.
    k, gamma = np.polyfit(x, y, 1)
    return float(k), float(gamma)


def fit_comp_power(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Fit alpha, beta, gamma for theta_comp(c) = alpha * c^(-beta) + gamma.

    With SciPy available, uses bounded non-linear least squares. As a
    fallback (no SciPy), assumes gamma=0 and fits log-linear model.
    Expects columns: 'cpu_size' (share in (0,1]), 'theta'.
    Returns (alpha, beta, gamma).
    """
    c = df["cpu_size"].to_numpy(dtype=float)
    y = df["theta"].to_numpy(dtype=float)

    if _HAVE_SCIPY:
        def f(xx, a, b, g):
            return theta_comp_model(xx, a, b, g)

        # initial guess
        a0 = max(y) - min(y)
        b0 = 1.0
        g0 = max(0.0, min(y) * 0.25)
        p0 = (max(a0, 1e-6), max(b0, 1e-6), g0)
        bounds = ([0.0, 0.0, 0.0], [np.inf, 10.0, np.inf])
        (a, b, g), _ = curve_fit(f, c, y, p0=p0, bounds=bounds, maxfev=20000)
        return float(a), float(b), float(g)

    # Fallback: gamma=0, fit log(theta) = log(alpha) - beta * log(c)
    c = np.clip(c, 1e-6, np.inf)
    y = np.maximum(y, 1e-6)
    X = np.vstack([np.log(c), np.ones_like(c)]).T
    # Solve: log(y) = -beta * log(c) + log(alpha)
    beta_neg, log_alpha = np.linalg.lstsq(X, np.log(y), rcond=None)[0]
    beta = -beta_neg
    alpha = float(np.exp(log_alpha))
    gamma = 0.0
    return float(alpha), float(beta), float(gamma)


@dataclass
class ClientModels:
    client_id: str
    # compression model
    k: float
    gamma_comm: float
    # compute model
    alpha: float
    beta: float
    gamma_comp: float

    def theta_all(self, bits: float, cpu: float, use_comm_gamma_as_total: bool = True) -> float:
        """Total theta according to: theta_all = k*b + alpha*c^(-beta) + gamma.

        When use_comm_gamma_as_total is True, use gamma_comm as the shared
        constant; otherwise use gamma_comp.
        """
        gamma_total = self.gamma_comm if use_comm_gamma_as_total else self.gamma_comp
        return float(self.k * bits + self.alpha * (cpu ** (-self.beta)) + gamma_total)

    def theta_comm(self, bits: float, use_comm_gamma_as_total: bool = True) -> float:
        gamma_total = self.gamma_comm if use_comm_gamma_as_total else self.gamma_comp
        return float(self.k * bits + gamma_total)


def load_models(
    compress_path: str,
    cpu_path: str,
    client: Optional[str] = None,
) -> List[ClientModels]:
    """Load both datasets and fit per-client models.

    - compress_path: Excel with columns ['client_id','cpu_size','theta'] where
      'cpu_size' are bits (e.g., 8,16,32).
    - cpu_path: Excel with columns ['client_id','cpu_size','theta'] where
      'cpu_size' are CPU shares (e.g., 0.3, 0.6, 1.0).
    - client: optional substring to filter client_id.
    """
    comp_df = pd.read_excel(compress_path)
    cpu_df = pd.read_excel(cpu_path)

    # Harmonize client sets (keep intersection)
    comp_ids = set(comp_df["client_id"].astype(str).unique())
    cpu_ids = set(cpu_df["client_id"].astype(str).unique())
    ids = sorted(comp_ids & cpu_ids)

    if client is not None:
        ids = [cid for cid in ids if client in str(cid)]
        if not ids:
            raise ValueError(f"No matching client_id contains '{client}'.")

    models: List[ClientModels] = []
    for cid in ids:
        sub_comm = comp_df[comp_df["client_id"].astype(str) == str(cid)]
        sub_cpu = cpu_df[cpu_df["client_id"].astype(str) == str(cid)]

        k, gamma_c = fit_comm_linear(sub_comm)
        a, b, g = fit_comp_power(sub_cpu)
        models.append(ClientModels(str(cid), k, gamma_c, a, b, g))

    return models


def pick_next_bit(current: float, path: Iterable[float]) -> Optional[float]:
    path_list = [float(x) for x in path]
    # choose first value in path that is strictly less than current
    for b in path_list:
        if b < current:
            return b
    return None


def decide_new_settings(
    mdl: ClientModels,
    theta_target: float,
    b_start: float = 32.0,
    cpu_start: float = 1.0,
    tau: float = 0.4,
    mu: float = 0.4,
    bit_path: Tuple[float, ...] = (32.0, 8.0, 4.0),
    cpu_min: float = 0.3,
    cpu_max: float = 1.0,
    use_comm_gamma_as_total: bool = True,
) -> Tuple[float, float, Dict[str, float]]:
    """Implement the control loop described by the user.

    - Start from (b_start, cpu_start).
    - While theta_all > target and conditions hold, step b along bit_path
      (32 -> 8 -> 4). If at b=4 or conditions fail, adjust CPU upward to
      meet the target if possible.
    - Returns (new_bit, new_cpu, info_dict)
    """
    b_curr = float(b_start)
    c_curr = float(cpu_start)

    # Normalize bit path to be relative to current position
    # Build a generator that yields candidates less than current
    def next_bit_after(b: float) -> Optional[float]:
        # path is ordered; take the first value smaller than b
        for x in bit_path:
            if x < b:
                return float(x)
        return None

    steps_taken = 0
    while True:
        theta_all = mdl.theta_all(b_curr, c_curr, use_comm_gamma_as_total)
        if theta_all <= theta_target:
            break

        theta_comm = mdl.theta_comm(b_curr, use_comm_gamma_as_total)
        ratio = theta_comm / theta_all if theta_all > 0 else 0.0

        # Candidate next compression level
        b_next = next_bit_after(b_curr)

        # If no more compression candidate, move to CPU adjustment
        if b_next is None:
            break

        # Expected communication theta reduction from compressing
        delta_comm = mdl.k * (b_curr - b_next)  # positive if k>0 and b_next < b_curr
        need_reduction = theta_all - theta_target

        # Check decision rules
        if (theta_all > theta_target) and (ratio > tau) and (delta_comm > mu * need_reduction):
            b_curr = b_next
            steps_taken += 1
            # Keep looping; may compress again if still above target
            continue
        else:
            # Do not compress further; switch to CPU adjustment
            break

    # If still above target, adjust CPU
    theta_all = mdl.theta_all(b_curr, c_curr, use_comm_gamma_as_total)
    if theta_all > theta_target:
        # Solve for cpu: k*b + alpha*c^{-beta} + gamma = theta_target
        gamma_total = mdl.gamma_comm if use_comm_gamma_as_total else mdl.gamma_comp
        rhs = theta_target - (mdl.k * b_curr + gamma_total)
        # alpha * c^{-beta} = rhs  ->  c = (alpha / rhs)^{1/beta}
        if rhs <= 0 or mdl.alpha <= 0 or mdl.beta <= 0:
            # Not solvable by increasing CPU (or model degenerate). Clamp to max.
            c_req = cpu_max
        else:
            c_req = (mdl.alpha / rhs) ** (1.0 / mdl.beta)
            c_req = float(np.clip(c_req, cpu_min, cpu_max))

        c_curr = max(c_curr, c_req)  # only increase CPU share

    info = {
        "steps": steps_taken,
        "k": mdl.k,
        "gamma": mdl.gamma_comm if use_comm_gamma_as_total else mdl.gamma_comp,
        "alpha": mdl.alpha,
        "beta": mdl.beta,
        "theta_final": mdl.theta_all(b_curr, c_curr, use_comm_gamma_as_total),
        "theta_comm_final": mdl.theta_comm(b_curr, use_comm_gamma_as_total),
        "theta_initial": mdl.theta_all(b_start, cpu_start, use_comm_gamma_as_total),
    }
    return b_curr, c_curr, info


def main() -> None:
    ap = argparse.ArgumentParser(description="Recommend (bit, cpu) to meet a target theta.")
    ap.add_argument("target", type=float, help="Target theta value (θ_target)")
    ap.add_argument("--client", type=str, default=None, help="client_id substring to filter")
    ap.add_argument("--tau", type=float, default=0.4, help="τ threshold for θ_comm/θ_all")
    ap.add_argument("--mu", type=float, default=0.4, help="μ threshold for k(b_old-b_new)")
    ap.add_argument("--b-start", type=float, default=32.0, help="Starting bits (default 32)")
    ap.add_argument("--cpu-start", type=float, default=1.0, help="Starting CPU share (default 1.0)")
    ap.add_argument("--compress-xlsx", type=str, default="compress.xlsx")
    ap.add_argument("--cpu-xlsx", type=str, default="cpudata10.xlsx")
    ap.add_argument(
        "--bit-path",
        type=str,
        default="32,8,4",
        help="Comma-separated compression ladder (descending)",
    )
    ap.add_argument("--cpu-min", type=float, default=0.3)
    ap.add_argument("--cpu-max", type=float, default=1.0)
    ap.add_argument(
        "--gamma-source",
        choices=["comm", "comp"],
        default="comm",
        help="Which gamma to use as the shared constant in θ_all",
    )
    args = ap.parse_args()

    bit_path = tuple(float(x.strip()) for x in args.bit_path.split(",") if x.strip())
    use_comm_gamma = args.gamma_source == "comm"

    models = load_models(args.compress_xlsx, args.cpu_xlsx, args.client)
    if not models:
        raise SystemExit("No clients found in both datasets.")

    # If multiple clients match, show recommendations for all
    for mdl in models:
        b_new, c_new, info = decide_new_settings(
            mdl,
            theta_target=args.target,
            b_start=args.b_start,
            cpu_start=args.cpu_start,
            tau=args.tau,
            mu=args.mu,
            bit_path=bit_path,
            cpu_min=args.cpu_min,
            cpu_max=args.cpu_max,
            use_comm_gamma_as_total=use_comm_gamma,
        )

        print(f"Client {mdl.client_id} -> new_bit={b_new:.0f}, new_cpu={c_new:.3f}")
        print(
            f"  params: k={mdl.k:.4f}, alpha={mdl.alpha:.4f}, beta={mdl.beta:.4f}, gamma={'comm' if use_comm_gamma else 'comp'}={info['gamma']:.4f}"
        )
        print(
            f"  theta: initial={info['theta_initial']:.2f} -> final={info['theta_final']:.2f} (θ_comm={info['theta_comm_final']:.2f})"
        )
        print(
            f"  steps taken: compress={int(info['steps'])}, ladder={list(bit_path)}; tau={args.tau}, mu={args.mu}\n"
        )


if __name__ == "__main__":
    main()

