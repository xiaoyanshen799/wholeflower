#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt


LINE_RE = re.compile(
    r"theta\s*=\s*([0-9.]+)\s*,\s*k\s*=\s*([0-9.]+)",
    re.IGNORECASE,
)


def parse_theta_k(path: Path) -> tuple[np.ndarray, np.ndarray]:
    thetas = []
    ks = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            m = LINE_RE.search(raw)
            if not m:
                continue
            theta = float(m.group(1))
            k = float(m.group(2))
            thetas.append(theta)
            ks.append(k)
    if not thetas:
        raise SystemExit(f"No theta/k pairs found in {path}")
    return np.array(thetas), np.array(ks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scatter plot of theta vs k and linear fit."
    )
    parser.add_argument(
        "log",
        type=Path,
        help="Path to mnist_cpu_theta.log",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs/theta_k_fit.png"),
        help="Output image path.",
    )
    args = parser.parse_args()

    theta, k = parse_theta_k(args.log)

    # Linear fit: k = a * theta + b
    a, b = np.polyfit(theta, k, 1)
    a = float(a)
    b = float(b)
    k_hat = a * theta + b

    # Plot
    plt.figure(figsize=(7, 5))
    plt.scatter(theta, k, s=20, alpha=0.7, label="data")
    x_line = np.linspace(theta.min(), theta.max(), 200)
    y_line = a * x_line + b
    plt.plot(x_line, y_line, "r--", linewidth=2, label=f"fit: k={a:.4f}*θ+{b:.4f}")
    plt.xlabel("theta")
    plt.ylabel("k")
    plt.title("Theta vs k (linear fit)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
