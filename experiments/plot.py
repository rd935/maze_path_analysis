from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Allow imports from project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.config import FIG_DIR, SUMMARY_RESULTS_PATH  # noqa: E402


def plot_runtime_vs_size(df: pd.DataFrame) -> None:
    for density in sorted(df["obstacle_density"].unique()):
        sub = df[df["obstacle_density"] == density]

        plt.figure(figsize=(8, 5))
        for algo in sorted(sub["algorithm"].unique()):
            s = sub[sub["algorithm"] == algo].sort_values("grid_rows")
            plt.errorbar(
                s["grid_rows"],
                s["mean_runtime_ms"],
                yerr=s["std_runtime_ms"],
                marker="o",
                capsize=4,
                label=algo,
            )

        plt.xlabel("Grid size (N for N x N)")
        plt.ylabel("Runtime (ms)")
        plt.title(f"Runtime vs Grid Size at Obstacle Density = {density:.2f}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_path = FIG_DIR / f"runtime_vs_size_density_{density:.2f}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved {out_path}")


def plot_expansions_vs_density(df: pd.DataFrame) -> None:
    for size in sorted(df["grid_rows"].unique()):
        sub = df[df["grid_rows"] == size]

        plt.figure(figsize=(8, 5))
        for algo in sorted(sub["algorithm"].unique()):
            s = sub[sub["algorithm"] == algo].sort_values("obstacle_density")
            plt.errorbar(
                s["obstacle_density"],
                s["mean_nodes_expanded"],
                yerr=s["std_nodes_expanded"],
                marker="o",
                capsize=4,
                label=algo,
            )

        plt.xlabel("Obstacle density")
        plt.ylabel("Nodes expanded")
        plt.title(f"Nodes Expanded vs Obstacle Density at Grid Size = {size}x{size}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_path = FIG_DIR / f"expanded_vs_density_size_{size}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved {out_path}")


def plot_runtime_vs_density(df: pd.DataFrame) -> None:
    for size in sorted(df["grid_rows"].unique()):
        sub = df[df["grid_rows"] == size]

        plt.figure(figsize=(8, 5))
        for algo in sorted(sub["algorithm"].unique()):
            s = sub[sub["algorithm"] == algo].sort_values("obstacle_density")
            plt.errorbar(
                s["obstacle_density"],
                s["mean_runtime_ms"],
                yerr=s["std_runtime_ms"],
                marker="o",
                capsize=4,
                label=algo,
            )

        plt.xlabel("Obstacle density")
        plt.ylabel("Runtime (ms)")
        plt.title(f"Runtime vs Obstacle Density at Grid Size = {size}x{size}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_path = FIG_DIR / f"runtime_vs_density_size_{size}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved {out_path}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not SUMMARY_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Summary results file not found: {SUMMARY_RESULTS_PATH}"
        )

    df = pd.read_csv(SUMMARY_RESULTS_PATH)

    plot_runtime_vs_size(df)
    plot_expansions_vs_density(df)
    plot_runtime_vs_density(df)


if __name__ == "__main__":
    main()