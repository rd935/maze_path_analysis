from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow imports from project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.config import FIG_DIR, SUMMARY_RESULTS_PATH  # noqa: E402


def smooth_line(x, y, num_points=200):
    """
    Smooths the plotted line using interpolation.
    Falls back to normal line if there are too few points.
    """
    x = np.array(x)
    y = np.array(y)

    if len(x) < 3:
        return x, y

    try:
        from scipy.interpolate import make_interp_spline

        x_smooth = np.linspace(x.min(), x.max(), num_points)
        y_smooth = make_interp_spline(x, y, k=2)(x_smooth)
        return x_smooth, y_smooth

    except Exception:
        # fallback if scipy is not installed
        y_smooth = pd.Series(y).rolling(
            window=3,
            center=True,
            min_periods=1
        ).mean()
        return x, y_smooth


def plot_runtime_vs_size(df: pd.DataFrame) -> None:
    for density in sorted(df["obstacle_density"].unique()):
        sub = df[df["obstacle_density"] == density]

        plt.figure(figsize=(8, 5))

        for algo in sorted(sub["algorithm"].unique()):
            s = sub[sub["algorithm"] == algo].sort_values("grid_rows")

            x = s["grid_rows"].to_numpy()
            y = s["mean_runtime_ms"].to_numpy()

            x_smooth, y_smooth = smooth_line(x, y)

            plt.plot(x_smooth, y_smooth, linewidth=2.5, label=algo)
            plt.scatter(x, y, s=35)

        plt.xlabel("Grid size (N for N x N)")
        plt.ylabel("Mean runtime (ms)")
        plt.title(f"Runtime vs Grid Size at Obstacle Density = {density:.2f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = FIG_DIR / f"runtime_vs_size_density_{density:.2f}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()


def plot_runtime_vs_density(df: pd.DataFrame) -> None:
    for grid_size in sorted(df["grid_rows"].unique()):
        sub = df[df["grid_rows"] == grid_size]

        plt.figure(figsize=(8, 5))

        for algo in sorted(sub["algorithm"].unique()):
            s = sub[sub["algorithm"] == algo].sort_values("obstacle_density")

            x = s["obstacle_density"].to_numpy()
            y = s["mean_runtime_ms"].to_numpy()

            x_smooth, y_smooth = smooth_line(x, y)

            plt.plot(x_smooth, y_smooth, linewidth=2.5, label=algo)
            plt.scatter(x, y, s=35)

        plt.xlabel("Obstacle density")
        plt.ylabel("Mean runtime (ms)")
        plt.title(f"Runtime vs Obstacle Density for {grid_size}x{grid_size} Grid")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = FIG_DIR / f"runtime_vs_density_size_{grid_size}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()


def plot_nodes_vs_size(df: pd.DataFrame) -> None:
    for density in sorted(df["obstacle_density"].unique()):
        sub = df[df["obstacle_density"] == density]

        plt.figure(figsize=(8, 5))

        for algo in sorted(sub["algorithm"].unique()):
            s = sub[sub["algorithm"] == algo].sort_values("grid_rows")

            x = s["grid_rows"].to_numpy()
            y = s["mean_nodes_expanded"].to_numpy()

            x_smooth, y_smooth = smooth_line(x, y)

            plt.plot(x_smooth, y_smooth, linewidth=2.5, label=algo)
            plt.scatter(x, y, s=35)

        plt.xlabel("Grid size (N for N x N)")
        plt.ylabel("Mean nodes expanded")
        plt.title(f"Nodes Expanded vs Grid Size at Obstacle Density = {density:.2f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = FIG_DIR / f"nodes_vs_size_density_{density:.2f}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()


def plot_nodes_vs_density(df: pd.DataFrame) -> None:
    for grid_size in sorted(df["grid_rows"].unique()):
        sub = df[df["grid_rows"] == grid_size]

        plt.figure(figsize=(8, 5))

        for algo in sorted(sub["algorithm"].unique()):
            s = sub[sub["algorithm"] == algo].sort_values("obstacle_density")

            x = s["obstacle_density"].to_numpy()
            y = s["mean_nodes_expanded"].to_numpy()

            x_smooth, y_smooth = smooth_line(x, y)

            plt.plot(x_smooth, y_smooth, linewidth=2.5, label=algo)
            plt.scatter(x, y, s=35)

        plt.xlabel("Obstacle density")
        plt.ylabel("Mean nodes expanded")
        plt.title(f"Nodes Expanded vs Obstacle Density for {grid_size}x{grid_size} Grid")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = FIG_DIR / f"nodes_vs_density_size_{grid_size}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

def plot_avg_runtime_vs_size(df: pd.DataFrame) -> None:
    grouped = (
        df.groupby(["algorithm", "grid_rows"], as_index=False)
        ["mean_runtime_ms"]
        .mean()
    )

    plt.figure(figsize=(8, 5))

    for algo in sorted(grouped["algorithm"].unique()):
        s = grouped[grouped["algorithm"] == algo].sort_values("grid_rows")

        x = s["grid_rows"].to_numpy()
        y = s["mean_runtime_ms"].to_numpy()

        x_smooth, y_smooth = smooth_line(x, y)

        plt.plot(x_smooth, y_smooth, linewidth=2.5, label=algo)
        plt.scatter(x, y, s=35)

    plt.xlabel("Grid size (N for N x N)")
    plt.ylabel("Average mean runtime (ms)")
    plt.title("Average Runtime vs Grid Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = FIG_DIR / "avg_runtime_vs_size.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_avg_runtime_vs_density(df: pd.DataFrame) -> None:
    grouped = (
        df.groupby(["algorithm", "obstacle_density"], as_index=False)
        ["mean_runtime_ms"]
        .mean()
    )

    plt.figure(figsize=(8, 5))

    for algo in sorted(grouped["algorithm"].unique()):
        s = grouped[grouped["algorithm"] == algo].sort_values("obstacle_density")

        x = s["obstacle_density"].to_numpy()
        y = s["mean_runtime_ms"].to_numpy()

        x_smooth, y_smooth = smooth_line(x, y)

        plt.plot(x_smooth, y_smooth, linewidth=2.5, label=algo)
        plt.scatter(x, y, s=35)

    plt.xlabel("Obstacle density")
    plt.ylabel("Average mean runtime (ms)")
    plt.title("Average Runtime vs Obstacle Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = FIG_DIR / "avg_runtime_vs_density.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_avg_nodes_vs_size(df: pd.DataFrame) -> None:
    grouped = (
        df.groupby(["algorithm", "grid_rows"], as_index=False)
        ["mean_nodes_expanded"]
        .mean()
    )

    plt.figure(figsize=(8, 5))

    for algo in sorted(grouped["algorithm"].unique()):
        s = grouped[grouped["algorithm"] == algo].sort_values("grid_rows")

        x = s["grid_rows"].to_numpy()
        y = s["mean_nodes_expanded"].to_numpy()

        x_smooth, y_smooth = smooth_line(x, y)

        plt.plot(x_smooth, y_smooth, linewidth=2.5, label=algo)
        plt.scatter(x, y, s=35)

    plt.xlabel("Grid size (N for N x N)")
    plt.ylabel("Average mean nodes expanded")
    plt.title("Average Nodes Expanded vs Grid Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = FIG_DIR / "avg_nodes_vs_size.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_avg_nodes_vs_density(df: pd.DataFrame) -> None:
    grouped = (
        df.groupby(["algorithm", "obstacle_density"], as_index=False)
        ["mean_nodes_expanded"]
        .mean()
    )

    plt.figure(figsize=(8, 5))

    for algo in sorted(grouped["algorithm"].unique()):
        s = grouped[grouped["algorithm"] == algo].sort_values("obstacle_density")

        x = s["obstacle_density"].to_numpy()
        y = s["mean_nodes_expanded"].to_numpy()

        x_smooth, y_smooth = smooth_line(x, y)

        plt.plot(x_smooth, y_smooth, linewidth=2.5, label=algo)
        plt.scatter(x, y, s=35)

    plt.xlabel("Obstacle density")
    plt.ylabel("Average mean nodes expanded")
    plt.title("Average Nodes Expanded vs Obstacle Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = FIG_DIR / "avg_nodes_vs_density.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SUMMARY_RESULTS_PATH)

    # Only plot successful runs if your summary includes failed cases
    if "success_rate" in df.columns:
        df = df[df["success_rate"] > 0]

    plot_avg_runtime_vs_size(df)
    plot_avg_runtime_vs_density(df)

    plot_avg_nodes_vs_size(df)
    plot_avg_nodes_vs_density(df)

    plot_runtime_vs_size(df)
    plot_runtime_vs_density(df)
    plot_nodes_vs_size(df)
    plot_nodes_vs_density(df)

    print(f"Figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()