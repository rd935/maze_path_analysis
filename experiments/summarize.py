from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow imports from project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.config import (  # noqa: E402
    RAW_RESULTS_PATH,
    SUMMARY_DIR,
    SUMMARY_RESULTS_PATH,
)


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Raw results file not found: {RAW_RESULTS_PATH}")

    df = pd.read_csv(RAW_RESULTS_PATH)

    # Normalize found column in case CSV stores it as strings
    if df["found"].dtype == object:
        df["found"] = df["found"].astype(str).str.lower().map(
            {"true": True, "false": False}
        )

    df_ok = df[df["found"] == True].copy()  # noqa: E712

    if df_ok.empty:
        raise ValueError("No successful runs found in raw results.")

    summary = (
        df_ok.groupby(["algorithm", "grid_rows", "grid_cols", "obstacle_density"])
        .agg(
            mean_runtime_ms=("runtime_ms", "mean"),
            std_runtime_ms=("runtime_ms", "std"),
            mean_nodes_expanded=("nodes_expanded", "mean"),
            std_nodes_expanded=("nodes_expanded", "std"),
            mean_path_cost=("path_cost", "mean"),
            std_path_cost=("path_cost", "std"),
            num_runs=("runtime_ms", "count"),
        )
        .reset_index()
        .sort_values(["algorithm", "grid_rows", "obstacle_density"])
    )

    summary.to_csv(SUMMARY_RESULTS_PATH, index=False)

    print(f"Saved summary results to: {SUMMARY_RESULTS_PATH}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()