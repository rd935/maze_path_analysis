from pathlib import Path

# -----------------------------
# Experiment parameter grid
# -----------------------------
GRID_SIZES = [128, 192, 256, 384, 512, 768, 1024]
OBSTACLE_DENSITIES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
SEEDS = list(range(15))

# For quicker pilot tests, temporarily use:
# GRID_SIZES = [128, 256, 512]
# OBSTACLE_DENSITIES = [0.35, 0.40, 0.45]
# SEEDS = list(range(10))

# -----------------------------
# Instance generation
# -----------------------------
MAX_SOLVABLE_ATTEMPTS = 20
MIN_START_GOAL_OCTILE_DISTANCE_FACTOR = 0.25
MAX_CONSECUTIVE_UNSOLVABLE = 10
TARGET_SOLVABLE_MAZES_PER_CONFIG = len(SEEDS)
# start-goal minimum distance = factor * grid size

# -----------------------------
# Output directories
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
SUMMARY_DIR = RESULTS_DIR / "summary"
FIG_DIR = RESULTS_DIR / "figures"

RAW_RESULTS_PATH = RAW_DIR / "results_raw.csv"
SUMMARY_RESULTS_PATH = SUMMARY_DIR / "results_summary.csv"

# RAW_RESULTS_PATH = RAW_DIR / "jps_stress_results_raw.csv"
# SUMMARY_RESULTS_PATH = SUMMARY_DIR / "jps_stress_results_summary.csv"


# -----------------------------
# CSV columns
# -----------------------------
RAW_COLUMNS = [
    "algorithm",
    "grid_rows",
    "grid_cols",
    "obstacle_density",
    "seed",
    "instance_seed",
    "start_row",
    "start_col",
    "goal_row",
    "goal_col",
    "found",
    "runtime_ms",
    "nodes_expanded",
    "path_cost",
]