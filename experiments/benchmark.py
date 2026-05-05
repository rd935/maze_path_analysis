from __future__ import annotations

import csv
import math
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

# Allow imports from project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.config import (  # noqa: E402
    GRID_SIZES,
    MAX_SOLVABLE_ATTEMPTS,
    MIN_START_GOAL_OCTILE_DISTANCE_FACTOR,
    OBSTACLE_DENSITIES,
    RAW_COLUMNS,
    RAW_DIR,
    RAW_RESULTS_PATH,
    SEEDS,
)
from src.algorithms.astar import astar  
from src.algorithms.jps import jps  
from src.algorithms.theta_star import theta_star  
from src.utils.grid import is_walkable  

Point = Tuple[int, int]


def octile_distance(a: Point, b: Point) -> float:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    diag = min(dr, dc)
    straight = max(dr, dc) - diag
    return diag * math.sqrt(2) + straight


def generate_random_grid(rows: int, cols: int, obstacle_density: float, seed: int):
    rng = random.Random(seed)
    grid = []

    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append(1 if rng.random() < obstacle_density else 0)
        grid.append(row)

    return grid


def get_random_free_cell(grid, rng: random.Random) -> Point:
    rows = len(grid)
    cols = len(grid[0])

    while True:
        r = rng.randrange(rows)
        c = rng.randrange(cols)

        if is_walkable(grid, (r, c)):
            return (r, c)


def select_start_goal(grid, seed: int) -> Tuple[Point, Point]:
    rng = random.Random(seed + 99991)
    rows = len(grid)
    cols = len(grid[0])

    min_required = MIN_START_GOAL_OCTILE_DISTANCE_FACTOR * min(rows, cols)

    for _ in range(1000):
        start = get_random_free_cell(grid, rng)
        goal = get_random_free_cell(grid, rng)

        if goal == start:
            continue

        if octile_distance(start, goal) >= min_required:
            return start, goal

    start = get_random_free_cell(grid, rng)
    goal = get_random_free_cell(grid, rng)

    while goal == start:
        goal = get_random_free_cell(grid, rng)

    return start, goal


def ensure_start_goal_free(grid, start: Point, goal: Point) -> None:
    grid[start[0]][start[1]] = 0
    grid[goal[0]][goal[1]] = 0


def build_solvable_instance(
    rows: int,
    cols: int,
    density: float,
    seed: int,
    max_tries: int = MAX_SOLVABLE_ATTEMPTS,
):
    for k in range(max_tries):
        instance_seed = seed * 1000 + k

        grid = generate_random_grid(rows, cols, density, instance_seed)
        start, goal = select_start_goal(grid, instance_seed)
        ensure_start_goal_free(grid, start, goal)

        try:
            result = astar(grid, start, goal, octile_distance)

            if result.found:
                return grid, start, goal, instance_seed

        except Exception:
            pass

    return None, None, None, None


def timed_run(
    algorithm_name: str,
    fn: Callable,
    grid,
    start: Point,
    goal: Point,
) -> Dict:
    t0 = time.perf_counter()
    result = fn(grid, start, goal, octile_distance)
    t1 = time.perf_counter()

    return {
        "algorithm": algorithm_name,
        "found": bool(result.found),
        "runtime_ms": (t1 - t0) * 1000.0,
        "nodes_expanded": int(result.nodes_expanded),
        "path_cost": float(result.path_cost),
    }


def ensure_csv_exists(path: Path) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
            writer.writeheader()


def append_row(path: Path, row: Dict) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        writer.writerow(row)


def run_all_experiments() -> None:
    ensure_csv_exists(RAW_RESULTS_PATH)

    algorithms = [
        ("astar", astar),
        ("theta_star", theta_star),
        ("jps", jps),
    ]

    target_solvable_per_config = len(SEEDS)
    max_consecutive_unsolvable = 10

    total_configs = len(GRID_SIZES) * len(OBSTACLE_DENSITIES)
    total_expected_mazes = total_configs * target_solvable_per_config

    config_idx = 0
    global_maze_counter = 0

    print(f"Writing raw results to: {RAW_RESULTS_PATH}")
    print(f"Expected maximum solvable mazes: {total_expected_mazes}")
    print(f"Total size/density configurations: {total_configs}")

    for size in GRID_SIZES:
        for density in OBSTACLE_DENSITIES:
            config_idx += 1

            print("\n" + "=" * 70)
            print(
                f"[Config {config_idx}/{total_configs}] "
                f"size={size}x{size}, density={density:.2f}"
            )
            print("=" * 70)

            solvable_count = 0
            attempt_seed = 0
            consecutive_unsolvable = 0

            while solvable_count < target_solvable_per_config:
                print(
                    f"\n  Searching for solvable maze "
                    f"{solvable_count + 1}/{target_solvable_per_config} "
                    f"for this config..."
                )
                print(
                    f"  Attempt seed={attempt_seed}, "
                    f"consecutive_unsolvable={consecutive_unsolvable}/"
                    f"{max_consecutive_unsolvable}"
                )

                grid, start, goal, instance_seed = build_solvable_instance(
                    rows=size,
                    cols=size,
                    density=density,
                    seed=attempt_seed,
                )

                attempt_seed += 1

                if grid is None:
                    consecutive_unsolvable += 1

                    print(
                        f"  No solvable maze found for this attempt. "
                        f"Consecutive failures: {consecutive_unsolvable}/"
                        f"{max_consecutive_unsolvable}"
                    )

                    if consecutive_unsolvable >= max_consecutive_unsolvable:
                        print(
                            f"  Skipping config {config_idx}/{total_configs} "
                            f"after {max_consecutive_unsolvable} consecutive "
                            f"unsolvable attempts."
                        )
                        break

                    continue

                consecutive_unsolvable = 0
                global_maze_counter += 1

                print(
                    f"\n  [Maze {global_maze_counter}/{total_expected_mazes}] "
                    f"(Config {config_idx}/{total_configs}) "
                    f"[{size}x{size}, density={density:.2f}]"
                )
                print(
                    f"  Solvable instance #{solvable_count + 1}/"
                    f"{target_solvable_per_config}"
                )
                print(
                    f"  instance_seed={instance_seed}, "
                    f"start={start}, goal={goal}"
                )

                for algo_name, algo_fn in algorithms:
                    try:
                        result = timed_run(algo_name, algo_fn, grid, start, goal)

                        row = {
                            "algorithm": algo_name,
                            "grid_rows": size,
                            "grid_cols": size,
                            "obstacle_density": density,
                            "seed": solvable_count,
                            "instance_seed": instance_seed,
                            "start_row": start[0],
                            "start_col": start[1],
                            "goal_row": goal[0],
                            "goal_col": goal[1],
                            "found": result["found"],
                            "runtime_ms": result["runtime_ms"],
                            "nodes_expanded": result["nodes_expanded"],
                            "path_cost": result["path_cost"],
                        }

                        append_row(RAW_RESULTS_PATH, row)

                        print(
                            f"  {algo_name:10s} | found={result['found']} "
                            f"| runtime={result['runtime_ms']:.3f} ms "
                            f"| expanded={result['nodes_expanded']} "
                            f"| cost={result['path_cost']:.3f}"
                        )

                    except Exception as e:
                        print(f"  {algo_name:10s} | ERROR: {e}")

                solvable_count += 1

            print(
                f"\nFinished config {config_idx}/{total_configs}: "
                f"{solvable_count}/{target_solvable_per_config} "
                f"solvable mazes completed."
            )

    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print(f"Total solvable mazes completed: {global_maze_counter}")
    print(f"Raw results saved to: {RAW_RESULTS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    run_all_experiments()