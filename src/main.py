from __future__ import annotations

from src.algorithms.astar import astar
from src.algorithms.jps import jps
from src.algorithms.theta_star import theta_star
from src.utils.grid import choose_free_start_goal, generate_random_grid, print_grid
from src.utils.heuristics import octile


def main() -> None:
    rows = 10
    cols = 10
    obstacle_prob = 0.2
    seed = 42

    algorithm = "jps"   # change to: "astar", "theta", or "jps"

    grid = generate_random_grid(
        rows=rows,
        cols=cols,
        obstacle_prob=obstacle_prob,
        seed=seed,
    )

    start, goal = choose_free_start_goal(grid, seed=seed)

    if algorithm == "astar":
        result = astar(
            grid=grid,
            start=start,
            goal=goal,
            heuristic_fn=octile,
            allow_diagonal=True,
        )
    elif algorithm == "theta":
        result = theta_star(
            grid=grid,
            start=start,
            goal=goal,
            heuristic_fn=octile,
            allow_diagonal=True,
        )
    elif algorithm == "jps":
        result = jps(
            grid=grid,
            start=start,
            goal=goal,
            heuristic_fn=octile,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    print(f"Algorithm: {algorithm}")
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"Found path: {result.found}")
    print(f"Path cost: {result.path_cost:.3f}")
    print(f"Nodes expanded: {result.nodes_expanded}")
    print()

    print_grid(grid, start=start, goal=goal, path=result.path)


if __name__ == "__main__":
    main()