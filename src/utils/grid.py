from __future__ import annotations

import random
from typing import List, Tuple

Grid = List[List[int]]
Point = Tuple[int, int]


def create_empty_grid(rows: int, cols: int) -> Grid:
    """Create an empty grid with all cells free (0)."""
    return [[0 for _ in range(cols)] for _ in range(rows)]


def generate_random_grid(
    rows: int,
    cols: int,
    obstacle_prob: float = 0.2,
    seed: int | None = None,
) -> Grid:
    """
    Generate a random binary occupancy grid.

    0 = free cell
    1 = obstacle
    """
    if not (0.0 <= obstacle_prob < 1.0):
        raise ValueError("obstacle_prob must be in [0.0, 1.0).")

    rng = random.Random(seed)
    grid = []

    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append(1 if rng.random() < obstacle_prob else 0)
        grid.append(row)

    return grid


def in_bounds(grid: Grid, point: Point) -> bool:
    """Return True if point is inside the grid."""
    r, c = point
    return 0 <= r < len(grid) and 0 <= c < len(grid[0])


def is_walkable(grid: Grid, point: Point) -> bool:
    """Return True if point is inside the grid and not an obstacle."""
    return in_bounds(grid, point) and grid[point[0]][point[1]] == 0


def set_cell(grid: Grid, point: Point, value: int) -> None:
    """Set a grid cell to 0 (free) or 1 (obstacle)."""
    if value not in (0, 1):
        raise ValueError("Grid cell value must be 0 or 1.")
    if not in_bounds(grid, point):
        raise IndexError("Point is out of bounds.")
    grid[point[0]][point[1]] = value


def get_neighbors(grid: Grid, point: Point, allow_diagonal: bool = True) -> List[Point]:
    """
    Return valid neighboring cells.

    If allow_diagonal is True, includes 8-connected motion.
    Prevents diagonal corner-cutting through blocked cells.
    """
    r, c = point

    cardinal_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    diagonal_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    neighbors: List[Point] = []

    for dr, dc in cardinal_dirs:
        nxt = (r + dr, c + dc)
        if is_walkable(grid, nxt):
            neighbors.append(nxt)

    if allow_diagonal:
        for dr, dc in diagonal_dirs:
            nxt = (r + dr, c + dc)
            if not is_walkable(grid, nxt):
                continue

            # Prevent diagonal corner cutting
            side1 = (r + dr, c)
            side2 = (r, c + dc)
            if is_walkable(grid, side1) and is_walkable(grid, side2):
                neighbors.append(nxt)

    return neighbors


def choose_free_start_goal(
    grid: Grid,
    seed: int | None = None,
) -> Tuple[Point, Point]:
    """Randomly choose distinct free start and goal cells."""
    rng = random.Random(seed)

    free_cells = [
        (r, c)
        for r in range(len(grid))
        for c in range(len(grid[0]))
        if grid[r][c] == 0
    ]

    if len(free_cells) < 2:
        raise ValueError("Need at least two free cells for start and goal.")

    start = rng.choice(free_cells)
    goal = rng.choice(free_cells)
    while goal == start:
        goal = rng.choice(free_cells)

    return start, goal


def print_grid(
    grid: Grid,
    start: Point | None = None,
    goal: Point | None = None,
    path: List[Point] | None = None,
) -> None:
    """
    Print a simple ASCII version of the grid.

    Symbols:
      . = free
      # = obstacle
      S = start
      G = goal
      * = path
    """
    path_set = set(path or [])

    for r in range(len(grid)):
        line = []
        for c in range(len(grid[0])):
            p = (r, c)
            if start is not None and p == start:
                line.append("S")
            elif goal is not None and p == goal:
                line.append("G")
            elif p in path_set:
                line.append("*")
            elif grid[r][c] == 1:
                line.append("#")
            else:
                line.append(".")
        print(" ".join(line))