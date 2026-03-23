from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from src.utils.grid import Grid, Point, get_neighbors, is_walkable


@dataclass
class ThetaStarResult:
    path: List[Point]
    path_cost: float
    nodes_expanded: int
    found: bool


def euclidean_cost(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def reconstruct_path(came_from: Dict[Point, Point], current: Point) -> List[Point]:
    path = [current]
    while came_from[current] != current:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def line_of_sight(grid: Grid, a: Point, b: Point) -> bool:
    """
    Bresenham-style line-of-sight check between two grid cells.
    Returns True if the straight segment between a and b does not
    pass through blocked cells.
    """
    x0, y0 = a
    x1, y1 = b

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1

    err = dx - dy
    x, y = x0, y0

    while True:
        if not is_walkable(grid, (x, y)):
            return False
        if (x, y) == (x1, y1):
            return True

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x += sx

        if e2 < dx:
            err += dx
            y += sy


def theta_star(
    grid: Grid,
    start: Point,
    goal: Point,
    heuristic_fn: Callable[[Point, Point], float],
    allow_diagonal: bool = True,
) -> ThetaStarResult:
    if not is_walkable(grid, start):
        raise ValueError("Start is not walkable.")
    if not is_walkable(grid, goal):
        raise ValueError("Goal is not walkable.")

    open_heap: List[Tuple[float, int, Point]] = []
    counter = 0

    g_score: Dict[Point, float] = {start: 0.0}
    parent: Dict[Point, Point] = {start: start}
    closed_set = set()

    heapq.heappush(open_heap, (heuristic_fn(start, goal), counter, start))
    nodes_expanded = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)

        if current in closed_set:
            continue

        nodes_expanded += 1

        if current == goal:
            path = reconstruct_path(parent, current)
            return ThetaStarResult(
                path=path,
                path_cost=g_score[current],
                nodes_expanded=nodes_expanded,
                found=True,
            )

        closed_set.add(current)

        for neighbor in get_neighbors(grid, current, allow_diagonal=allow_diagonal):
            if neighbor in closed_set:
                continue

            if neighbor not in g_score:
                g_score[neighbor] = float("inf")
                parent[neighbor] = current

            p = parent[current]

            # Standard Theta* relaxation:
            # Try connecting neighbor directly to current's parent
            if line_of_sight(grid, p, neighbor):
                tentative_g = g_score[p] + euclidean_cost(p, neighbor)
                if tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    parent[neighbor] = p
                    counter += 1
                    f = tentative_g + heuristic_fn(neighbor, goal)
                    heapq.heappush(open_heap, (f, counter, neighbor))
            else:
                tentative_g = g_score[current] + euclidean_cost(current, neighbor)
                if tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    parent[neighbor] = current
                    counter += 1
                    f = tentative_g + heuristic_fn(neighbor, goal)
                    heapq.heappush(open_heap, (f, counter, neighbor))

    return ThetaStarResult(
        path=[],
        path_cost=float("inf"),
        nodes_expanded=nodes_expanded,
        found=False,
    )