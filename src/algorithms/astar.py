from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from src.utils.grid import Grid, Point, get_neighbors, is_walkable


@dataclass
class AStarResult:
    path: List[Point]
    path_cost: float
    nodes_expanded: int
    found: bool


def reconstruct_path(came_from: Dict[Point, Point], current: Point) -> List[Point]:
    """Reconstruct path from goal back to start."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def movement_cost(a: Point, b: Point) -> float:
    """
    Cost of moving from a to b.
    Cardinal move cost = 1
    Diagonal move cost = sqrt(2)
    """
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])

    if dr + dc == 1:
        return 1.0
    return math.sqrt(2)


def astar(
    grid: Grid,
    start: Point,
    goal: Point,
    heuristic_fn: Callable[[Point, Point], float],
    allow_diagonal: bool = True,
) -> AStarResult:
    """
    Run A* search on a grid.

    Returns:
        AStarResult containing path, cost, expanded nodes, and found flag.
    """
    if not is_walkable(grid, start):
        raise ValueError("Start is not walkable.")
    if not is_walkable(grid, goal):
        raise ValueError("Goal is not walkable.")

    open_heap: List[Tuple[float, int, Point]] = []
    heap_counter = 0

    g_score: Dict[Point, float] = {start: 0.0}
    came_from: Dict[Point, Point] = {}
    closed_set = set()

    start_f = heuristic_fn(start, goal)
    heapq.heappush(open_heap, (start_f, heap_counter, start))

    nodes_expanded = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)

        if current in closed_set:
            continue

        nodes_expanded += 1

        if current == goal:
            path = reconstruct_path(came_from, current)
            return AStarResult(
                path=path,
                path_cost=g_score[current],
                nodes_expanded=nodes_expanded,
                found=True,
            )

        closed_set.add(current)

        for neighbor in get_neighbors(grid, current, allow_diagonal=allow_diagonal):
            if neighbor in closed_set:
                continue

            tentative_g = g_score[current] + movement_cost(current, neighbor)

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic_fn(neighbor, goal)

                heap_counter += 1
                heapq.heappush(open_heap, (f_score, heap_counter, neighbor))

    return AStarResult(
        path=[],
        path_cost=float("inf"),
        nodes_expanded=nodes_expanded,
        found=False,
    )