from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from src.utils.grid import Grid, Point, in_bounds, is_walkable


@dataclass
class JPSResult:
    path: List[Point]
    path_cost: float
    nodes_expanded: int
    found: bool


def sign(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def distance(a: Point, b: Point) -> float:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    diag = min(dr, dc)
    straight = max(dr, dc) - diag
    return diag * math.sqrt(2) + straight


def reconstruct_path(came_from: Dict[Point, Point], current: Point) -> List[Point]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def expand_path(path: List[Point]) -> List[Point]:
    """
    Expand jump points into all intermediate grid cells for visualization.
    """
    if not path:
        return []

    full_path = [path[0]]

    for i in range(1, len(path)):
        r0, c0 = path[i - 1]
        r1, c1 = path[i]

        dr = sign(r1 - r0)
        dc = sign(c1 - c0)

        r, c = r0, c0
        while (r, c) != (r1, c1):
            r += dr
            c += dc
            full_path.append((r, c))

    return full_path


def has_forced_neighbor(grid: Grid, node: Point, direction: Tuple[int, int]) -> bool:
    r, c = node
    dr, dc = direction

    # Diagonal movement
    if dr != 0 and dc != 0:
        if is_walkable(grid, (r - dr, c + dc)) and not is_walkable(grid, (r - dr, c)):
            return True
        if is_walkable(grid, (r + dr, c - dc)) and not is_walkable(grid, (r, c - dc)):
            return True

    # Horizontal movement
    elif dr == 0 and dc != 0:
        if is_walkable(grid, (r + 1, c + dc)) and not is_walkable(grid, (r + 1, c)):
            return True
        if is_walkable(grid, (r - 1, c + dc)) and not is_walkable(grid, (r - 1, c)):
            return True

    # Vertical movement
    elif dr != 0 and dc == 0:
        if is_walkable(grid, (r + dr, c + 1)) and not is_walkable(grid, (r, c + 1)):
            return True
        if is_walkable(grid, (r + dr, c - 1)) and not is_walkable(grid, (r, c - 1)):
            return True

    return False


def natural_neighbors(direction: Tuple[int, int]) -> List[Tuple[int, int]]:
    dr, dc = direction

    if dr != 0 and dc != 0:
        return [(dr, dc), (dr, 0), (0, dc)]
    if dr == 0 and dc != 0:
        return [(0, dc)]
    if dr != 0 and dc == 0:
        return [(dr, 0)]
    return []


def pruned_neighbors(grid: Grid, current: Point, parent: Optional[Point]) -> List[Tuple[int, int]]:
    """
    Return movement directions (dr, dc) to consider from current.
    """
    if parent is None:
        dirs = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        valid = []
        for dr, dc in dirs:
            nxt = (current[0] + dr, current[1] + dc)
            if not is_walkable(grid, nxt):
                continue
            if dr != 0 and dc != 0:
                if not (is_walkable(grid, (current[0] + dr, current[1])) and
                        is_walkable(grid, (current[0], current[1] + dc))):
                    continue
            valid.append((dr, dc))
        return valid

    dr = sign(current[0] - parent[0])
    dc = sign(current[1] - parent[1])
    directions = []

    # Natural neighbors
    for ndr, ndc in natural_neighbors((dr, dc)):
        nxt = (current[0] + ndr, current[1] + ndc)
        if not is_walkable(grid, nxt):
            continue
        if ndr != 0 and ndc != 0:
            if not (is_walkable(grid, (current[0] + ndr, current[1])) and
                    is_walkable(grid, (current[0], current[1] + ndc))):
                continue
        directions.append((ndr, ndc))

    # Forced neighbors
    r, c = current

    if dr != 0 and dc != 0:
        candidates = [(-dr, dc), (dr, -dc)]
        for fdr, fdc in candidates:
            nxt = (r + fdr, c + fdc)
            if is_walkable(grid, nxt):
                directions.append((fdr, fdc))

    elif dr == 0 and dc != 0:
        if not is_walkable(grid, (r + 1, c)) and is_walkable(grid, (r + 1, c + dc)):
            directions.append((1, dc))
        if not is_walkable(grid, (r - 1, c)) and is_walkable(grid, (r - 1, c + dc)):
            directions.append((-1, dc))

    elif dr != 0 and dc == 0:
        if not is_walkable(grid, (r, c + 1)) and is_walkable(grid, (r + dr, c + 1)):
            directions.append((dr, 1))
        if not is_walkable(grid, (r, c - 1)) and is_walkable(grid, (r + dr, c - 1)):
            directions.append((dr, -1))

    # Deduplicate
    out = []
    seen = set()
    for d in directions:
        if d not in seen:
            seen.add(d)
            out.append(d)

    return out


def jump(
    grid: Grid,
    current: Point,
    direction: Tuple[int, int],
    goal: Point,
) -> Optional[Point]:
    r, c = current
    dr, dc = direction

    nr, nc = r + dr, c + dc
    nxt = (nr, nc)

    if not is_walkable(grid, nxt):
        return None

    if dr != 0 and dc != 0:
        if not (is_walkable(grid, (r + dr, c)) and is_walkable(grid, (r, c + dc))):
            return None

    if nxt == goal:
        return nxt

    if has_forced_neighbor(grid, nxt, direction):
        return nxt

    # For diagonal movement, recurse on horizontal/vertical components too
    if dr != 0 and dc != 0:
        if jump(grid, nxt, (dr, 0), goal) is not None:
            return nxt
        if jump(grid, nxt, (0, dc), goal) is not None:
            return nxt

    return jump(grid, nxt, direction, goal)


def identify_successors(
    grid: Grid,
    current: Point,
    parent: Optional[Point],
    goal: Point,
) -> List[Point]:
    successors = []
    for direction in pruned_neighbors(grid, current, parent):
        jp = jump(grid, current, direction, goal)
        if jp is not None:
            successors.append(jp)
    return successors


def jps(
    grid: Grid,
    start: Point,
    goal: Point,
    heuristic_fn: Callable[[Point, Point], float],
) -> JPSResult:
    if not is_walkable(grid, start):
        raise ValueError("Start is not walkable.")
    if not is_walkable(grid, goal):
        raise ValueError("Goal is not walkable.")

    open_heap: List[Tuple[float, int, Point]] = []
    counter = 0

    g_score: Dict[Point, float] = {start: 0.0}
    came_from: Dict[Point, Point] = {}
    parent_map: Dict[Point, Optional[Point]] = {start: None}
    closed = set()

    heapq.heappush(open_heap, (heuristic_fn(start, goal), counter, start))
    nodes_expanded = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        nodes_expanded += 1

        if current == goal:
            jump_path = reconstruct_path(came_from, current)
            full_path = expand_path(jump_path)
            return JPSResult(
                path=full_path,
                path_cost=g_score[current],
                nodes_expanded=nodes_expanded,
                found=True,
            )

        closed.add(current)

        parent = parent_map[current]
        successors = identify_successors(grid, current, parent, goal)

        for succ in successors:
            if succ in closed:
                continue

            tentative_g = g_score[current] + distance(current, succ)

            if tentative_g < g_score.get(succ, float("inf")):
                g_score[succ] = tentative_g
                came_from[succ] = current
                parent_map[succ] = current
                counter += 1
                f = tentative_g + heuristic_fn(succ, goal)
                heapq.heappush(open_heap, (f, counter, succ))

    return JPSResult(
        path=[],
        path_cost=float("inf"),
        nodes_expanded=nodes_expanded,
        found=False,
    )