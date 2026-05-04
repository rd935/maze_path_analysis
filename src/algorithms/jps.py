from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from src.utils.grid import Grid, Point, is_walkable


@dataclass
class JPSResult:
    path: List[Point]
    path_cost: float
    nodes_expanded: int
    found: bool


SQRT2 = math.sqrt(2.0)

DIRS_8 = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
]


def sign(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def octile_distance(a: Point, b: Point) -> float:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    diag = min(dr, dc)
    straight = max(dr, dc) - diag
    return diag * SQRT2 + straight


def step_cost(direction: Tuple[int, int]) -> float:
    dr, dc = direction
    return SQRT2 if dr != 0 and dc != 0 else 1.0


def valid_move(grid: Grid, current: Point, direction: Tuple[int, int]) -> bool:
    r, c = current
    dr, dc = direction
    nxt = (r + dr, c + dc)

    if not is_walkable(grid, nxt):
        return False

    # no corner cutting
    if dr != 0 and dc != 0:
        if not is_walkable(grid, (r + dr, c)):
            return False
        if not is_walkable(grid, (r, c + dc)):
            return False

    return True


def reconstruct_jump_path(came_from: Dict[Point, Point], current: Point) -> List[Point]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def expand_jump_path(path: List[Point]) -> List[Point]:
    if not path:
        return []

    full = [path[0]]
    for i in range(1, len(path)):
        r0, c0 = path[i - 1]
        r1, c1 = path[i]

        dr = sign(r1 - r0)
        dc = sign(c1 - c0)

        r, c = r0, c0
        while (r, c) != (r1, c1):
            r += dr
            c += dc
            full.append((r, c))

    return full


def all_valid_neighbors(grid: Grid, current: Point) -> List[Tuple[int, int]]:
    return [d for d in DIRS_8 if valid_move(grid, current, d)]


def prune_neighbors(
    grid: Grid,
    current: Point,
    parent: Optional[Point],
) -> List[Tuple[int, int]]:
    """
    Conservative pruning.
    If parent is None -> all valid directions.
    Otherwise use a light pruning rule, but do NOT rely on it for completeness.
    """
    if parent is None:
        return all_valid_neighbors(grid, current)

    r, c = current
    pr, pc = parent
    dr = sign(r - pr)
    dc = sign(c - pc)

    neighbors: List[Tuple[int, int]] = []

    # Always allow forward continuation
    if valid_move(grid, current, (dr, dc)):
        neighbors.append((dr, dc))

    # For diagonal motion, also allow straight components
    if dr != 0 and dc != 0:
        if valid_move(grid, current, (dr, 0)):
            neighbors.append((dr, 0))
        if valid_move(grid, current, (0, dc)):
            neighbors.append((0, dc))

    # Conservative forced-turn additions
    for d in DIRS_8:
        if d not in neighbors and valid_move(grid, current, d):
            nr, nc = current[0] + d[0], current[1] + d[1]
            # Prefer neighbors that are near obstacles / possible turning points
            if any(
                not is_walkable(grid, (nr + ar, nc + ac))
                for ar, ac in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            ):
                neighbors.append(d)

    # If pruning got too aggressive, fall back
    if not neighbors:
        neighbors = all_valid_neighbors(grid, current)

    # deduplicate
    out: List[Tuple[int, int]] = []
    seen = set()
    for d in neighbors:
        if d not in seen:
            seen.add(d)
            out.append(d)

    return out


def has_forced_neighbor(grid: Grid, node: Point, direction: Tuple[int, int]) -> bool:
    r, c = node
    dr, dc = direction

    # Horizontal
    if dr == 0 and dc != 0:
        if not is_walkable(grid, (r + 1, c)) and is_walkable(grid, (r + 1, c + dc)):
            return True
        if not is_walkable(grid, (r - 1, c)) and is_walkable(grid, (r - 1, c + dc)):
            return True

    # Vertical
    elif dr != 0 and dc == 0:
        if not is_walkable(grid, (r, c + 1)) and is_walkable(grid, (r + dr, c + 1)):
            return True
        if not is_walkable(grid, (r, c - 1)) and is_walkable(grid, (r + dr, c - 1)):
            return True

    # Diagonal
    else:
        if not is_walkable(grid, (r - dr, c)) and is_walkable(grid, (r - dr, c + dc)):
            return True
        if not is_walkable(grid, (r, c - dc)) and is_walkable(grid, (r + dr, c - dc)):
            return True

    return False


def jump(
    grid: Grid,
    current: Point,
    direction: Tuple[int, int],
    goal: Point,
) -> Optional[Point]:
    dr, dc = direction
    r, c = current

    while True:
        if not valid_move(grid, (r, c), direction):
            return None

        r += dr
        c += dc
        node = (r, c)

        if node == goal:
            return node

        if has_forced_neighbor(grid, node, direction):
            return node

        if dr != 0 and dc != 0:
            if jump(grid, node, (dr, 0), goal) is not None:
                return node
            if jump(grid, node, (0, dc), goal) is not None:
                return node


def identify_successors(grid, current, parent, goal):
    successors = []

    dirs = all_valid_neighbors(grid, current)

    for direction in dirs:
        # Always add the immediate neighbor for completeness
        nr = current[0] + direction[0]
        nc = current[1] + direction[1]
        successors.append((nr, nc))

        # Also add the jump point if one exists
        jp = jump(grid, current, direction, goal)
        if jp is not None:
            successors.append(jp)

    out = []
    seen = set()
    for s in successors:
        if s not in seen:
            seen.add(s)
            out.append(s)

    return out

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
            jump_path = reconstruct_jump_path(came_from, current)
            full_path = expand_jump_path(jump_path)
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

            move_cost = octile_distance(current, succ)
            tentative_g = g_score[current] + move_cost

            if tentative_g < g_score.get(succ, float("inf")):
                g_score[succ] = tentative_g
                came_from[succ] = current
                parent_map[succ] = current

                counter += 1
                f_score = tentative_g + heuristic_fn(succ, goal)
                heapq.heappush(open_heap, (f_score, counter, succ))

    return JPSResult(
        path=[],
        path_cost=float("inf"),
        nodes_expanded=nodes_expanded,
        found=False,
    )