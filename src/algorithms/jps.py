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


def sign(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def octile_step_distance(a: Point, b: Point) -> float:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    diag = min(dr, dc)
    straight = max(dr, dc) - diag
    return diag * math.sqrt(2) + straight


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


def valid_move(grid: Grid, current: Point, direction: Tuple[int, int]) -> bool:
    r, c = current
    dr, dc = direction
    nxt = (r + dr, c + dc)

    if not is_walkable(grid, nxt):
        return False

    # Prevent diagonal corner cutting
    if dr != 0 and dc != 0:
        if not is_walkable(grid, (r + dr, c)):
            return False
        if not is_walkable(grid, (r, c + dc)):
            return False

    return True


def pruned_directions(grid: Grid, current: Point, parent: Optional[Point]) -> List[Tuple[int, int]]:
    """
    JPS pruning adapted for 8-connected grids with NO corner cutting.

    Key idea:
    - diagonal parent move -> keep diagonal + its two straight components
    - straight parent move -> keep straight continuation
      and allow side turns only when they become newly available
      after passing an obstacle boundary
    """
    if parent is None:
        dirs = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
        return [d for d in dirs if valid_move(grid, current, d)]

    r, c = current
    pr, pc = parent
    dr = sign(r - pr)
    dc = sign(c - pc)

    directions: List[Tuple[int, int]] = []

    # Diagonal movement
    if dr != 0 and dc != 0:
        for d in [(dr, dc), (dr, 0), (0, dc)]:
            if valid_move(grid, current, d):
                directions.append(d)

    # Vertical movement
    elif dr != 0 and dc == 0:
        # natural continuation
        if valid_move(grid, current, (dr, 0)):
            directions.append((dr, 0))

        # side branch becomes available now because the previous row
        # along the travel direction had that side blocked
        if (not is_walkable(grid, (r - dr, c + 1))) and is_walkable(grid, (r, c + 1)):
            directions.append((0, 1))

        if (not is_walkable(grid, (r - dr, c - 1))) and is_walkable(grid, (r, c - 1)):
            directions.append((0, -1))

    # Horizontal movement
    elif dr == 0 and dc != 0:
        # natural continuation
        if valid_move(grid, current, (0, dc)):
            directions.append((0, dc))

        # side branch becomes available now because the previous column
        # along the travel direction had that side blocked
        if (not is_walkable(grid, (r - 1, c - dc))) and is_walkable(grid, (r - 1, c)):
            directions.append((-1, 0))

        if (not is_walkable(grid, (r + 1, c - dc))) and is_walkable(grid, (r + 1, c)):
            directions.append((1, 0))

    # deduplicate while preserving order
    out: List[Tuple[int, int]] = []
    seen = set()
    for d in directions:
        if d not in seen:
            seen.add(d)
            out.append(d)

    return out


def has_forced_neighbor(grid: Grid, node: Point, direction: Tuple[int, int]) -> bool:
    """
    Forced-neighbor test adapted for NO-corner-cutting movement.

    For straight motion, a node is a jump point when a side direction
    becomes newly available after being blocked on the previous step.

    For diagonal motion, we rely primarily on recursive straight sub-jump
    checks in jump().
    """
    r, c = node
    dr, dc = direction

    # Vertical movement
    if dr != 0 and dc == 0:
        if (not is_walkable(grid, (r - dr, c + 1))) and is_walkable(grid, (r, c + 1)):
            return True
        if (not is_walkable(grid, (r - dr, c - 1))) and is_walkable(grid, (r, c - 1)):
            return True

    # Horizontal movement
    elif dr == 0 and dc != 0:
        if (not is_walkable(grid, (r - 1, c - dc))) and is_walkable(grid, (r - 1, c)):
            return True
        if (not is_walkable(grid, (r + 1, c - dc))) and is_walkable(grid, (r + 1, c)):
            return True

    # Diagonal movement
    elif dr != 0 and dc != 0:
        # For no-corner-cutting JPS, diagonal jump points are largely handled by
        # the recursive straight-component checks inside jump().
        return False

    return False


def jump(
    grid: Grid,
    current: Point,
    direction: Tuple[int, int],
    goal: Point,
) -> Optional[Point]:
    """
    Move repeatedly in a direction until:
    - blocked
    - goal reached
    - forced neighbor found
    - for diagonal moves: one of horizontal/vertical sub-jumps succeeds
    """
    dr, dc = direction
    r, c = current

    while True:
        if not valid_move(grid, (r, c), direction):
            return None

        nxt = (r + dr, c + dc)

        if nxt == goal:
            return nxt

        if has_forced_neighbor(grid, nxt, direction):
            return nxt

        # For diagonal moves, if either straight component reaches a jump point,
        # then this diagonal node is also a jump point.
        if dr != 0 and dc != 0:
            if jump(grid, nxt, (dr, 0), goal) is not None:
                return nxt
            if jump(grid, nxt, (0, dc), goal) is not None:
                return nxt

        r, c = nxt


def identify_successors(
    grid: Grid,
    current: Point,
    parent: Optional[Point],
    goal: Point,
) -> List[Point]:
    successors: List[Point] = []
    for direction in pruned_directions(grid, current, parent):
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

            tentative_g = g_score[current] + octile_step_distance(current, succ)

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