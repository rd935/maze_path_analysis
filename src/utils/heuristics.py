from __future__ import annotations

import math
from typing import Tuple

Point = Tuple[int, int]


def manhattan(a: Point, b: Point) -> float:
    """Manhattan distance for 4-connected grids."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: Point, b: Point) -> float:
    """Euclidean distance."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def octile(a: Point, b: Point) -> float:
    """
    Octile distance for 8-connected grids.
    Commonly used when diagonal moves are allowed.
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)