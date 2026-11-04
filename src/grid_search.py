"""Four-connected grid search with entry costs."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count

Point = tuple[int, int]
Costs = Sequence[Sequence[int | None]]
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass
class GridResult:
    path: list[Point]
    cost: int
    expanded: int
    peak_heap: int
    stale_pops: int
    goal_pops: int


def manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start: Point, end: Point, costs: Costs,
          heuristic: Callable[[Point], int]) -> GridResult:
    """Search a grid of nonnegative entry costs. None marks a blocked cell."""
    rows, cols = len(costs), len(costs[0]) if costs else 0
    if any(len(row) != cols for row in costs):
        raise ValueError("grid must be rectangular")

    def passable(point):
        r, c = point
        return 0 <= r < rows and 0 <= c < cols and costs[r][c] is not None

    if not passable(start) or not passable(end):
        return GridResult([], -1, 0, 0, 0, 0)

    order = count()
    heap = [(heuristic(start), next(order), 0, start)]
    best = {start: 0}
    parent = {}
    expanded, peak, stale = 0, 1, 0
    while heap:
        _, _, g, point = heappop(heap)
        if g != best[point]:
            stale += 1
            continue
        if point == end:
            path = [point]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return GridResult(path[::-1], g, expanded, peak, stale, 1)

        expanded += 1
        r, c = point
        for dr, dc in DIRECTIONS:
            neighbor = (r + dr, c + dc)
            if not passable(neighbor):
                continue
            new_g = g + costs[neighbor[0]][neighbor[1]]
            if neighbor not in best or new_g < best[neighbor]:
                best[neighbor] = new_g
                parent[neighbor] = point
                heappush(heap, (new_g + heuristic(neighbor), next(order), new_g, neighbor))
        peak = max(peak, len(heap))

    return GridResult([], -1, expanded, peak, stale, 0)


def maze_search(start, end, grid, *, use_heuristic=True):
    costs = [[None if cell == "X" else 1 for cell in row] for row in grid]
    return astar(start, end, costs, lambda point: manhattan(point, end) if use_heuristic else 0)


def maze_path(start, end, grid) -> list[Point]:
    return maze_search(start, end, grid).path


def terrain_costs(grid, end):
    """Convert the terrain markers and reject negative or fractional costs."""
    costs = []
    for r, row in enumerate(grid):
        converted = []
        for c, cell in enumerate(row):
            if cell == "X":
                converted.append(None)
            elif cell == "S":
                converted.append(1)
            elif cell == "E":
                if (r, c) != end:
                    raise ValueError("E must mark the requested end")
                converted.append(0)
            else:
                if type(cell) not in (int, str) or int(cell) < 0:
                    raise ValueError("terrain costs must be nonnegative integers")
                converted.append(int(cell))
        costs.append(converted)
    return costs


def terrain_search(start, end, grid, *, use_heuristic=True):
    costs = terrain_costs(grid, end)
    positive = [cost for row in costs for cost in row if cost is not None and cost > 0]
    free = [(r, c) for r, row in enumerate(costs) for c, cost in enumerate(row) if cost == 0]
    scale = min(positive, default=0) if use_heuristic and all(p == end for p in free) else 0
    return astar(start, end, costs, lambda point: scale * max(manhattan(point, end) - len(free), 0))


def minimum_terrain_cost(start, end, grid) -> int:
    return terrain_search(start, end, grid).cost
