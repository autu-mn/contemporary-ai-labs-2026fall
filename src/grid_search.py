"""A* search for four-connected grids with destination-entry costs."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count

Point = tuple[int, int]
CostGrid = Sequence[Sequence[int | None]]

# The order keeps the official examples deterministic while remaining a
# normal four-connected search.
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass(frozen=True)
class GridResult:
    path: list[Point]
    cost: int
    expanded: int
    peak_frontier: int


def manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start: Point, end: Point, costs: CostGrid,
          heuristic: Callable[[Point], int]) -> GridResult:
    """Return a shortest path on a nonnegative entry-cost grid.

    ``None`` marks an obstacle. The source cell is not charged; moving into a
    neighbor charges that neighbor's value.
    """
    if not costs or not costs[0]:
        return GridResult([], -1, 0, 0)
    rows, cols = len(costs), len(costs[0])
    if any(len(row) != cols for row in costs):
        raise ValueError("grid must be rectangular")
    if any(value is not None and (type(value) is not int or value < 0)
           for row in costs for value in row):
        raise ValueError("entry costs must be nonnegative integers")

    def passable(point: Point) -> bool:
        row, col = point
        return 0 <= row < rows and 0 <= col < cols and costs[row][col] is not None

    if not passable(start) or not passable(end):
        return GridResult([], -1, 0, 0)

    serial = count()
    frontier = [(heuristic(start), next(serial), 0, start)]
    best_cost = {start: 0}
    parent: dict[Point, Point] = {}
    expanded = 0
    peak = 1

    while frontier:
        _, _, current_cost, current = heappop(frontier)
        if current_cost != best_cost.get(current):
            continue
        if current == end:
            path = [current]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return GridResult(path, current_cost, expanded, peak)

        expanded += 1
        row, col = current
        for dr, dc in DIRECTIONS:
            neighbor = (row + dr, col + dc)
            if not passable(neighbor):
                continue
            new_cost = current_cost + costs[neighbor[0]][neighbor[1]]
            if new_cost >= best_cost.get(neighbor, float("inf")):
                continue
            best_cost[neighbor] = new_cost
            parent[neighbor] = current
            heappush(frontier, (
                new_cost + heuristic(neighbor), next(serial), new_cost, neighbor
            ))
        peak = max(peak, len(frontier))

    return GridResult([], -1, expanded, peak)


def maze_search(start: Point, end: Point, grid, *, use_heuristic: bool = True) -> GridResult:
    costs = [[None if cell == "X" else 1 for cell in row] for row in grid]
    heuristic = (lambda point: manhattan(point, end)) if use_heuristic else (lambda _: 0)
    return astar(start, end, costs, heuristic)


def maze_path(start: Point, end: Point, grid) -> list[Point]:
    """Question 1 interface: return the complete shortest path or ``[]``."""
    return maze_search(start, end, grid).path


def terrain_cost_grid(grid, end: Point) -> list[list[int | None]]:
    """Convert S/E/X and numeric cells into destination-entry costs."""
    converted: list[list[int | None]] = []
    for row_index, row in enumerate(grid):
        output_row: list[int | None] = []
        for col_index, cell in enumerate(row):
            if cell == "X":
                output_row.append(None)
            elif cell == "S":
                output_row.append(1)
            elif cell == "E":
                if (row_index, col_index) != end:
                    raise ValueError("E must mark the requested end")
                output_row.append(0)
            else:
                try:
                    value = int(cell)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"invalid terrain cell: {cell!r}") from error
                if value < 0 or isinstance(cell, float):
                    raise ValueError("terrain costs must be nonnegative integers")
                output_row.append(value)
        converted.append(output_row)
    return converted


def terrain_search(start: Point, end: Point, grid, *, use_heuristic: bool = True) -> GridResult:
    costs = terrain_cost_grid(grid, end)
    positive = [value for row in costs for value in row if value is not None and value > 0]
    zero_cells = [(row, col) for row, values in enumerate(costs)
                  for col, value in enumerate(values) if value == 0]

    # If E is the only free cell, at most the final Manhattan step is free.
    # Any additional zero-cost cell invalidates this positive lower bound.
    if use_heuristic and zero_cells and all(point == end for point in zero_cells):
        scale = min(positive, default=0)
        heuristic = lambda point: scale * max(manhattan(point, end) - 1, 0)
    elif use_heuristic and not zero_cells:
        scale = min(positive, default=0)
        heuristic = lambda point: scale * manhattan(point, end)
    else:
        heuristic = lambda _: 0
    return astar(start, end, costs, heuristic)


def minimum_terrain_cost(start: Point, end: Point, grid) -> int:
    """Question 2 interface: return the minimum total cost or ``-1``."""
    return terrain_search(start, end, grid).cost

