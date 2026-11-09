"""Direct entry for Question 2."""

from src.cases import TERRAIN, TERRAIN_END, TERRAIN_START
from src.grid_search import minimum_terrain_cost


def solve(start, end, grid):
    return minimum_terrain_cost(start, end, grid)


if __name__ == "__main__":
    print(solve(TERRAIN_START, TERRAIN_END, TERRAIN))

