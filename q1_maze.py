"""Direct entry for Question 1."""

from src.cases import MAZE, MAZE_END, MAZE_START
from src.grid_search import maze_path


def solve(start, end, grid):
    return maze_path(start, end, grid)


if __name__ == "__main__":
    print(solve(MAZE_START, MAZE_END, MAZE))

