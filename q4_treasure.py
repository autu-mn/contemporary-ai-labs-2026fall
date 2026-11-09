"""Direct entry for Question 4."""

import json

from src.cases import TREASURE_MAP, TREASURE_POINTS, TREASURE_VALUES
from src.treasure_game import TreasureGame, distance_matrix


def solve():
    matrix = distance_matrix(TREASURE_MAP, TREASURE_POINTS)
    game = TreasureGame(matrix, TREASURE_VALUES)
    results = {
        ("Minimax", 2): game.solve(2, algorithm="minimax"),
        ("Alpha-Beta", 2): game.solve(2, algorithm="alpha_beta"),
        ("Minimax", 4): game.solve(4, algorithm="minimax"),
        ("Alpha-Beta", 4): game.solve(4, algorithm="alpha_beta"),
    }
    return matrix, results


if __name__ == "__main__":
    official_matrix, official_results = solve()
    print("Distance matrix:")
    for matrix_row in official_matrix:
        print(list(matrix_row))
    print()
    for depth in (2, 4):
        for algorithm in ("Minimax", "Alpha-Beta"):
            result = json.dumps(official_results[(algorithm, depth)], ensure_ascii=False)
            print(f"{algorithm} H={depth} => {result}")

