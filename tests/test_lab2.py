import unittest
from heapq import heappop, heappush
from random import Random

from q3_game_tree import build_official_tree, build_ordering_tree
from src import cases
from src.game_search import alpha_beta_search, minimax_search
from src.grid_search import maze_path, minimum_terrain_cost, terrain_cost_grid, terrain_search
from src.treasure_game import TreasureGame, distance_matrix


def dijkstra(start, end, costs):
    frontier = [(0, start)]
    best = {start: 0}
    while frontier:
        cost, point = heappop(frontier)
        if cost != best[point]:
            continue
        if point == end:
            return cost
        row, col = point
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            neighbor = row + dr, col + dc
            nr, nc = neighbor
            if not (0 <= nr < len(costs) and 0 <= nc < len(costs[0])):
                continue
            if costs[nr][nc] is None:
                continue
            new_cost = cost + costs[nr][nc]
            if new_cost < best.get(neighbor, float("inf")):
                best[neighbor] = new_cost
                heappush(frontier, (new_cost, neighbor))
    return -1


class Question1Tests(unittest.TestCase):
    def test_official_path(self):
        path = maze_path(cases.MAZE_START, cases.MAZE_END, cases.MAZE)
        self.assertEqual(path, cases.MAZE_EXPECTED_PATH)
        self.assertEqual(len(path) - 1, 7)
        for first, second in zip(path, path[1:]):
            self.assertEqual(abs(first[0] - second[0]) + abs(first[1] - second[1]), 1)

    def test_no_path_and_same_point(self):
        self.assertEqual(maze_path((0, 0), (0, 2), ("SXE",)), [])
        self.assertEqual(maze_path((0, 0), (0, 0), (("S",),)), [(0, 0)])


class Question2Tests(unittest.TestCase):
    def test_official_cost_and_path(self):
        result = terrain_search(cases.TERRAIN_START, cases.TERRAIN_END, cases.TERRAIN)
        self.assertEqual(result.cost, 7)
        self.assertEqual(result.path, cases.TERRAIN_EXPECTED_PATH)

    def test_free_end_and_expensive_shortcut(self):
        self.assertEqual(minimum_terrain_cost((0, 0), (0, 1), (("S", "E"),)), 0)
        self.assertEqual(minimum_terrain_cost((0, 0), (0, 3), cases.EXPENSIVE_SHORTCUT), 4)

    def test_astar_matches_independent_dijkstra(self):
        random = Random(42)
        for _ in range(30):
            grid = [[random.choice((1, 2, 3, 8, "X")) for _ in range(5)] for _ in range(4)]
            grid[0][0] = "S"
            grid[3][4] = "E"
            costs = terrain_cost_grid(grid, (3, 4))
            self.assertEqual(
                terrain_search((0, 0), (3, 4), grid).cost,
                dijkstra((0, 0), (3, 4), costs),
            )


class Question3Tests(unittest.TestCase):
    def test_official_tree(self):
        root = build_official_tree()
        minimax = minimax_search(root)
        alpha_beta = alpha_beta_search(root)
        self.assertEqual((minimax.value, alpha_beta.value), (5, 5))
        self.assertEqual((minimax.stats.leaves, alpha_beta.stats.leaves), (8, 7))

    def test_order_changes_work_not_value(self):
        root = build_ordering_tree()
        favorable = alpha_beta_search(root, order="left")
        unfavorable = alpha_beta_search(root, order="reverse")
        self.assertEqual((favorable.value, unfavorable.value), (5, 5))
        self.assertLess(favorable.stats.leaves, unfavorable.stats.leaves)


class Question4Tests(unittest.TestCase):
    def setUp(self):
        self.matrix = distance_matrix(cases.TREASURE_MAP, cases.TREASURE_POINTS)
        self.game = TreasureGame(self.matrix, cases.TREASURE_VALUES)

    def test_distance_matrix(self):
        self.assertEqual(self.matrix, cases.EXPECTED_DISTANCES)

    def test_official_depths_and_algorithms(self):
        for depth, target, value in ((2, "T2", -1), (4, "T3", 3)):
            self.assertEqual(self.game.exact_root_values(depth), cases.EXPECTED_ROOT_VALUES[depth])
            expected = {"best_target": target, "best_value": value}
            for algorithm in ("minimax", "alpha_beta"):
                for order in ("index", "reverse", "gain"):
                    self.assertEqual(self.game.solve(depth, algorithm=algorithm, order=order), expected)

    def test_state_transition(self):
        state = self.game.move(self.game.initial, 3)
        self.assertEqual((state.a_position, state.b_position), (3, 1))
        self.assertEqual((state.score_a, state.score_b, state.a_turn), (2, 0, False))
        self.assertEqual(state.remaining, (2, 4, 5))

    def test_zero_cost_custom_map_keeps_distance_optimal(self):
        grid = (
            ("A", 5, "T1"),
            (0, 0, 0),
        )
        matrix = distance_matrix(grid, ((0, 0), (0, 2)))
        self.assertEqual(matrix[0][1], 1)


if __name__ == "__main__":
    unittest.main()
