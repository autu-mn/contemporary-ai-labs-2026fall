import unittest
from random import Random

from src import cases as c
from src.game_search import TreeNode, alpha_beta, minimax, tree_from_json, tree_search
from src.grid_search import maze_path, minimum_terrain_cost, terrain_search
from src.treasure_game import TreasureGame, distance_matrix


class GridTests(unittest.TestCase):
    def test_official_maze(self):
        path = maze_path(c.MAZE_START, c.MAZE_END, c.MAZE)
        self.assertEqual((path[0], path[-1], len(path)), (c.MAZE_START, c.MAZE_END, 8))
        for r, col in path:
            self.assertNotEqual(c.MAZE[r][col], "X")
        for (r, col), (nr, nc) in zip(path, path[1:]):
            self.assertEqual(abs(r - nr) + abs(col - nc), 1)

    def test_maze_edges(self):
        self.assertEqual(maze_path((0, 0), (0, 2), ("SXE",)), [])
        self.assertEqual(maze_path((0, 0), (0, 0), ("S",)), [(0, 0)])
        self.assertEqual(len(maze_path((0, 0), (1, 1), ("S.", ".E"))), 3)
        self.assertEqual(len(maze_path((0, 0), (0, 2), ("S.E",))), 3)

    def test_official_terrain(self):
        result = terrain_search(c.TERRAIN_START, c.TERRAIN_END, c.TERRAIN)
        self.assertEqual((result.cost, len(result.path) - 1), (7, 8))
        self.assertEqual(minimum_terrain_cost((0, 0), (0, 3), c.SHORTCUT), 4)

    def test_terrain_edges(self):
        self.assertEqual(minimum_terrain_cost((0, 0), (0, 1), (("S", "E"),)), 0)
        self.assertEqual(minimum_terrain_cost((0, 0), (0, 1), (("S", "2"),)), 2)
        self.assertEqual(minimum_terrain_cost((0, 0), (0, 2), (("S", "X", "E"),)), -1)
        self.assertEqual(minimum_terrain_cost((0, 0), (0, 0), ((5,),)), 0)
        self.assertEqual(minimum_terrain_cost((0, 0), (0, 2), (("S", 0, "E"),)), 0)

    def test_weighted_differential(self):
        rng = Random(42)
        for _ in range(80):
            grid = [[rng.choice((1, 2, 3, 8, "X")) for _ in range(5)] for _ in range(4)]
            grid[0][0] = "S"
            grid[3][4] = rng.choice(("E", 3))
            self.assertEqual(terrain_search((0, 0), (3, 4), grid).cost,
                             terrain_search((0, 0), (3, 4), grid, use_heuristic=False).cost)


class GameTests(unittest.TestCase):
    def test_official_tree(self):
        root = tree_from_json(c.TREE)
        self.assertEqual((minimax(root), alpha_beta(root)), (5, 5))
        self.assertLess(tree_search(root).stats.leaves, tree_search(root, prune=False).stats.leaves)

    def test_leaf_and_equal_cutoff(self):
        self.assertEqual(alpha_beta(TreeNode(-2)), -2)
        root = TreeNode(children=[TreeNode(0), TreeNode(children=[TreeNode(0), TreeNode(-4)])])
        result = tree_search(root)
        self.assertEqual((result.value, result.stats.cutoffs, result.stats.leaves), (0, 1, 2))

    def test_tree_differential(self):
        rng = Random(42)

        def make_tree(depth):
            if depth == 0 or rng.random() < .2:
                return TreeNode(rng.randint(-5, 5))
            return TreeNode(children=[make_tree(depth - 1) for _ in range(rng.randint(1, 4))])

        for _ in range(80):
            root = make_tree(4)
            for order in ("left", "reverse", "shuffle"):
                self.assertEqual(tree_search(root, order=order).value, minimax(root))

    def test_official_treasure(self):
        distances = distance_matrix(c.numeric_grid(c.TREASURE_MAP), c.TREASURE_POINTS)
        self.assertEqual(distances, c.EXPECTED_DISTANCES)
        game = TreasureGame(distances, c.TREASURE_VALUES)
        for depth, target, value in ((2, "T2", -1), (4, "T3", 3)):
            self.assertEqual(game.exact_root_values(depth), c.EXPECTED_ROOT_VALUES[depth])
            self.assertEqual(game.optimal_line(depth)[-1]["utility"], value)
            for prune in (True, False):
                for order in ("index", "reverse", "gain"):
                    self.assertEqual(game.solve(depth, prune=prune, order=order),
                                     {"best_target": target, "best_value": value})

    def test_directed_distances_and_unreachable_treasure(self):
        self.assertEqual(distance_matrix([[1, 4]], ((0, 0), (0, 1))), ((0, 4), (1, 0)))
        distances = distance_matrix([[1, 1, "X", 1]], ((0, 0), (0, 1), (0, 3)))
        game = TreasureGame(distances, (10,))
        self.assertEqual(game.solve(2), {"best_target": None, "best_value": 0})

    def test_treasure_tie_and_transitions(self):
        game = TreasureGame(((0, 1, 2, 2), (1, 0, 2, 2), (2, 2, 0, 2), (2, 2, 2, 0)), (1, 1))
        self.assertEqual(game.solve(0), {"best_target": None, "best_value": 0})
        self.assertEqual(game.solve(4, order="gain"), {"best_target": "T1", "best_value": 0})
        next_state = game.move(game.initial, 2)
        self.assertEqual((next_state.a, next_state.b, next_state.remaining), (2, 1, (3,)))
        self.assertEqual((next_state.score_a, next_state.score_b, next_state.a_turn), (-1, 0, False))
        self.assertEqual(game.initial.remaining, (2, 3))


if __name__ == "__main__":
    unittest.main()
