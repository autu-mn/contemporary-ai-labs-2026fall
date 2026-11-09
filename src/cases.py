"""Official fixtures from the handout and the small comparison cases."""

MAZE = (
    "S..X.",
    ".X...",
    ".X.X.",
    "...XE",
    "XX...",
)
MAZE_START = (0, 0)
MAZE_END = (3, 4)
MAZE_EXPECTED_PATH = [
    (0, 0), (0, 1), (0, 2), (1, 2),
    (1, 3), (1, 4), (2, 4), (3, 4),
]

TERRAIN = (
    ("S", 1, 1, 8, 1),
    (1, 8, 3, 1, 1),
    (1, 8, 1, 8, 1),
    (1, 1, 1, 8, 1),
    (8, 8, 1, 1, "E"),
)
TERRAIN_START = (0, 0)
TERRAIN_END = (4, 4)
TERRAIN_EXPECTED_PATH = [
    (0, 0), (1, 0), (2, 0), (3, 0), (3, 1),
    (3, 2), (4, 2), (4, 3), (4, 4),
]

# The values in the shared [6, 9] subtree appear under both MIN nodes.
OFFICIAL_TREE_GROUPS = ((3, 5), (6, 9), (1, 2))

# This extra tree has the same root value under either order but exposes a
# larger Alpha-Beta difference: left-to-right is favorable, reverse is poor.
ORDERING_TREE_GROUPS = (
    ((3, 5), (6, 9)),
    ((1, 2), (0, -1)),
)

TREASURE_MAP = (
    ("A", 1, 1, "X", 2, 1, "T1"),
    (1, "X", 6, "X", 1, "X", 1),
    (1, 1, "T2", 1, 1, 1, 1),
    ("X", 1, "X", 8, "X", 1, "X"),
    (1, 1, 1, 1, "T3", 1, 1),
    (1, "X", 9, "X", 1, "X", 1),
    ("T4", 1, 1, "X", 1, 1, "B"),
)
TREASURE_LABELS = ("A", "B", "T1", "T2", "T3", "T4")
TREASURE_POINTS = ((0, 0), (6, 6), (0, 6), (2, 2), (4, 4), (6, 0))
TREASURE_VALUES = (9, 6, 7, 7)
EXPECTED_DISTANCES = (
    (0, 12, 10, 4, 8, 8),
    (12, 0, 8, 8, 4, 10),
    (10, 8, 0, 6, 6, 12),
    (4, 8, 6, 0, 6, 6),
    (8, 4, 6, 6, 0, 6),
    (8, 10, 12, 6, 6, 0),
)
EXPECTED_ROOT_VALUES = {
    2: {"T1": -4, "T2": -1, "T3": -2, "T4": -4},
    4: {"T1": -5, "T2": 1, "T3": 3, "T4": -7},
}

OPEN_GRID = tuple("." * 21 for _ in range(21))
DETOUR_GRID = tuple("." * 10 + ("." if row == 2 else "X") + "." * 10
                    for row in range(21))
EXPENSIVE_SHORTCUT = (("S", 9, 9, "E"), (1, 1, 1, 1))

EXTENSION_MAP = (
    ("A", 1, 1, "X", 1, 1, 1, 1, "T1"),
    (1, "X", 1, "X", 1, 8, "X", 1, 1),
    (1, 1, "T2", 1, 1, 1, 1, 1, 1),
    ("X", 1, "X", 7, "X", 1, "X", 1, "X"),
    (1, 1, 1, 1, "T3", 1, 1, 1, 1),
    (1, "X", 9, 1, "X", 1, "X", 1, 1),
    (1, 1, 1, 1, 1, 1, "T5", 1, 1),
    (1, "X", 1, "X", 2, 1, "X", 1, 1),
    ("T4", 1, 1, 1, "T6", 1, 1, 1, "B"),
)
EXTENSION_LABELS = ("A", "B", "T1", "T2", "T3", "T4", "T5", "T6")
EXTENSION_POINTS = ((0, 0), (8, 8), (0, 8), (2, 2), (4, 4), (8, 0), (6, 6), (8, 4))
EXTENSION_VALUES = (13, 8, 9, 11, 6, 12)

