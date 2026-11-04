"""Course fixtures transcribed from the handout and fixed comparison cases."""

MAZE = (
    "S..X.",
    ".X...",
    ".X.X.",
    "...XE",
    "XX...",
)
MAZE_START, MAZE_END = (0, 0), (3, 4)

TERRAIN = (
    ("S", 1, 1, 8, 1),
    (1, 8, 3, 1, 1),
    (1, 8, 1, 8, 1),
    (1, 1, 1, 8, 1),
    (8, 8, 1, 1, "E"),
)
TERRAIN_START, TERRAIN_END = (0, 0), (4, 4)

TREE = {"tree_structure": {
    "is_maximizing_player": True,
    "children": [
        {"is_maximizing_player": False, "children": [
            {"is_maximizing_player": True, "children": [{"value": 3}, {"value": 5}]},
            {"is_maximizing_player": True, "children": [{"value": 6}, {"value": 9}]},
        ]},
        {"is_maximizing_player": False, "children": [
            {"is_maximizing_player": True, "children": [{"value": 6}, {"value": 9}]},
            {"is_maximizing_player": True, "children": [{"value": 1}, {"value": 2}]},
        ]},
    ],
}}

TREASURE_MAP = (
    "111X211",
    "1X6X1X1",
    "1111111",
    "X1X8X1X",
    "1111111",
    "1X9X1X1",
    "111X111",
)
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
    2: dict(T1=-4, T2=-1, T3=-2, T4=-4),
    4: dict(T1=-5, T2=1, T3=3, T4=-7),
}

OPEN = ("." * 21,) * 21
DETOUR = tuple("." * 10 + ("." if r == 2 else "X") + "." * 10 for r in range(21))
SHORTCUT = (("S", 9, 9, "E"), (1, 1, 1, 1))

EXTENSION_MAP = (
    "111X11111",
    "1X1X18X11",
    "111111111",
    "X1X7X1X1X",
    "111111111",
    "1X91X1X11",
    "111111111",
    "1X1X21X11",
    "111111111",
)
EXTENSION_POINTS = ((0, 0), (8, 8), (0, 8), (2, 2), (4, 4), (8, 0), (6, 6), (8, 4))
EXTENSION_VALUES = (13, 8, 9, 11, 6, 12)


def numeric_grid(rows):
    return [[cell if cell == "X" else int(cell) for cell in row] for row in rows]
