"""Run the fixed comparisons and export the measured results."""

import csv
import json
import platform
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from . import cases as c
from .game_search import tree_from_json, tree_search
from .grid_search import maze_search, terrain_costs, terrain_search
from .treasure_game import TreasureGame, distance_matrix, target_name

BATCHES = 7
SEED = 42
OUTPUT = Path(__file__).resolve().parents[1] / "results"


def measure(function, repeats):
    result = function()
    samples = []
    for _ in range(BATCHES):
        start = perf_counter_ns()
        for _ in range(repeats):
            function()
        samples.append((perf_counter_ns() - start) / repeats)
    return result, round(median(samples) / 1000, 2)


def require(condition, case):
    if not condition:
        raise RuntimeError(f"correctness check failed: {case}")


def grid_comparisons():
    fixtures = (
        ("Q1", "official_maze", maze_search, c.MAZE_START, c.MAZE_END, c.MAZE, 7),
        ("Q1", "open_21", maze_search, (10, 0), (10, 20), c.OPEN, 20),
        ("Q1", "detour_21", maze_search, (10, 0), (10, 20), c.DETOUR, 36),
        ("Q2", "official_terrain", terrain_search, c.TERRAIN_START, c.TERRAIN_END, c.TERRAIN, 7),
        ("Q2", "expensive_shortcut", terrain_search, (0, 0), (0, 3), c.SHORTCUT, 4),
    )
    rows, paths = [], {}
    for question, name, search, start, end, grid, expected in fixtures:
        for enabled in (False, True):
            heuristic = ("manhattan" if question == "Q1" else "adjusted") if enabled else "zero"
            result, elapsed = measure(lambda: search(start, end, grid, use_heuristic=enabled), 40)
            require(result.cost == expected, (name, heuristic))
            require(result.path[0] == start and result.path[-1] == end, name)
            costs = terrain_costs(grid, end) if question == "Q2" else None
            paid = [costs[r][col] if costs is not None else 1 for r, col in result.path[1:]]
            require(sum(paid) == expected, (name, "path cost"))
            rows.append(dict(question=question, case=name, heuristic=heuristic, cost=result.cost,
                             steps=len(result.path) - 1, expanded=result.expanded,
                             peak_heap=result.peak_heap, stale_pops=result.stale_pops,
                             goal_pops=result.goal_pops, search_us=elapsed,
                             batches=BATCHES, repeats_per_batch=40))
            if enabled:
                cumulative = 0
                trace = [dict(point=start, entry=0, cumulative=0)]
                for point, entry in zip(result.path[1:], paid):
                    cumulative += entry
                    trace.append(dict(point=point, entry=entry, cumulative=cumulative))
                paths[name] = trace
    return rows, paths


def game_row(question, name, depth, prune, order, result, elapsed, repeats, **extra):
    return dict(question=question, case=name, depth=depth,
                algorithm="alpha_beta" if prune else "minimax", order=order,
                best_target=extra.get("target"), best_value=result.value, **asdict(result.stats),
                leaf_saving_pct=round(100 * (1 - result.stats.leaves / extra["baseline"]), 2),
                search_us=elapsed, matrix_us=extra.get("matrix"), total_us=extra.get("total"),
                sorting_probe_us=extra.get("sort"), batches=BATCHES, repeats_per_batch=repeats)


def game_comparisons():
    rows, details = [], {}
    root = tree_from_json(c.TREE)
    baseline = tree_search(root, prune=False).stats.leaves
    for prune, order in ((False, "left"), (True, "left"), (True, "reverse"), (True, "shuffle")):
        result, elapsed = measure(lambda: tree_search(root, prune=prune, order=order, seed=SEED), 200)
        require(result.value == 5, ("tree", order))
        rows.append(game_row("Q3", "official_tree", None, prune, order, result, elapsed, 200,
                             baseline=baseline))

    fixtures = (
        ("official_treasure", c.TREASURE_MAP, c.TREASURE_POINTS, c.TREASURE_VALUES, (2, 4)),
        ("six_treasures", c.EXTENSION_MAP, c.EXTENSION_POINTS, c.EXTENSION_VALUES, (2, 4, 6)),
    )
    for name, grid, points, values, depths in fixtures:
        grid = c.numeric_grid(grid)

        def preprocess():
            return distance_matrix(grid, points)

        matrix, matrix_time = measure(preprocess, 8)
        if name == "official_treasure":
            require(matrix == c.EXPECTED_DISTANCES, "official distance matrix")
        game = TreasureGame(matrix, values)
        details[name] = dict(matrix=matrix, points=points, values=values, depths={})
        for depth in depths:
            baseline = game.search(depth, prune=False)
            values_by_target = game.exact_root_values(depth)
            line = game.optimal_line(depth)
            require(line[-1]["utility"] == baseline.value, (name, depth, "optimal line"))
            if name == "official_treasure":
                require(values_by_target == c.EXPECTED_ROOT_VALUES[depth], ("root values", depth))
            details[name]["depths"][depth] = dict(root_values=values_by_target, optimal_line=line)
            for prune, order in ((False, "index"), (True, "index"), (True, "reverse"), (True, "gain")):
                result, elapsed = measure(lambda: game.search(depth, prune=prune, order=order), 25)
                require((result.value, result.action) == (baseline.value, baseline.action),
                        (name, depth, order))
                _, total = measure(
                    lambda: TreasureGame(preprocess(), values).search(depth, prune=prune, order=order), 8
                )
                sorting = None
                if order == "gain":
                    def profile_sorting():
                        samples = []
                        game.search(depth, prune=prune, order=order, sort_samples=samples)
                        return sum(samples)

                    sorting = round(median(profile_sorting() for _ in range(31)) / 1000, 2)
                rows.append(game_row("Q4", name, depth, prune, order, result, elapsed, 25,
                                     baseline=baseline.stats.leaves, target=target_name(result.action),
                                     matrix=matrix_time, total=total, sort=sorting))
    return rows, details


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    grid_rows, paths = grid_comparisons()
    game_rows, games = game_comparisons()
    OUTPUT.mkdir(exist_ok=True)
    write_csv(OUTPUT / "grid_comparisons.csv", grid_rows)
    write_csv(OUTPUT / "game_comparisons.csv", game_rows)
    details = dict(environment=dict(python=platform.python_version(), platform=platform.platform(),
                                    seed=SEED, batches=BATCHES), paths=paths, games=games)
    (OUTPUT / "details.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(f"Verified and exported {len(grid_rows)} grid rows and {len(game_rows)} game rows.")
    print("Q1: 7 steps | Q2: cost 7 | Q3: score 5 | Q4: H=2 T2/-1, H=4 T3/3")


if __name__ == "__main__":
    main()
