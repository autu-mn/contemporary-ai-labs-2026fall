"""Run verified comparisons and export the data used by the report."""

import argparse
import csv
import json
import platform
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from q3_game_tree import build_official_tree, build_ordering_tree

from . import cases
from .game_search import alpha_beta_search, minimax_search
from .grid_search import maze_search, terrain_cost_grid, terrain_search
from .treasure_game import TreasureGame, distance_matrix, target_name

BATCHES = 7


def measure(function, repeats):
    """Return one result and median time per call in microseconds."""
    result = function()
    samples = []
    for _ in range(BATCHES):
        started = perf_counter_ns()
        for _ in range(repeats):
            function()
        samples.append((perf_counter_ns() - started) / repeats / 1000)
    return result, round(median(samples), 2)


def require(condition, description):
    if not condition:
        raise RuntimeError(f"verification failed: {description}")


def trace_path(path, costs=None):
    cumulative = 0
    trace = []
    for index, point in enumerate(path):
        entry = 0 if index == 0 else (1 if costs is None else costs[point[0]][point[1]])
        cumulative += entry
        trace.append({"point": list(point), "entry_cost": entry, "cumulative_cost": cumulative})
    return trace


def grid_experiments():
    fixtures = (
        ("Q1", "official_maze", maze_search, cases.MAZE_START, cases.MAZE_END, cases.MAZE, 7),
        ("Q1", "open_21", maze_search, (10, 0), (10, 20), cases.OPEN_GRID, 20),
        ("Q1", "detour_21", maze_search, (10, 0), (10, 20), cases.DETOUR_GRID, 36),
        ("Q2", "official_terrain", terrain_search,
         cases.TERRAIN_START, cases.TERRAIN_END, cases.TERRAIN, 7),
        ("Q2", "expensive_shortcut", terrain_search,
         (0, 0), (0, 3), cases.EXPENSIVE_SHORTCUT, 4),
    )
    rows = []
    official = {}
    for question, name, search, start, end, grid, expected_cost in fixtures:
        for enabled in (False, True):
            result, elapsed = measure(
                lambda: search(start, end, grid, use_heuristic=enabled), 60
            )
            require(result.cost == expected_cost, (name, "cost"))
            require(result.path and result.path[0] == start and result.path[-1] == end,
                    (name, "path endpoints"))
            heuristic = "zero" if not enabled else (
                "manhattan" if question == "Q1" else "adjusted_manhattan"
            )
            rows.append({
                "question": question,
                "case": name,
                "heuristic": heuristic,
                "cost": result.cost,
                "steps": len(result.path) - 1,
                "expanded": result.expanded,
                "peak_frontier": result.peak_frontier,
                "time_us": elapsed,
                "batches": BATCHES,
                "repeats": 60,
            })
            if enabled and name == "official_maze":
                official["q1"] = {
                    "path": [list(point) for point in result.path],
                    "steps": len(result.path) - 1,
                    "trace": trace_path(result.path),
                }
            if enabled and name == "official_terrain":
                costs = terrain_cost_grid(grid, end)
                official["q2"] = {
                    "cost": result.cost,
                    "path": [list(point) for point in result.path],
                    "trace": trace_path(result.path, costs),
                }
    require(official["q1"]["path"] == [list(point) for point in cases.MAZE_EXPECTED_PATH],
            "official maze path")
    require(official["q2"]["path"] == [list(point) for point in cases.TERRAIN_EXPECTED_PATH],
            "official terrain path")
    return rows, official


def game_row(question, case_name, depth, algorithm, order, result, elapsed, repeats):
    return {
        "question": question,
        "case": case_name,
        "depth": "" if depth is None else depth,
        "algorithm": algorithm,
        "order": order,
        "best_target": "" if result.action is None or question == "Q3" else target_name(result.action),
        "best_value": result.value,
        **asdict(result.stats),
        "time_us": elapsed,
        "batches": BATCHES,
        "repeats": repeats,
    }


def game_experiments():
    rows = []

    for case_name, builder in (
        ("official_tree", build_official_tree),
        ("order_sensitive_tree", build_ordering_tree),
    ):
        root = builder()
        for algorithm, order in (
            ("minimax", "left"),
            ("alpha_beta", "left"),
            ("alpha_beta", "reverse"),
        ):
            function = minimax_search if algorithm == "minimax" else alpha_beta_search
            result, elapsed = measure(lambda f=function, o=order: f(root, order=o), 400)
            require(result.value == 5, (case_name, algorithm, order))
            rows.append(game_row("Q3", case_name, None, algorithm, order,
                                 result, elapsed, 400))

    official_root = build_official_tree()
    official_tree_result = alpha_beta_search(official_root, order="left")
    official = {
        "q3": {
            "root_value": official_tree_result.value,
            "left_to_right": asdict(official_tree_result.stats),
        }
    }

    for case_name, grid, points, values, depths in (
        ("official_treasure", cases.TREASURE_MAP, cases.TREASURE_POINTS,
         cases.TREASURE_VALUES, (2, 4)),
        ("six_treasures", cases.EXTENSION_MAP, cases.EXTENSION_POINTS,
         cases.EXTENSION_VALUES, (2, 4, 6)),
    ):
        matrix, matrix_time = measure(lambda: distance_matrix(grid, points), 15)
        if case_name == "official_treasure":
            require(matrix == cases.EXPECTED_DISTANCES, "official distance matrix")
        game = TreasureGame(matrix, values)

        case_details = {
            "labels": cases.TREASURE_LABELS if case_name == "official_treasure"
            else cases.EXTENSION_LABELS,
            "points": points,
            "values": values,
            "distance_matrix": matrix,
            "distance_matrix_time_us": matrix_time,
            "depths": {},
        }
        for depth in depths:
            minimax_result = game.minimax_search(depth)
            root_values = game.exact_root_values(depth)
            if case_name == "official_treasure":
                require(root_values == cases.EXPECTED_ROOT_VALUES[depth],
                        ("official root values", depth))
            case_details["depths"][str(depth)] = {
                "root_values": root_values,
                "optimal_line": game.optimal_line(depth),
                "best_target": target_name(minimax_result.action),
                "best_value": minimax_result.value,
            }

            configurations = (
                ("minimax", "index"),
                ("alpha_beta", "index"),
                ("alpha_beta", "reverse"),
                ("alpha_beta", "gain"),
            )
            repeats = 35 if depth < 6 else 15
            for algorithm, order in configurations:
                function = game.minimax_search if algorithm == "minimax" else game.alpha_beta_search
                result, elapsed = measure(lambda f=function, o=order: f(depth, order=o), repeats)
                require(result.value == minimax_result.value,
                        (case_name, depth, algorithm, order, "root value"))
                if case_name == "official_treasure":
                    require(result.action == minimax_result.action,
                            (case_name, depth, algorithm, order, "best target"))
                rows.append(game_row("Q4", case_name, depth, algorithm, order,
                                     result, elapsed, repeats))

        official[case_name] = case_details

    require(official["official_treasure"]["depths"]["2"]["best_target"] == "T2",
            "Q4 H=2 target")
    require(official["official_treasure"]["depths"]["2"]["best_value"] == -1,
            "Q4 H=2 value")
    require(official["official_treasure"]["depths"]["4"]["best_target"] == "T3",
            "Q4 H=4 target")
    require(official["official_treasure"]["depths"]["4"]["best_value"] == 3,
            "Q4 H=4 value")
    return rows, official


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42,
                        help="recorded for reproducibility; current experiments are deterministic")
    arguments = parser.parse_args(argv)

    grid_rows, grid_official = grid_experiments()
    game_rows, game_official = game_experiments()

    details = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "seed": arguments.seed,
            "batches": BATCHES,
        },
        **grid_official,
        **game_official,
    }

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        arguments.output_dir / "grid_comparisons.csv",
        ("question", "case", "heuristic", "cost", "steps", "expanded",
         "peak_frontier", "time_us", "batches", "repeats"),
        grid_rows,
    )
    write_csv(
        arguments.output_dir / "game_comparisons.csv",
        ("question", "case", "depth", "algorithm", "order", "best_target",
         "best_value", "visited", "leaves", "cutoffs", "pruned_children",
         "time_us", "batches", "repeats"),
        game_rows,
    )
    (arguments.output_dir / "official_results.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Verified and wrote {len(grid_rows)} grid rows and {len(game_rows)} game rows.")
    print("Q1: 7 steps | Q2: cost 7 | Q3: score 5 | Q4: H=2 T2/-1, H=4 T3/3")


if __name__ == "__main__":
    main()

