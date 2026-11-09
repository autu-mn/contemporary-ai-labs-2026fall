"""A* distance preprocessing and explicit game search for Question 4."""

from dataclasses import dataclass, replace
from math import inf

from .game_search import GameResult, SearchStats
from .grid_search import astar, manhattan


def treasure_cost_grid(grid) -> list[list[int | None]]:
    """Map X to an obstacle and A/B/T labels to entry cost 1."""
    output: list[list[int | None]] = []
    for row in grid:
        converted: list[int | None] = []
        for cell in row:
            if cell == "X":
                converted.append(None)
            elif isinstance(cell, int):
                if cell < 0:
                    raise ValueError("entry costs must be nonnegative")
                converted.append(cell)
            elif isinstance(cell, str) and cell.isdigit():
                converted.append(int(cell))
            elif cell in {"A", "B"} or (isinstance(cell, str) and cell.startswith("T")):
                converted.append(1)
            else:
                raise ValueError(f"invalid treasure-map cell: {cell!r}")
        output.append(converted)
    if output and any(len(row) != len(output[0]) for row in output):
        raise ValueError("grid must be rectangular")
    return output


def distance_matrix(grid, points) -> tuple[tuple[int, ...], ...]:
    """Compute every ordered A/B/treasure pair with A*."""
    costs = treasure_cost_grid(grid)
    positive = [value for row in costs for value in row if value is not None and value > 0]
    # A positive Manhattan scale is a lower bound only when every traversable
    # cell has positive entry cost.  If a custom map contains a zero-cost
    # ordinary cell, fall back to Dijkstra so the heuristic cannot overestimate
    # a route that uses a chain of free cells.  Official maps are unchanged.
    has_free_cell = any(value == 0 for row in costs for value in row)
    scale = 0 if has_free_cell else min(positive, default=0)
    matrix = []
    for source in points:
        row = []
        for target in points:
            result = astar(
                source,
                target,
                costs,
                lambda point, goal=target: scale * manhattan(point, goal),
            )
            row.append(result.cost)
        matrix.append(tuple(row))
    return tuple(matrix)


@dataclass(frozen=True)
class State:
    a_position: int
    b_position: int
    remaining: tuple[int, ...]
    score_a: int = 0
    score_b: int = 0
    a_turn: bool = True

    @property
    def current_position(self) -> int:
        return self.a_position if self.a_turn else self.b_position

    @property
    def utility(self) -> int:
        return self.score_a - self.score_b


class TreasureGame:
    """Minimax and Alpha-Beta over the treasure-allocation state space."""

    def __init__(self, distances, values):
        size = len(values) + 2
        if len(distances) != size or any(len(row) != size for row in distances):
            raise ValueError("distance matrix dimensions do not match the treasure count")
        self.distances = tuple(tuple(row) for row in distances)
        self.values = tuple(values)
        self.initial = State(0, 1, tuple(range(2, size)))

    def gain(self, state: State, target: int) -> int:
        distance = self.distances[state.current_position][target]
        if distance < 0:
            raise ValueError("unreachable treasure is not a legal action")
        return self.values[target - 2] - distance

    def legal_actions(self, state: State, order: str = "index") -> tuple[int, ...]:
        if order not in {"index", "reverse", "gain"}:
            raise ValueError(f"unknown treasure order: {order}")
        actions = [target for target in state.remaining
                   if self.distances[state.current_position][target] >= 0]
        if order == "reverse":
            actions.reverse()
        elif order == "gain":
            actions.sort(key=lambda target: (-self.gain(state, target), target))
        return tuple(actions)

    def move(self, state: State, target: int) -> State:
        if target not in self.legal_actions(state):
            raise ValueError("target is not available")
        gain = self.gain(state, target)
        remaining = tuple(item for item in state.remaining if item != target)
        if state.a_turn:
            return replace(
                state,
                a_position=target,
                score_a=state.score_a + gain,
                remaining=remaining,
                a_turn=False,
            )
        return replace(
            state,
            b_position=target,
            score_b=state.score_b + gain,
            remaining=remaining,
            a_turn=True,
        )

    @staticmethod
    def _better(value, best_value, action, best_action, maximizing):
        # Alpha-Beta may return a bound for a pruned child. Updating only on a
        # strict improvement prevents an equal bound from replacing an action
        # whose exact value has already established the current optimum.
        return value > best_value if maximizing else value < best_value

    def minimax_search(self, depth: int, *, state: State | None = None,
                       order: str = "index") -> GameResult:
        if depth < 0:
            raise ValueError("depth must be nonnegative")
        root = self.initial if state is None else state
        stats = SearchStats()

        def visit(node: State, remaining_depth: int) -> tuple[int, int | None]:
            stats.visited += 1
            actions = self.legal_actions(node, order) if remaining_depth > 0 else ()
            if not actions:
                stats.leaves += 1
                return node.utility, None

            maximizing = node.a_turn
            best_value = -inf if maximizing else inf
            best_action = None
            for action in actions:
                child_value, _ = visit(self.move(node, action), remaining_depth - 1)
                if self._better(child_value, best_value, action, best_action, maximizing):
                    best_value, best_action = child_value, action
            return int(best_value), best_action

        value, action = visit(root, depth)
        return GameResult(value, action, stats)

    def alpha_beta_search(self, depth: int, *, state: State | None = None,
                          order: str = "index") -> GameResult:
        if depth < 0:
            raise ValueError("depth must be nonnegative")
        root = self.initial if state is None else state
        stats = SearchStats()

        def visit(node: State, remaining_depth: int,
                  alpha: float, beta: float) -> tuple[int, int | None]:
            stats.visited += 1
            actions = self.legal_actions(node, order) if remaining_depth > 0 else ()
            if not actions:
                stats.leaves += 1
                return node.utility, None

            maximizing = node.a_turn
            best_value = -inf if maximizing else inf
            best_action = None
            for position, action in enumerate(actions):
                child_value, _ = visit(
                    self.move(node, action), remaining_depth - 1, alpha, beta
                )
                if self._better(child_value, best_value, action, best_action, maximizing):
                    best_value, best_action = child_value, action

                if maximizing:
                    alpha = max(alpha, best_value)
                else:
                    beta = min(beta, best_value)
                if alpha >= beta and position + 1 < len(actions):
                    stats.cutoffs += 1
                    stats.pruned_children += len(actions) - position - 1
                    break
            return int(best_value), best_action

        value, action = visit(root, depth, -inf, inf)
        return GameResult(value, action, stats)

    def solve(self, depth: int, *, algorithm: str = "alpha_beta",
              order: str = "index") -> dict[str, int | str | None]:
        if algorithm == "minimax":
            result = self.minimax_search(depth, order=order)
        elif algorithm == "alpha_beta":
            result = self.alpha_beta_search(depth, order=order)
        else:
            raise ValueError(f"unknown algorithm: {algorithm}")
        return {"best_target": target_name(result.action), "best_value": result.value}

    def exact_root_values(self, depth: int) -> dict[str, int]:
        if depth <= 0:
            return {}
        values = {}
        for target in self.legal_actions(self.initial):
            child = self.move(self.initial, target)
            values[target_name(target)] = self.minimax_search(depth - 1, state=child).value
        return values

    def optimal_line(self, depth: int) -> list[dict[str, int | str]]:
        state = self.initial
        line = []
        for remaining_depth in range(depth, 0, -1):
            action = self.minimax_search(remaining_depth, state=state).action
            if action is None:
                break
            player = "A" if state.a_turn else "B"
            source = ("A", "B")[state.current_position] if state.current_position < 2 \
                else target_name(state.current_position)
            distance = self.distances[state.current_position][action]
            gain = self.gain(state, action)
            state = self.move(state, action)
            line.append({
                "player": player,
                "source": source,
                "target": target_name(action),
                "distance": distance,
                "gain": gain,
                "score_a": state.score_a,
                "score_b": state.score_b,
                "utility": state.utility,
            })
        return line


def target_name(action: int | None) -> str | None:
    return None if action is None else f"T{action - 1}"
