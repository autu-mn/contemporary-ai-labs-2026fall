"""Distance preprocessing and adversarial allocation of treasures."""

from dataclasses import dataclass, replace
from time import perf_counter_ns

from .game_search import search_game
from .grid_search import astar, manhattan


def distance_matrix(grid, points):
    """Compute ordered distances, excluding the source's entry cost."""
    costs = [[None if cell == "X" else cell for cell in row] for row in grid]
    minimum = min(cost for row in costs for cost in row if cost is not None)
    return tuple(
        tuple(astar(source, target, costs, lambda p: minimum * manhattan(p, target)).cost
              for target in points)
        for source in points
    )


@dataclass(frozen=True)
class State:
    a: int
    b: int
    remaining: tuple[int, ...]
    score_a: int = 0
    score_b: int = 0
    a_turn: bool = True

    @property
    def position(self):
        return self.a if self.a_turn else self.b

    @property
    def utility(self):
        return self.score_a - self.score_b


class TreasureGame:
    def __init__(self, distances, values):
        self.distances = distances
        self.values = values
        self.initial = State(0, 1, tuple(range(2, 2 + len(values))))

    def gain(self, state, target):
        return self.values[target - 2] - self.distances[state.position][target]

    def legal(self, state):
        return tuple(target for target in state.remaining
                     if self.distances[state.position][target] >= 0)

    def move(self, state, target):
        gain = self.gain(state, target)
        remaining = tuple(other for other in state.remaining if other != target)
        if state.a_turn:
            return replace(state, a=target, score_a=state.score_a + gain,
                           remaining=remaining, a_turn=False)
        return replace(state, b=target, score_b=state.score_b + gain,
                       remaining=remaining, a_turn=True)

    def search(self, depth, *, state=None, prune=True, order="index", sort_samples=None):
        if order not in ("index", "reverse", "gain"):
            raise ValueError(f"unknown treasure order: {order}")
        if state is None:
            state = self.initial

        def actions(node, is_root):
            choices = self.legal(node)
            if is_root or order == "index":
                return choices
            if order == "reverse":
                return choices[::-1]
            if sort_samples is None:
                return sorted(choices, key=lambda target: -self.gain(node, target))
            started = perf_counter_ns()
            ordered = sorted(choices, key=lambda target: -self.gain(node, target))
            sort_samples.append(perf_counter_ns() - started)
            return ordered

        return search_game(state, lambda node: node.utility, actions, self.move,
                           depth=depth, prune=prune, maximizing=state.a_turn)

    def solve(self, depth, *, prune=True, order="index"):
        result = self.search(depth, prune=prune, order=order)
        return {"best_target": target_name(result.action), "best_value": result.value}

    def exact_root_values(self, depth):
        if depth <= 0:
            return {}
        return {
            target_name(target): self.search(depth - 1, state=self.move(self.initial, target),
                                             prune=False).value
            for target in self.legal(self.initial)
        }

    def optimal_line(self, depth):
        state = self.initial
        line = []
        for remaining in range(depth, 0, -1):
            action = self.search(remaining, state=state, prune=False).action
            if action is None:
                break
            distance = self.distances[state.position][action]
            player = "A" if state.a_turn else "B"
            source = ("A", "B")[state.position] if state.position < 2 else target_name(state.position)
            gain = self.gain(state, action)
            state = self.move(state, action)
            line.append(dict(player=player, source=source, target=target_name(action),
                             distance=distance, gain=gain, score_a=state.score_a,
                             score_b=state.score_b, utility=state.utility))
        return line


def target_name(action):
    return None if action is None else f"T{action - 1}"
