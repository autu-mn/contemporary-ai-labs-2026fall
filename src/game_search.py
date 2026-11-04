"""Shared minimax recursion for explicit trees and implicit game states."""

from dataclasses import dataclass, field
from math import inf
from random import Random


@dataclass
class TreeNode:
    value: int | None = None
    children: list["TreeNode"] = field(default_factory=list)


@dataclass
class GameStats:
    visited: int = 0
    leaves: int = 0
    cutoffs: int = 0
    skipped_children: int = 0


@dataclass
class GameResult:
    value: int
    action: object
    stats: GameStats


def search_game(root, evaluate, actions, advance, *, depth=None, prune=True, maximizing=True):
    """Callbacks expose a terminal score, ordered actions, and a successor."""
    stats = GameStats()

    def visit(state, maximizing, remaining, alpha, beta, is_root=False):
        stats.visited += 1
        choices = actions(state, is_root) if remaining != 0 else ()
        if not choices:
            stats.leaves += 1
            return evaluate(state), None

        value = -inf if maximizing else inf
        best_action = None
        next_depth = None if remaining is None else remaining - 1
        for i, action in enumerate(choices):
            child_value, _ = visit(advance(state, action), not maximizing, next_depth, alpha, beta)
            if (maximizing and child_value > value) or (not maximizing and child_value < value):
                value, best_action = child_value, action
            if prune:
                if maximizing:
                    alpha = max(alpha, value)
                else:
                    beta = min(beta, value)
                if alpha >= beta and i + 1 < len(choices):
                    stats.cutoffs += 1
                    stats.skipped_children += len(choices) - i - 1
                    break
        return value, best_action

    if depth is not None and depth < 0:
        raise ValueError("depth must be nonnegative")
    value, action = visit(root, maximizing, depth, -inf, inf, True)
    return GameResult(value, action, stats)


def tree_from_json(payload):
    def build(node, maximizing):
        if "children" not in node:
            return TreeNode(node["value"])
        if node.get("is_maximizing_player", maximizing) != maximizing:
            raise ValueError("players must alternate from a MAX root")
        return TreeNode(children=[build(child, not maximizing) for child in node["children"]])

    return build(payload["tree_structure"], True)


def tree_search(root, *, prune=True, order="left", seed=42):
    if order not in ("left", "reverse", "shuffle"):
        raise ValueError(f"unknown tree order: {order}")
    rng = Random(seed)

    def actions(node, _):
        choices = list(range(len(node.children)))
        if order == "reverse":
            choices.reverse()
        elif order == "shuffle":
            rng.shuffle(choices)
        return choices

    def evaluate(node):
        if node.value is None:
            raise ValueError("leaf must have a score")
        return node.value

    return search_game(root, evaluate, actions, lambda node, i: node.children[i], prune=prune)


def minimax(root) -> int:
    return tree_search(root, prune=False).value


def alpha_beta(root) -> int:
    return tree_search(root).value
