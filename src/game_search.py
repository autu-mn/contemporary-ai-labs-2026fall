"""Explicit Minimax and Alpha-Beta implementations for Question 3."""

from dataclasses import dataclass, field
from math import inf


@dataclass
class TreeNode:
    value: int | None = None
    children: list["TreeNode"] = field(default_factory=list)


@dataclass
class SearchStats:
    visited: int = 0
    leaves: int = 0
    cutoffs: int = 0
    pruned_children: int = 0


@dataclass
class GameResult:
    value: int
    action: int | None
    stats: SearchStats


def _children(node: TreeNode, order: str) -> list[tuple[int, TreeNode]]:
    if order not in {"left", "reverse"}:
        raise ValueError(f"unknown child order: {order}")
    indexed = list(enumerate(node.children))
    return indexed if order == "left" else indexed[::-1]


def minimax_search(root: TreeNode, *, order: str = "left") -> GameResult:
    """Search the full tree. The root is MAX and levels alternate."""
    stats = SearchStats()

    def visit(node: TreeNode, maximizing: bool) -> tuple[int, int | None]:
        stats.visited += 1
        if not node.children:
            if node.value is None:
                raise ValueError("a leaf must contain a numeric value")
            stats.leaves += 1
            return node.value, None

        best_value = -inf if maximizing else inf
        best_action = None
        for action, child in _children(node, order):
            child_value, _ = visit(child, not maximizing)
            better = child_value > best_value if maximizing else child_value < best_value
            if better or (child_value == best_value and
                          (best_action is None or action < best_action)):
                best_value, best_action = child_value, action
        return int(best_value), best_action

    value, action = visit(root, True)
    return GameResult(value, action, stats)


def alpha_beta_search(root: TreeNode, *, order: str = "left") -> GameResult:
    """Run left-to-right Alpha-Beta unless a comparison order is requested."""
    stats = SearchStats()

    def visit(node: TreeNode, maximizing: bool, alpha: float, beta: float) -> tuple[int, int | None]:
        stats.visited += 1
        if not node.children:
            if node.value is None:
                raise ValueError("a leaf must contain a numeric value")
            stats.leaves += 1
            return node.value, None

        choices = _children(node, order)
        best_value = -inf if maximizing else inf
        best_action = None
        for position, (action, child) in enumerate(choices):
            child_value, _ = visit(child, not maximizing, alpha, beta)
            better = child_value > best_value if maximizing else child_value < best_value
            if better or (child_value == best_value and
                          (best_action is None or action < best_action)):
                best_value, best_action = child_value, action

            if maximizing:
                alpha = max(alpha, best_value)
            else:
                beta = min(beta, best_value)
            if alpha >= beta and position + 1 < len(choices):
                stats.cutoffs += 1
                stats.pruned_children += len(choices) - position - 1
                break
        return int(best_value), best_action

    value, action = visit(root, True, -inf, inf)
    return GameResult(value, action, stats)


def minimax(root: TreeNode) -> int:
    return minimax_search(root).value


def alpha_beta(root: TreeNode) -> int:
    return alpha_beta_search(root).value

