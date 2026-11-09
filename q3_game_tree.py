"""Direct entry for Question 3."""

from src.cases import OFFICIAL_TREE_GROUPS, ORDERING_TREE_GROUPS
from src.game_search import TreeNode, alpha_beta


def _max_node(values):
    return TreeNode(children=[TreeNode(value) for value in values])


def build_official_tree() -> TreeNode:
    """Follow the handout code, including its shared [6, 9] subtree."""
    max1 = _max_node(OFFICIAL_TREE_GROUPS[0])
    max2 = _max_node(OFFICIAL_TREE_GROUPS[1])
    max3 = _max_node(OFFICIAL_TREE_GROUPS[2])
    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])
    return TreeNode(children=[min1, min2])


def build_ordering_tree() -> TreeNode:
    min_nodes = []
    for groups in ORDERING_TREE_GROUPS:
        min_nodes.append(TreeNode(children=[_max_node(values) for values in groups]))
    return TreeNode(children=min_nodes)


def solve(root: TreeNode) -> int:
    return alpha_beta(root)


if __name__ == "__main__":
    print(solve(build_official_tree()))

