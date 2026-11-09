# 实验二：搜索算法

本项目完成当代人工智能实验课实验二的四道题：题目一、二使用 A* 完成迷宫与加权地形寻路，题目三使用 Minimax / Alpha-Beta 求解博弈树，题目四先用 A* 计算移动代价，再用博弈搜索决定宝藏分配。除官方四题外，项目还包含一个独立的六宝藏访问顺序对照实验。

## 快速运行

算法、测试和实验数据生成仅依赖 Python 标准库（建议 Python 3.11 或更新版本）。在项目根目录运行四道官方题目的直接入口：

```powershell
python q1_maze.py
python q2_terrain.py
python q3_game_tree.py
python q4_treasure.py
```

运行全部测试和统一实验：

```powershell
python -m unittest discover -s tests -v
python -m src.run_experiments --output-dir results --seed 42
```

四个 `q*.py` 分别对应四道官方题目；`src/run_experiments.py` 用于统一验证、运行对照实验并导出结果。测试覆盖官方答案、必要的边界情况，并使用 Dijkstra、完整 Minimax 等独立方法进行交叉验证。

## 四道官方题目

### 题目一：迷宫寻路

`q1_maze.solve(start, end, grid)` 调用四方向 A*，状态为网格坐标，每次移动代价为 1。搜索按 `f=g+h` 选择节点，其中 `g` 是已走步数，`h` 是到终点的曼哈顿距离；`parent` 记录前驱，用于在到达终点后恢复完整最短路径。实现同时保留 `h=0` 的模式，供实验报告比较启发式搜索与 Dijkstra 等价搜索的扩展量。

官方输入的结果是从 `(0, 0)` 到 `(3, 4)` 的 7 步最短路径；接口返回包含起点和终点的坐标列表，无解时返回 `[]`。

### 题目二：加权地形

`q2_terrain.solve(start, end, grid)` 沿用同一套 A* 框架，但 `g` 改为累计“进入下一个格子”的权值：起点不收费，进入 `E` 的最后一步代价为 0。因为终点这一步免费，普通的“最小权值 × 曼哈顿距离”可能高估剩余代价，代码把可收费步数修正为 `max(曼哈顿距离 - 1, 0)`；若自定义地图还有其他零权格，则退回 `h=0` 以保证最优性。

接口返回最小总代价，无解时返回 `-1`；官方输入结果为 **7**。

### 题目三：Minimax 与 Alpha-Beta

`q3_game_tree.py` 使用 `TreeNode(value, children)` 表示题目给定的树：叶节点保存分值，内部节点保存子节点。Minimax 从 MAX 根节点开始，在 MAX 层取最大值、MIN 层取最小值并逐层回传；Alpha-Beta 保持相同博弈语义，通过 `alpha`、`beta` 界限跳过不可能影响最终选择的分支。

官方题按从左到右的顺序访问子节点，`q3_game_tree.solve(root)` 返回根节点最佳得分 **5**。代码中的逆序访问只用于实验对照，用来观察访问顺序对剪枝量的影响，不属于官方求解流程。

### 题目四：竞争环境下的宝藏分配

题目四把路径搜索与博弈搜索分成两个阶段。首先，`distance_matrix` 对 `A、B、T1～T4` 六个关键位置两两运行 A*，得到 `6×6` 最小移动代价矩阵；随后 `TreasureGame` 在这个矩阵上运行 Minimax / Alpha-Beta，无需在每个博弈节点中重复寻路。

博弈状态包含 A、B 的当前位置、剩余宝藏、双方累计收益和当前玩家。玩家从位置 `p` 选择宝藏 `Ti` 后获得 `Vi-d(p,Ti)`，叶节点效用为 `U=ScoreA-ScoreB`；A 最大化 U，B 最小化 U。`H` 表示向前搜索的行动次数：H=2 为 A、B 各行动一次，H=4 为 A→B→A→B 共四回合。

`q4_treasure.solve()` 返回距离矩阵以及两种算法在两个深度下的结果。官方结果为：

- H=2：最优首步 `T2`，最优收益差 `-1`；
- H=4：最优首步 `T3`，最优收益差 `3`。

Minimax 与 Alpha-Beta 必须得到一致的最优首步和根值，区别只体现在搜索开销。

## 自设拓展：六宝藏场景

在官方四宝藏题之外，项目增加了独立的六宝藏场景，并比较 H=2、H=4、H=6 下三种 Alpha-Beta 后继访问顺序：宝藏编号序、逆序，以及按当前玩家即时收益 `Vi-d(p,Ti)` 从高到低排序。

六宝藏地图、H=6 和即时收益排序均为自设拓展，不是官方题目四的要求。详细实验结果与分析见[正式实验报告](report/report.pdf)。

## 项目结构

```text
.
├── q1_maze.py
├── q2_terrain.py
├── q3_game_tree.py
├── q4_treasure.py
├── src/
│   ├── cases.py
│   ├── grid_search.py
│   ├── game_search.py
│   ├── treasure_game.py
│   └── run_experiments.py
├── tests/
│   └── test_lab2.py
├── results/
│   ├── official_results.json
│   ├── grid_comparisons.csv
│   ├── game_comparisons.csv
│   └── final_report/
└── report/
    ├── final_assets/
    └── report.pdf

```

关键模块的职责如下：

- `src/grid_search.py`
  - 实现通用的四方向 A*、优先队列、最优代价记录与路径恢复。
  - 为题目一提供等权迷宫搜索，为题目二提供加权地形搜索。
  - 被题目四复用，用于计算关键位置之间的最小移动代价。
- `src/game_search.py`
  - 定义题目三的 `TreeNode`、搜索结果和统计信息。
  - 分别实现完整 Minimax 与 Alpha-Beta，并支持左右两种访问顺序对照。
- `src/treasure_game.py`
  - 将题目四地图转换为代价格，并生成关键位置距离矩阵。
  - 定义宝藏博弈状态、收益计算、合法行动和状态转移。
  - 实现宝藏场景下的 Minimax、Alpha-Beta 与最优行动序列提取。
- `src/cases.py`
  - 集中保存四道题的官方输入、期望结果与坐标信息。
  - 保存启发式、访问顺序和六宝藏自设实验所需输入。
- `src/run_experiments.py`
  - 在写出结果前验证官方路径、代价、距离矩阵和根动作值。
  - 统一运行启发式、算法、深度与访问顺序对照。
  - 导出 JSON 和 CSV，供复现与正式报告引用。
- `tests/test_lab2.py`
  - 检查四道题的官方样例和无解、零代价等边界情况。
  - 用独立 Dijkstra 和完整 Minimax 交叉验证 A* 与 Alpha-Beta 的正确性。

## 实验结果

- `results/official_results.json`：官方路径、距离矩阵、各根动作值、最优行动及行动序列等详细结果；
- `results/grid_comparisons.csv`：A* 启发式对照数据；
- `results/game_comparisons.csv`：Minimax、Alpha-Beta、搜索深度和访问顺序对照数据；
- `results/final_report/`：最终报告对应的一组独立复现实验数据。

实验脚本会先完成正确性检查，再写出这些文件。传入的 `--seed 42` 会记录在结果中；当前实验本身是确定性的。

## 指标说明

- `expanded`：A* 从优先队列中有效取出并实际扩展的节点数；
- `visited`：博弈搜索递归访问的节点数；
- `leaves`：实际执行效用评价的叶节点数；
- `cutoffs`：至少跳过一个后继的有效剪枝次数；
- `pruned_children`：因剪枝被直接跳过的后继数量。

运行时间只用于辅助观察，会受到机器与系统负载影响；正式报告主要依据节点类指标分析算法性能。

## 正式报告

[PDF 报告](report/report.pdf)
