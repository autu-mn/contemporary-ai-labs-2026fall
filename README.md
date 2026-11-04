# 实验二 搜索算法

当代人工智能实验课，2026 年秋季。实现四向网格 A*、Minimax 和 Alpha-Beta，并在宝藏分配问题中组合两类搜索。

## 复现

在项目根目录使用 Python 3.11 或更新版本运行。测试与对照实验只依赖标准库。

```powershell
python -m unittest discover -s tests -v
python -m src.run_experiments
```

实验入口先核对官方答案及对照组一致性，然后写出 10 条网格记录、24 条博弈记录和 JSON 详情。完整计时包含多批重复执行，耗时随机器负载变化。

已生成的 Word 报告位于 `report/实验二搜索算法实验报告.docx`。如需按新一轮结果重建报告，安装可选的报告依赖后在根目录运行。

```powershell
python -m pip install -r report/requirements.txt
python -m report.build_report
```

## 四题接口

| 题目 | 接口 | 输入与输出 |
|---|---|---|
| 1 迷宫 | `src.grid_search.maze_path(start, end, grid)` | 0 起始的 `(row, col)`、二维字符地图。返回含起终点的坐标列表，无解为 `[]` |
| 2 加权地形 | `src.grid_search.minimum_terrain_cost(start, end, grid)` | 数字或数字字符串表示进入代价。初始代价为 0，标记格 `E` 进入代价为 0。返回整数代价，无解为 `-1` |
| 3 博弈树 | `src.game_search.alpha_beta(root)` | 接受带 `children` 和 `value` 的根节点，MAX 开局，从左到右，返回根值。可用 `tree_from_json` 读取题面结构 |
| 4 宝藏 | `src.treasure_game.distance_matrix(grid, points)` 和 `TreasureGame(matrix, values).solve(depth)` | 地图中的 A、B、T 格归一成 1，点顺序为 A、B、T1…，值按宝藏编号排列。返回 `best_target` 与 `best_value`，传 `prune=False` 使用 Minimax |

`X` 在网格中表示不可通行。题二支持普通零权格，有额外零权格时采用零启发式。综合题根层固定按编号访问，等分时选择编号最小的宝藏。深层的 `index`、`reverse` 和 `gain` 分别表示编号、逆序及当前行动方的即时收益降序。自设地图若当前玩家没有可达目标，按当前收益差终止。

## 代码结构

```text
.
├── README.md                         项目说明、运行方法与结果索引
├── src/
│   ├── cases.py                      官方样例、自设对照地图和预期结果
│   ├── grid_search.py                通用 A*、迷宫寻路和加权地形搜索
│   ├── game_search.py                通用 Minimax、Alpha-Beta 和博弈树接口
│   ├── treasure_game.py              距离矩阵、宝藏状态转移与对抗搜索
│   └── run_experiments.py            正确性验收、计时对照与结果导出
├── tests/
│   └── test_search.py                官方样例、边界用例与随机交叉校验
├── results/
│   ├── grid_comparisons.csv          A* 启发式与加权地形对照数据
│   ├── game_comparisons.csv          Minimax、剪枝、深度和排序对照数据
│   └── details.json                  路径、距离矩阵、根动作值与最优应对
└── report/
    └──  report.pdf                    最终实验报告
```

### `src/grid_search.py`

该模块负责题目一和题目二，并为题目四提供最短路计算能力。

- `GridResult` 统一保存路径、总代价、展开节点数、堆峰值、过期堆项数量和终点弹出次数。
- `astar` 是通用四连通网格搜索。调用者提供每个格子的进入代价和启发函数，模块负责优先队列、`g_score`、前驱回溯和过期堆项处理。
- `maze_search` 把非障碍格统一设为代价 1，可切换曼哈顿距离或零启发式。`maze_path` 是题目一要求的简化接口，只返回坐标序列。
- `terrain_costs` 将数字、数字字符串、`S`、`E` 和 `X` 转换为内部代价地图，并检查非法输入。
- `terrain_search` 使用进入目标格的权值累计代价。`E` 的进入代价为 0，并使用扣除免费终点一步后的可采纳启发式。`minimum_terrain_cost` 是题目二要求的简化接口，只返回最小总代价或 `-1`。

### `src/game_search.py`

该模块实现题目三，同时提供可复用于题目四的通用博弈搜索框架。

- `TreeNode` 表示题面给出的博弈树。叶子使用 `value`，非叶节点使用 `children`。
- `GameStats` 和 `GameResult` 保存根值、首个最优动作、访问节点、叶评估、截断次数和跳过的子节点数。
- `search_game` 通过“评价函数、动作生成器、状态转移函数”三个回调搜索显式树或隐式状态空间。`prune=False` 时执行 Minimax，`prune=True` 时执行 Alpha-Beta。
- `tree_from_json` 将题面 JSON 转为 `TreeNode`，并检查 MAX、MIN 是否逐层交替。
- `tree_search` 支持左至右、逆序和固定随机种子的访问顺序。`minimax` 与 `alpha_beta` 是题目三的直接接口。

### `src/treasure_game.py`

该模块完成题目四，将网格最短路和对抗搜索连接起来。

- `distance_matrix` 对 A、B 和全部宝藏计算有序点对最短距离，保留进格代价可能造成的方向差异。
- `State` 是不可变博弈状态，记录 A、B 的当前位置、剩余宝藏、双方累计收益和当前行动方。`utility` 返回 `Score_A - Score_B`。
- `TreasureGame.move` 按 `宝藏价值 - 最短移动代价` 更新行动方收益和位置，并从剩余集合中移除目标。
- `TreasureGame.search` 在指定深度运行 Minimax 或 Alpha-Beta，并支持编号、逆序和即时收益三种深层访问顺序。
- `solve` 返回题目要求的 `best_target` 和 `best_value`。`exact_root_values` 为每个首步重新打开完整搜索窗口，得到可用于报告的精确回传值。`optimal_line` 输出一条完整最优应对序列。

### `src/cases.py`

该模块集中保存实验输入，使测试、实验脚本和报告使用同一份数据。

- `MAZE`、`TERRAIN`、`TREE` 和 `TREASURE_MAP` 是从题面转录的四道官方样例。
- `EXPECTED_DISTANCES` 与 `EXPECTED_ROOT_VALUES` 是综合题的验收基准。
- `OPEN`、`DETOUR` 和 `SHORTCUT` 用于分析启发函数和加权路径选择。
- `EXTENSION_MAP`、`EXTENSION_POINTS` 和 `EXTENSION_VALUES` 构成六宝藏拓展实验。
- `numeric_grid` 将字符串地图中的数字转为整数，同时保留障碍标记。

### `src/run_experiments.py`

该模块是一键复现实验的入口，先检查结果，再导出数据。

- `grid_comparisons` 比较迷宫与加权地形中的零启发式和 A* 启发式，记录代价、步数、展开数、堆峰值和耗时。
- `game_comparisons` 比较 Minimax、Alpha-Beta、搜索深度和访问顺序，并运行六宝藏拓展实验。
- `measure` 预热后执行 7 批重复测量，保存每次运行耗时的中位数。
- `require` 在导出前核对官方答案、距离矩阵、根动作值和不同算法的一致性。任何验收失败都会终止实验。
- `main` 将汇总数据写入 `results/`，终端同时显示四题的关键答案。

### 测试、结果与报告

- `tests/test_search.py` 的 `GridTests` 覆盖官方迷宫、无解、起终点相同、加权地形、免费终点和随机地图的启发式一致性。`GameTests` 覆盖官方树、负数与零、相等值剪枝、随机树交叉校验、官方宝藏题、有向距离、不可达目标和等分首步。
- `results/grid_comparisons.csv` 与 `results/game_comparisons.csv` 是报告表格的原始数据。`results/details.json` 保存不适合放入扁平 CSV 的逐步路径、矩阵和最优行动序列。

## 样例结果

| 题目 | 结果 |
|---|---|
| 1 | 7 步，8 个坐标 |
| 2 | 代价 7，含 8 次移动 |
| 3 | 根值 5 |
| 4 | H=2 为 `T2 / -1`，H=4 为 `T3 / 3`，两种算法一致 |

## 文件和指标

- `src/cases.py` 包含官方样例及自设地图，`tests/test_search.py` 包含边界、官方答案和固定种子的随机交叉校验。
- `results/grid_comparisons.csv` 保存代价、步数、展开、堆峰值、过时项弹出及耗时。展开数只计实际生成邻居的节点。
- `results/game_comparisons.csv` 保存首步、收益、访问数、叶评估、剪枝及耗时。只有实际跳过后继才记一次截断。矩阵、搜索和端到端时间独立测量。即时排序探测单独运行，包含计时调用开销。
- `results/details.json` 保存环境、逐格路径代价、距离矩阵、精确根动作值和最优应对。深度和排序对照的原始记录可以直接从 CSV 筛选。
