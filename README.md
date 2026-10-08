# 实验一：新闻文本 10 分类

使用 TF-IDF 特征，比较 MultinomialNB（生成式）、线性 SVM（判别式最大间隔）、MLP（非线性神经网络）三类模型，
并在验证集上完成预处理、特征和超参数选择。最终方案为 **线性 SVM + 删除邮件头 + Subject 重复 3 次 + 词级与字符级 n-gram TF-IDF**。

## 环境

```bash
conda create -n project1 python=3.11
conda activate project1
pip install -r requirements.txt
```

实验环境：Windows 11，Python 3.11.9，依赖版本见 `requirements.txt`。词干化使用 nltk 的 PorterStemmer，无需额外下载语料。

## 运行

所有命令在项目根目录执行，代码只使用相对路径。

```bash
python src/run_all.py          # 按顺序复现全部实验（约 1 小时）
python src/final.py            # 只做最终评估并生成 predictions.csv（约 4 分钟）
python src/test_preprocess.py  # 预处理边界用例检查
```

也可以单独运行某一阶段，例如 `python src/stage3_models.py`。阶段 1–4 的选择结论固化在 `src/config.py` 中，
`final.py` 运行前会核对它们与阶段 3、4 输出的选择文件是否一致。

## 数据划分与使用规则

`train_data.csv`（7368 条）按 70/15/15 分层划分，`random_state=42`：

| 子集 | 条数 | 用途 |
|---|---|---|
| train | 5156 | 拟合向量化器（词表、IDF）与模型参数 |
| val | 1106 | 所有方案比较与超参数选择 |
| test_local | 1106 | 方案锁定后只评估一次，作为最终性能 |
| 官方测试集 | 2457 | 只用最终模型预测一次，生成 `predictions.csv` |

- 向量化器与分类器放在同一个 `Pipeline` 中，只在当前训练部分上 `fit`；交叉验证只在 train+val（开发集）内进行。
- 预处理（删邮件头、删引用、词干化等）都是逐条规则，不使用语料统计量，因此可以统一应用到各子集。
- 选择规则：验证集准确率在最优者 1 个标准误内、且开发集 5 折 CV 在参照配置 1 个标准差内的方案视为等价，
  取其中最简单（正则化最强 / 维度最小 / 耗时最短）的方案。

## 代码结构

```text
src/
├── data.py               数据加载与 70/15/15 分层划分
├── preprocess.py         逐条清洗（邮件头、引用、噪声、停用词、词干化、Subject 加权）
├── features.py           TF-IDF / 词袋 / 词级+字符级向量化器
├── models.py             NB、LinearSVC、SVC、MLP、硬投票集成；参数量统计
├── experiment.py         统一的训练-评估-记录函数（train 拟合，val 评估，可选开发集 CV）
├── config.py             各阶段锁定的配置
├── test_preprocess.py    预处理边界用例检查
├── stage0_eda_baseline.py  划分检查、训练集数据探查、E0 基线
├── stage1_preprocess.py    预处理消融
├── stage2_features.py      特征工程
├── stage3_models.py        三个模型调参、核函数对比、MLP 逐轮曲线
├── stage4_analysis.py      CV 稳健性、误差分析、权重词、创新实验、最终模型选择
├── final.py                本地测试集评估一次 + 全量重训 + 生成 predictions.csv
└── run_all.py              一键复现
results/
├── stage*_*.csv / *.json   各阶段完整结果
├── figures/                图表
└── logs/                   各阶段运行日志
```

## 主要结果

| 模型 | 验证集 Acc | 开发集 5 折 CV | 本地测试集 Acc | 参数量 | 训练耗时 (dev) |
|---|---|---|---|---|---|
| MultinomialNB (alpha=0.1) | 0.9412 | 0.9376 ± 0.0065 | 0.9250 | 0.38M | 约 3 s |
| Linear SVM (C=0.3) | 0.9331 | 0.9348 ± 0.0080 | 0.9385 | 0.38M | 约 2 s |
| MLP (100,), alpha=1e-4 | 0.9430 | 0.9353 ± 0.0083 | 0.9385 | 3.77M | 约 88 s |
| **最终方案** | **0.9548** | **0.9495 ± 0.0034** | **0.9530** | 1.38M | 约 46 s |

三个基础模型的准确率在统计上无显著差异，差异主要体现在复杂度上（MLP 参数量约为其余两者的 10 倍，训练耗时约为 30–40 倍）。
最终方案的提升来自 Subject 加权与字符 n-gram 两项特征改进。训练耗时受机器负载影响有波动，仅供数量级比较。

## 复现说明

- 所有随机过程（数据划分、LinearSVC、SVC、MLP、交叉验证的打乱）固定 `random_state=42`。
- MLP 的阶段 3 调参使用多线程并行训练，记录的耗时偏大；阶段 4、5 为单独计时。
- 多线程 BLAS 可能导致 MLP 的结果在不同机器上有极小的浮点差异。
- 提交文件 `predictions.csv`：2457 行、单列、无表头、整数标签 0–9，顺序与 `test_data_unlabeled.csv` 一致。
