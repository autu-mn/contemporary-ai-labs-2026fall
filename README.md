# 当代人工智能实验（2026 秋）

华东师范大学「当代人工智能」实验课的个人仓库。`main` 只放课程说明和实验索引；每次实验单独一个分支，代码、数据、结果和报告都留在对应分支里。

## 实验

| 实验 | 分支 | 内容 |
| --- | --- | --- |
| 实验一 | [`lab1-text-classification`](https://github.com/autu-mn/contemporary-ai-labs-2026fall/tree/lab1-text-classification) | 新闻文本 10 分类。比较 MultinomialNB、线性 SVM 和 MLP，最终方案是线性 SVM：删除邮件头，Subject 重复 3 次，拼接词级与字符级 n-gram TF-IDF。 |

后续实验按同样方式从 `main` 开分支，并补进上表。

## 使用

```bash
git clone https://github.com/autu-mn/contemporary-ai-labs-2026fall.git
cd contemporary-ai-labs-2026fall
git checkout lab1-text-classification
```

切到实验分支后，按该分支的 `README.md` 安装依赖、跑实验。实验一里：

```bash
pip install -r requirements.txt
python src/run_all.py    # 按阶段复现全部实验，大约 1 小时
python src/final.py      # 只做最终评估并生成 predictions.csv
```

## 约定

- 一个实验一个分支，不把实验代码直接堆在 `main`。
- 环境、数据划分、选择规则和主要结果写在该实验分支的 `README.md`。
- 方案比较只用验证集；本地测试集在方案锁定之后评估一次。
- 课件不放进本仓库。
