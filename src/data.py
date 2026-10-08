"""数据加载与 70/15/15 分层划分。

职责约定：
- train      : 拟合向量化器与模型参数
- val        : 所有方案比较与超参数选择
- test_local : 方案锁定后只评估一次
- dev        : train + val，用于开发集内交叉验证与锁定后的重训
官方 test_data_unlabeled.csv 只在 final.py 中做一次预测。
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
FIGURES = RESULTS / 'figures'
SEED = 42


def load_train():
    df = pd.read_csv(ROOT / 'train_data.csv')
    return df['text'].astype(str), df['target'].astype(int)


def load_official_test():
    return pd.read_csv(ROOT / 'test_data_unlabeled.csv')['text'].astype(str)


def split_data(seed=SEED):
    X, y = load_train()
    X_dev, X_te, y_dev, y_te = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_dev, y_dev, test_size=0.15 / 0.85, stratify=y_dev, random_state=seed)
    return {
        'train': (X_tr, y_tr),
        'val': (X_va, y_va),
        'test_local': (X_te, y_te),
        'dev': (X_dev, y_dev),
    }
