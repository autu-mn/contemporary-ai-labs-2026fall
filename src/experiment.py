"""统一的训练-评估-记录工具。

evaluate() 只用 train 拟合、只在 val 上评估；cv=True 时额外在 dev(train+val) 内做分层 K 折交叉验证。
本地测试集与官方测试集不会出现在这里。
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from data import RESULTS, SEED
from features import make_vectorizer, n_features
from models import make_model, n_params


class TextCache:
    """缓存清洗后的文本，避免同一预处理在多个实验中重复计算。"""

    def __init__(self, splits):
        self.splits = splits
        self._cache = {}

    def get(self, prep, part):
        key = (repr(prep), part)
        if key not in self._cache:
            X, _ = self.splits[part]
            self._cache[key] = X.map(prep) if prep is not None else X
        return self._cache[key]

    def y(self, part):
        return self.splits[part][1]


def build_pipeline(vec_kind, vec_params, model, model_params):
    return Pipeline([
        ('vec', make_vectorizer(vec_kind, **vec_params)),
        ('clf', make_model(model, **model_params)),
    ])


def evaluate(cache, group, name, prep, vec_kind, vec_params, model, model_params,
             cv=False, cv_folds=5, cv_jobs=5):
    X_tr, y_tr = cache.get(prep, 'train'), cache.y('train')
    X_va, y_va = cache.get(prep, 'val'), cache.y('val')
    pipe = build_pipeline(vec_kind, vec_params, model, model_params)

    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    fit_sec = time.time() - t0
    t0 = time.time()
    p_va = pipe.predict(X_va)
    pred_sec = time.time() - t0
    p_tr = pipe.predict(X_tr)

    row = dict(
        group=group, name=name, model=model,
        train_acc=accuracy_score(y_tr, p_tr),
        val_acc=accuracy_score(y_va, p_va),
        val_f1=f1_score(y_va, p_va, average='macro'),
        n_features=n_features(pipe['vec']),
        n_params=n_params(pipe['clf']),
        fit_sec=round(fit_sec, 2), pred_sec=round(pred_sec, 3),
        prep=repr(prep), vec_kind=vec_kind, vec_params=str(vec_params), model_params=str(model_params),
    )
    clf = pipe['clf']
    if hasattr(clf, 'n_iter_'):
        row['n_iter'] = int(np.max(clf.n_iter_))

    if cv:
        X_dev, y_dev = cache.get(prep, 'dev'), cache.y('dev')
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
        scores = cross_val_score(build_pipeline(vec_kind, vec_params, model, model_params),
                                 X_dev, y_dev, cv=skf, scoring='accuracy', n_jobs=cv_jobs)
        row['cv_mean'] = scores.mean()
        row['cv_std'] = scores.std()

    msg = (f'[{group}] {name:<34s} train={row["train_acc"]:.4f} val={row["val_acc"]:.4f} '
           f'f1={row["val_f1"]:.4f} dim={row["n_features"]:>6d} fit={row["fit_sec"]:.1f}s')
    if cv:
        msg += f' cv={row["cv_mean"]:.4f}±{row["cv_std"]:.4f}'
    print(msg, flush=True)
    return row, pipe, p_va


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def setup_log(stage):
    """同时输出到控制台与 results/logs/<stage>.log（UTF-8）。"""
    import sys
    log_dir = RESULTS / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding='utf-8')
    f = open(log_dir / f'{stage}.log', 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.stdout, f)


def save_rows(rows, filename):
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / filename, index=False, encoding='utf-8-sig')
    return df


def val_se(acc, n):
    """验证集准确率的二项分布标准误，用于判断差距是否超出抽样噪声。"""
    return float(np.sqrt(acc * (1 - acc) / n))
