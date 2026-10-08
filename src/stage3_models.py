"""阶段 3：三个模型调参（预处理与特征固定为 config 中阶段 1、2 的结果）。

每个模型的选择规则（事先确定，与阶段 1、2 一致）：
  验证集准确率在该模型最优者 1 个 SE 内的配置为候选；若有开发集 CV，则还要求 CV 均值
  不低于候选中 CV 最高者的 (均值 - 标准差)；候选中取最简单/正则化最强的配置：
    NB  → alpha 最大（平滑最强）
    SVM → C 最小（间隔最宽）
    MLP → 参数量最少，其次 alpha 最大
MLP 的 6 个配置并行训练，fit_sec 受资源竞争影响，最终耗时在阶段 4 单独重测。
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score

from config import PREPROCESS, VEC_KIND, VEC_PARAMS
from data import FIGURES, RESULTS, SEED, split_data
from experiment import TextCache, evaluate, save_rows, setup_log, val_se
from features import make_vectorizer
from models import make_model
from preprocess import Preprocessor

NB_ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
SVM_CS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100]
SVC_LINEAR_CS = [0.3, 1, 10]
# L2 归一化的 TF-IDF 上 gamma='scale' 约等于 1，因此另取 0.1（核更平滑、更接近线性）配合更大的 C
RBF_GRID = [(1, 'scale'), (10, 'scale'), (10, 0.1), (100, 0.1)]
MLP_HIDDEN = [(100,), (256,), (256, 128)]
MLP_ALPHAS = [1e-4, 1e-2]
MLP_BASE = {'early_stopping': True, 'max_iter': 50, 'n_iter_no_change': 5}
EPOCHS = 30


def select(df, n_val, simplest_key):
    se = val_se(df['val_acc'].max(), n_val)
    cand = df[df['val_acc'] >= df['val_acc'].max() - se]
    if 'cv_mean' in cand and cand['cv_mean'].notna().all():
        top = cand.loc[cand['cv_mean'].idxmax()]
        cand = cand[cand['cv_mean'] >= top['cv_mean'] - top['cv_std']]
    return cand.sort_values(simplest_key[0], ascending=simplest_key[1]).iloc[0], se


def tune_nb(cache, prep):
    rows = []
    for a in NB_ALPHAS:
        row, _, _ = evaluate(cache, 'NB', f'alpha={a}', prep, VEC_KIND, VEC_PARAMS, 'nb', {'alpha': a}, cv=True)
        rows.append({**row, 'alpha': a})
    # 旁证：阶段 2 用 SVM 选特征，检查 sublinear_tf 对 NB 是否有不同影响
    row, _, _ = evaluate(cache, 'NB-check', 'alpha=0.1 sublinear_tf=True', prep, VEC_KIND,
                         {**VEC_PARAMS, 'sublinear_tf': True}, 'nb', {'alpha': 0.1}, cv=True)
    return pd.DataFrame(rows), row


def tune_svm(cache, prep):
    rows = []
    for c in SVM_CS:
        row, _, _ = evaluate(cache, 'SVM-linear', f'LinearSVC C={c}', prep, VEC_KIND, VEC_PARAMS,
                             'linsvc', {'C': c, 'max_iter': 5000}, cv=True)
        rows.append({**row, 'C': c})
    return pd.DataFrame(rows)


def kernel_compare(cache, prep):
    """在同一实现 (libsvm SVC) 内比较线性核与 RBF 核，各自调 C。
    LinearSVC（平方合页损失 + 一对多）与 SVC（合页损失 + 一对一）的 C 尺度不同，不能直接沿用。"""
    rows = []
    for c in SVC_LINEAR_CS:
        row, _, _ = evaluate(cache, 'SVM-kernel', f'SVC linear C={c}', prep, VEC_KIND, VEC_PARAMS,
                             'svc', {'kernel': 'linear', 'C': c})
        rows.append({**row, 'kernel': 'linear', 'C': c, 'gamma': '-'})
    for c, g in RBF_GRID:
        row, _, _ = evaluate(cache, 'SVM-kernel', f'SVC rbf C={c} gamma={g}', prep, VEC_KIND, VEC_PARAMS,
                             'svc', {'kernel': 'rbf', 'C': c, 'gamma': g})
        rows.append({**row, 'kernel': 'rbf', 'C': c, 'gamma': g})
    return pd.DataFrame(rows)


def _mlp_job(cache, prep, h, a):
    row, _, _ = evaluate(cache, 'MLP', f'hidden={h} alpha={a}', prep, VEC_KIND, VEC_PARAMS,
                         'mlp', {'hidden_layer_sizes': h, 'alpha': a, **MLP_BASE})
    return {**row, 'hidden': str(h), 'alpha': a}


def tune_mlp(cache, prep):
    for part in ('train', 'val'):
        cache.get(prep, part)
    jobs = [(h, a) for h in MLP_HIDDEN for a in MLP_ALPHAS]
    rows = Parallel(n_jobs=len(jobs), backend='threading')(
        delayed(_mlp_job)(cache, prep, h, a) for h, a in jobs)
    return pd.DataFrame(rows)


def mlp_epoch_curve(cache, prep, hidden, alpha):
    """不使用早停，逐轮 partial_fit，记录训练/验证准确率与损失，观察多训练是否带来提升或过拟合。"""
    X_tr, y_tr = cache.get(prep, 'train'), cache.y('train')
    X_va, y_va = cache.get(prep, 'val'), cache.y('val')
    vec = make_vectorizer(VEC_KIND, **VEC_PARAMS)
    A, B = vec.fit_transform(X_tr), vec.transform(X_va)
    clf = make_model('mlp', hidden_layer_sizes=hidden, alpha=alpha)
    classes = np.unique(y_tr)
    rows = []
    for ep in range(1, EPOCHS + 1):
        clf.partial_fit(A, y_tr, classes=classes)
        rows.append(dict(epoch=ep, loss=clf.loss_,
                         train_acc=accuracy_score(y_tr, clf.predict(A)),
                         val_acc=accuracy_score(y_va, clf.predict(B))))
        print(f'[MLP-epoch] {ep:2d} loss={clf.loss_:.4f} train={rows[-1]["train_acc"]:.4f} val={rows[-1]["val_acc"]:.4f}', flush=True)
    return pd.DataFrame(rows)


def plot(nb, svm, kern, mlp, curve):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax = axes[0, 0]
    ax.semilogx(nb['alpha'], nb['train_acc'], 'o--', label='train')
    ax.semilogx(nb['alpha'], nb['val_acc'], 'o-', label='val')
    ax.errorbar(nb['alpha'], nb['cv_mean'], yerr=nb['cv_std'], fmt='s:', label='dev 5-fold CV', capsize=3)
    ax.set_xlabel('alpha (smoothing)')
    ax.set_title('MultinomialNB')
    ax.set_ylabel('accuracy')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.semilogx(svm['C'], svm['train_acc'], 'o--', label='train')
    ax.semilogx(svm['C'], svm['val_acc'], 'o-', label='val')
    ax.errorbar(svm['C'], svm['cv_mean'], yerr=svm['cv_std'], fmt='s:', label='dev 5-fold CV', capsize=3)
    ax.set_xlabel('C')
    ax.set_title('Linear SVM (LinearSVC)')
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    labels = [f"{h}\nα={a:g}" for h, a in zip(mlp['hidden'], mlp['alpha'])]
    x = np.arange(len(mlp))
    ax.bar(x - 0.2, mlp['train_acc'], 0.4, label='train')
    ax.bar(x + 0.2, mlp['val_acc'], 0.4, label='val')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0.85, 1.03)
    ax.set_title('MLP (early stopping)')
    ax.set_ylabel('accuracy')
    ax.legend(fontsize=8, ncol=2, loc='upper center')

    ax = axes[1, 1]
    ax.plot(curve['epoch'], curve['train_acc'], label='train acc')
    ax.plot(curve['epoch'], curve['val_acc'], label='val acc')
    ax.set_xlabel('epoch (no early stopping)')
    ax.set_ylabel('accuracy')
    ax2 = ax.twinx()
    ax2.plot(curve['epoch'], curve['loss'], 'k:', label='train loss')
    ax2.set_ylabel('loss')
    ax.set_title('Best MLP: accuracy & loss vs epoch')
    ax.legend(fontsize=8, loc='center right')
    fig.tight_layout()
    fig.savefig(FIGURES / 'stage3_models.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    names = [n.replace('SVC ', '') for n in kern['name']]
    ax.barh(names, kern['val_acc'], color=['C0' if k == 'linear' else 'C3' for k in kern['kernel']])
    for i, (acc, t) in enumerate(zip(kern['val_acc'], kern['fit_sec'])):
        ax.text(acc + 0.002, i, f'{acc:.4f} | {t:.0f}s', va='center', fontsize=8)
    ax.set_xlim(kern['val_acc'].min() - 0.05, 1.0)
    ax.set_xlabel('val accuracy | fit time')
    ax.set_title('SVC kernel comparison')
    fig.tight_layout()
    fig.savefig(FIGURES / 'stage3_kernels.png', dpi=150)
    plt.close(fig)


def main():
    setup_log('stage3')
    splits = split_data()
    cache = TextCache(splits)
    prep = Preprocessor(**PREPROCESS)
    n_val = len(splits['val'][1])

    nb, nb_check = tune_nb(cache, prep)
    nb_best, se = select(nb, n_val, ('alpha', False))
    print(f'  → NB 选择 alpha={nb_best["alpha"]} val={nb_best["val_acc"]:.4f} cv={nb_best["cv_mean"]:.4f} (SE≈{se:.4f})\n')

    svm = tune_svm(cache, prep)
    svm_best, se = select(svm, n_val, ('C', True))
    print(f'  → SVM 选择 C={svm_best["C"]} val={svm_best["val_acc"]:.4f} cv={svm_best["cv_mean"]:.4f} (SE≈{se:.4f})\n')

    kern = kernel_compare(cache, prep)

    mlp = tune_mlp(cache, prep)
    mlp_best, se = select(mlp, n_val, (['n_params', 'alpha'], [True, False]))
    print(f'  → MLP 选择 hidden={mlp_best["hidden"]} alpha={mlp_best["alpha"]} val={mlp_best["val_acc"]:.4f} (SE≈{se:.4f})\n')

    curve = mlp_epoch_curve(cache, prep, eval(mlp_best['hidden']), mlp_best['alpha'])

    save_rows(nb.to_dict('records') + [nb_check], 'stage3_nb.csv')
    save_rows(svm.to_dict('records'), 'stage3_svm.csv')
    save_rows(kern.to_dict('records'), 'stage3_kernels.csv')
    save_rows(mlp.to_dict('records'), 'stage3_mlp.csv')
    save_rows(curve.to_dict('records'), 'stage3_mlp_epochs.csv')
    plot(nb, svm, kern, mlp, curve)

    selected = {
        'nb': {'alpha': float(nb_best['alpha'])},
        'linsvc': {'C': float(svm_best['C']), 'max_iter': 5000},
        'mlp': {'hidden_layer_sizes': list(eval(mlp_best['hidden'])), 'alpha': float(mlp_best['alpha']), **MLP_BASE},
    }
    (RESULTS / 'stage3_selected.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    print('各模型选定参数:', selected)


if __name__ == '__main__':
    main()
