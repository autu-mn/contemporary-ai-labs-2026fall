"""阶段 2：特征工程（逐项贪心，单因素对照）。

预处理固定为 config.PREPROCESS，分类器固定为 LinearSVC(C=1)。
每一步只改一个向量化参数，其余沿用上一步已选定的值。

选择规则：
  验证集准确率在本步最优者 1 个 SE 内、且开发集 CV 均值不低于"本阶段至今 CV 最高配置"的
  (均值 - 标准差) 的选项视为等价；等价选项中取特征维度最小者，维度相同则取列表中靠前（默认）的选项；
  若没有等价选项，取本步 CV 均值最高者。
  参照"本阶段至今最优"而不是"本步最优"，是为了防止每步都在噪声范围内选更简单的选项而使性能逐步累积下滑
  （初版按本步最优判断时，最终配置的 CV 比本阶段最优低了 1 个标准差以上）。
NB 的验证集准确率只作旁证，不参与选择。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import PREPROCESS
from data import FIGURES, split_data
from experiment import TextCache, evaluate, save_rows, setup_log, val_se
from preprocess import Preprocessor

# 未归一化的 count/binary 特征在默认 max_iter=1000 下不收敛；已收敛的配置不受该上限影响
SVM = ('linsvc', {'C': 1.0, 'max_iter': 20000})
NB = ('nb', {'alpha': 0.1})

# (步骤名, 参数名, 候选值列表；第一个为默认值)
STEPS = [
    ('F1 representation', 'kind', ['tfidf', 'tf', 'count', 'binary']),
    ('F2 ngram_range', 'ngram_range', [(1, 1), (1, 2)]),
    ('F3 sublinear_tf', 'sublinear_tf', [False, True]),
    ('F4 min_df', 'min_df', [1, 2, 5]),
    ('F5 max_df', 'max_df', [1.0, 0.5]),
    ('F6 max_features', 'max_features', [None, 50000, 20000, 5000]),
]


def select(rows, n_val, ref):
    best_val = max(r['val_acc'] for r in rows)
    se = val_se(best_val, n_val)
    close = [r for r in rows if r['val_acc'] >= best_val - se]
    equiv = [r for r in close if r['cv_mean'] >= ref['cv_mean'] - ref['cv_std']]
    if not equiv:
        return max(rows, key=lambda r: r['cv_mean']), se
    return min(equiv, key=lambda r: (r['n_features'], r['order'])), se


def main():
    setup_log('stage2')
    splits = split_data()
    cache = TextCache(splits)
    prep = Preprocessor(**PREPROCESS)
    n_val = len(splits['val'][1])

    kind, params = 'tfidf', {}
    all_rows = []
    ref = None
    for step, key, values in STEPS:
        if key == 'sublinear_tf' and kind not in ('tf', 'tfidf'):
            print(f'跳过 {step}: 表示方法为 {kind}，不适用')
            continue
        rows = []
        for order, v in enumerate(values):
            k, p = (v, dict(params)) if key == 'kind' else (kind, {**params, key: v})
            row, _, _ = evaluate(cache, step, f'{key}={v}', prep, k, p, *SVM, cv=True)
            nb_row, _, _ = evaluate(cache, step + '-nb', f'{key}={v}', prep, k, p, *NB)
            row.update(step=step, key=key, value=str(v), order=order, nb_val_acc=nb_row['val_acc'])
            rows.append(row)
        step_top = max(rows, key=lambda r: r['cv_mean'])
        if ref is None or step_top['cv_mean'] > ref['cv_mean']:
            ref = step_top
        chosen, se = select(rows, n_val, ref)
        v = values[chosen['order']]
        if key == 'kind':
            kind = v
        else:
            params[key] = v
        print(f'  → {step} 选择 {key}={v}  (val SE≈{se:.4f}; 参照 {ref["step"]} {ref["name"]} '
              f'cv={ref["cv_mean"]:.4f}±{ref["cv_std"]:.4f})  当前配置: kind={kind}, {params}\n')
        all_rows.extend(rows)

    df = save_rows(all_rows, 'stage2_features.csv')
    print('最终特征配置: kind =', kind, ' params =', params)
    plot(df)


def plot(df):
    steps = list(dict.fromkeys(df['step']))
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharey=True)
    axes = axes.ravel()
    for ax, step in zip(axes, steps):
        d = df[df['step'] == step]
        x = range(len(d))
        ax.bar(x, d['val_acc'], width=0.5, color='C0', label='SVM val')
        ax.errorbar(x, d['cv_mean'], yerr=d['cv_std'], fmt='o', color='C1', label='SVM dev CV')
        ax.plot(list(x), d['nb_val_acc'], 's--', color='C2', label='NB val')
        ax.set_xticks(list(x))
        ax.set_xticklabels(d['value'], rotation=30, fontsize=8)
        ax.set_title(step, fontsize=9)
    axes[0].set_ylim(0.85, 0.97)
    axes[0].set_ylabel('accuracy')
    axes[-1].legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(FIGURES / 'stage2_features.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
