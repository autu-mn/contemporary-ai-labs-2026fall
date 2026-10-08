"""阶段 0：检查划分、训练集数据探查、E0 基线。

数据探查只使用 train 子集；验证集/本地测试集只统计条数和类别分布，用于确认分层划分正确。
"""
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from data import FIGURES, RESULTS, split_data
from experiment import TextCache, evaluate, save_rows, setup_log


def check_split(splits):
    X_tr, y_tr = splits['train']
    X_va, y_va = splits['val']
    X_te, y_te = splits['test_local']
    idx = [set(X_tr.index), set(X_va.index), set(X_te.index)]
    assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2]), '子集之间有重叠'
    assert len(idx[0]) + len(idx[1]) + len(idx[2]) == 7368
    dist = pd.DataFrame({
        'train': y_tr.value_counts().sort_index(),
        'val': y_va.value_counts().sort_index(),
        'test_local': y_te.value_counts().sort_index(),
    })
    print('子集大小:', len(X_tr), len(X_va), len(X_te))
    print('各子集类别比例（应基本一致）:')
    print((dist / dist.sum()).round(3).to_string())
    dist.to_csv(RESULTS / 'stage0_split_distribution.csv', encoding='utf-8-sig')
    return dist


def eda_train(X_tr, y_tr):
    n_words = X_tr.str.split().str.len()
    print('\n训练集词数统计:\n', n_words.describe().round(1).to_string())

    has_header = X_tr.str.match(r'^From:') & X_tr.str.contains('\n\n')
    has_subject = X_tr.str.contains(r'^Subject:', flags=re.M)
    has_quote = X_tr.str.contains(r'^\s*>', flags=re.M)
    has_org = X_tr.str.contains(r'^Organization:', flags=re.M)
    print(f'\n以 From: 开头且有空行分隔头部: {has_header.mean():.3f}')
    print(f'含 Subject 行: {has_subject.mean():.3f}  含 Organization 行: {has_org.mean():.3f}  含 > 引用行: {has_quote.mean():.3f}')

    fields = X_tr.str.partition('\n\n')[0].str.findall(r'^([A-Za-z-]+):', flags=re.M).explode()
    print('\n头部字段出现频率 Top10:\n', (fields.value_counts().head(10) / len(X_tr)).round(3).to_string())

    vec = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b[a-zA-Z]{3,}\b', min_df=3)
    X = vec.fit_transform(X_tr)
    vocab = np.array(vec.get_feature_names_out())
    top = {}
    print('\n各类平均 TF-IDF 最高的词（仅训练集）:')
    for c in sorted(y_tr.unique()):
        m = np.asarray(X[(y_tr == c).values].mean(axis=0)).ravel()
        top[c] = ' '.join(vocab[m.argsort()[::-1][:15]])
        print(c, top[c])
    pd.Series(top, name='top_words').to_csv(RESULTS / 'stage0_top_words.csv', encoding='utf-8-sig')
    return n_words


def plot_eda(dist, n_words):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    (dist / dist.sum()).plot.bar(ax=axes[0], width=0.8)
    axes[0].set_xlabel('class label')
    axes[0].set_ylabel('proportion')
    axes[0].set_ylim(0, 0.13)
    axes[0].set_title('Class proportion per split (stratified)')
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].legend(ncol=3, loc='upper center', fontsize=8)
    axes[1].hist(np.log10(n_words), bins=50, color='#4C72B0')
    axes[1].set_xlabel('log10(#words per document)')
    axes[1].set_ylabel('count')
    axes[1].set_title(f'Document length (train, median={int(n_words.median())})')
    fig.tight_layout()
    fig.savefig(FIGURES / 'stage0_eda.png', dpi=150)
    plt.close(fig)


def baseline(cache):
    """E0：draft_main.py 中的默认表示（TF-IDF, max_features=5000）+ 三个模型的默认/参考参数。"""
    vec = {'max_features': 5000}
    rows = []
    for name, model, params in [
        ('draft SVC() rbf', 'svc', {}),
        ('NB default', 'nb', {}),
        ('LinearSVC default', 'linsvc', {}),
        ('MLP (100,) max_iter=300', 'mlp', {'hidden_layer_sizes': (100,), 'max_iter': 300}),
    ]:
        row, _, _ = evaluate(cache, 'E0', name, None, 'tfidf', vec, model, params)
        rows.append(row)
    return save_rows(rows, 'stage0_baseline.csv')


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    setup_log('stage0')
    splits = split_data()
    dist = check_split(splits)
    n_words = eda_train(*splits['train'])
    plot_eda(dist, n_words)
    print('\n=== E0 基线（仅 train 拟合，val 评估）===')
    baseline(TextCache(splits))


if __name__ == '__main__':
    main()
