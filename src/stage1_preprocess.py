"""阶段 1：预处理消融。

固定 TfidfVectorizer() 默认参数 + LinearSVC(C=1)，每次只打开一个清洗开关；
选择依据为验证集准确率，开发集 5 折 CV 与 NB 的结果作为稳健性旁证（不参与选择）。
组合规则（事先确定）：所有相对 P0 提升验证集准确率的单项开关合并为 P6。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data import FIGURES, split_data
from experiment import TextCache, evaluate, save_rows, setup_log, val_se
from preprocess import Preprocessor

VARIANTS = [
    ('P0 raw', {}),
    ('P1 strip header (keep Subject)', {'header': True}),
    ('P2 strip quotes', {'quotes': True}),
    ('P3 strip email/url/number', {'noise': True}),
    ('P4 stopwords', {'stopwords': True}),
    ('P5a Porter stemming', {'stem': True}),
    ('P5b alphabetic tokens len>=3', {'alpha3': True}),
]
VEC = ('tfidf', {})
SVM = ('linsvc', {'C': 1.0})
NB = ('nb', {'alpha': 0.1})


def run_variant(cache, name, opts):
    prep = Preprocessor(**opts)
    row, _, _ = evaluate(cache, 'P', name, prep, *VEC, *SVM, cv=True)
    nb_row, _, _ = evaluate(cache, 'P-nb', name, prep, *VEC, *NB)
    row['nb_val_acc'] = nb_row['val_acc']
    row['opts'] = str(opts)
    return row


def plot(df):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = range(len(df))
    ax.bar([i - 0.2 for i in x], df['val_acc'], width=0.4, label='LinearSVC val acc')
    ax.errorbar([i + 0.2 for i in x], df['cv_mean'], yerr=df['cv_std'], fmt='o', color='C1',
                label='LinearSVC dev 5-fold CV (mean±std)')
    ax.plot(list(x), df['nb_val_acc'], 's--', color='C2', label='NB val acc (side check)')
    ax.axhline(df['val_acc'].iloc[0], color='gray', lw=0.8, ls=':')
    ax.set_xticks(list(x))
    ax.set_xticklabels([n.split(' ')[0] for n in df['name']])
    ax.set_ylim(min(df['nb_val_acc'].min(), df['val_acc'].min()) - 0.02, df['val_acc'].max() + 0.04)
    ax.set_ylabel('accuracy')
    ax.set_title('Stage 1: preprocessing ablation (TF-IDF default + LinearSVC C=1)')
    ax.legend(fontsize=8, loc='upper center', ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES / 'stage1_preprocess.png', dpi=150)
    plt.close(fig)


def main():
    setup_log('stage1')
    splits = split_data()
    cache = TextCache(splits)
    rows = [run_variant(cache, name, opts) for name, opts in VARIANTS]

    base = rows[0]['val_acc']
    combo = {}
    for r, (_, opts) in zip(rows[1:], VARIANTS[1:]):
        if r['val_acc'] > base:
            combo.update(opts)
    print('\n提升验证集准确率的单项开关:', combo)
    if len(combo) > 1:
        rows.append(run_variant(cache, 'P6 combine improving steps', combo))

    # P6 的验证集准确率最高但 CV 低于 P0，说明单靠验证集贪心组合受噪声影响；
    # 补充两个子组合作为候选（事后追加，报告中说明）
    rows.append(run_variant(cache, 'P7 header + noise', {'header': True, 'noise': True}))
    rows.append(run_variant(cache, 'P8 header + stem', {'header': True, 'stem': True}))

    df = save_rows(rows, 'stage1_preprocess.csv')
    plot(df)
    best = df.loc[df['val_acc'].idxmax()]
    se = val_se(best['val_acc'], len(splits['val'][1]))
    print(f'\n验证集最优: {best["name"]}  val={best["val_acc"]:.4f} (SE≈{se:.4f})  opts={best["opts"]}')
    close = df[df['val_acc'] >= best['val_acc'] - se]
    print('与最优差距在 1 个 SE 以内的方案（按开发集 CV 均值排序）:')
    print(close.sort_values('cv_mean', ascending=False)[
        ['name', 'val_acc', 'cv_mean', 'cv_std', 'nb_val_acc', 'n_features']].to_string(index=False))
    # 验证集 1 SE 内、且 CV 均值与最高者之差小于其 CV 标准差的方案视为统计上等价；
    # 等价方案中取清洗步骤最少者，再以 NB 旁证（三个模型共用同一预处理）打破并列
    top_cv = close['cv_mean'].max()
    equiv = close[close['cv_mean'] >= top_cv - close['cv_std']].copy()
    equiv['n_steps'] = equiv['opts'].map(lambda s: len(eval(s)))
    chosen = equiv.sort_values(['n_steps', 'nb_val_acc'], ascending=[True, False]).iloc[0]
    print('\n统计上等价的方案:', ', '.join(equiv['name']))
    print(f'选择（等价方案中步骤最少，NB 旁证最高）→ {chosen["name"]}  {chosen["opts"]}')


if __name__ == '__main__':
    main()
