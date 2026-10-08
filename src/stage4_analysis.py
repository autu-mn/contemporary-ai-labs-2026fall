"""阶段 4：稳健性、误差分析、可解释性与创新实验（全部只用 train / val / dev）。

1. 三个模型的选定配置：单进程重测训练/预测耗时与参数量（阶段 3 的 MLP 为并行训练，耗时不可比）
2. 开发集 5 折分层 CV（三个模型使用相同的折），报告均值±标准差与逐折配对差值
3. 验证集误差分析：混淆矩阵、逐类 F1、最常混淆的类别对与错分样例
4. 线性 SVM 每类权重最高的词：对比保留邮件头 (P0) 与删除邮件头 (P1)，检查捷径特征
5. 创新实验：Subject 加权、词级+字符级 n-gram、三模型多数投票（均含验证集与开发集 CV）
6. 最终模型选择：CV 最高者 1 个标准差内且验证集最优者 1 SE 内的候选视为等价，取训练耗时最短者
"""
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import PREPROCESS, VEC_KIND, VEC_PARAMS
from data import FIGURES, RESULTS, SEED, split_data
from experiment import TextCache, build_pipeline, evaluate, save_rows, setup_log, val_se
from models import n_params
from preprocess import Preprocessor

# 由阶段 0 各类高频词推断的类别含义，仅用于图表标注
LABELS = ['graphics', 'windows', 'pc.hw', 'autos', 'motorcyc', 'baseball', 'crypt', 'electron', 'med', 'mideast']
MODEL_NAMES = {'nb': 'MultinomialNB', 'linsvc': 'Linear SVM', 'mlp': 'MLP'}
FINAL_CS = [0.1, 0.3, 1, 3]


def load_selected():
    sel = json.loads((RESULTS / 'stage3_selected.json').read_text(encoding='utf-8'))
    sel['mlp']['hidden_layer_sizes'] = tuple(sel['mlp']['hidden_layer_sizes'])
    return sel


def complexity_and_preds(cache, prep, sel):
    X_tr, y_tr = cache.get(prep, 'train'), cache.y('train')
    X_va, y_va = cache.get(prep, 'val'), cache.y('val')
    rows, preds, pipes = [], {}, {}
    for m, params in sel.items():
        pipe = build_pipeline(VEC_KIND, VEC_PARAMS, m, params)
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        fit_sec = time.time() - t0
        t0 = time.time()
        p = pipe.predict(X_va)
        pred_sec = time.time() - t0
        preds[m], pipes[m] = p, pipe
        rows.append(dict(model=MODEL_NAMES[m], params=str(params),
                         train_acc=accuracy_score(y_tr, pipe.predict(X_tr)),
                         val_acc=accuracy_score(y_va, p), val_f1=f1_score(y_va, p, average='macro'),
                         n_features=len(pipe['vec'].vocabulary_), n_params=n_params(pipe['clf']),
                         fit_sec=round(fit_sec, 2), pred_sec=round(pred_sec, 3)))
        print(f'[final-val] {MODEL_NAMES[m]:<14s} val={rows[-1]["val_acc"]:.4f} f1={rows[-1]["val_f1"]:.4f} '
              f'params={rows[-1]["n_params"]:,} fit={fit_sec:.1f}s pred={pred_sec:.2f}s', flush=True)
    return pd.DataFrame(rows), preds, pipes


def cross_validate_all(cache, prep, sel):
    X_dev, y_dev = cache.get(prep, 'dev'), cache.y('dev')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {}
    for m, params in sel.items():
        scores[m] = cross_val_score(build_pipeline(VEC_KIND, VEC_PARAMS, m, params), X_dev, y_dev,
                                    cv=skf, scoring='accuracy', n_jobs=5)
        print(f'[CV] {MODEL_NAMES[m]:<14s} folds={np.round(scores[m], 4)} mean={scores[m].mean():.4f}±{scores[m].std():.4f}', flush=True)
    df = pd.DataFrame({MODEL_NAMES[m]: s for m, s in scores.items()})
    df.index = [f'fold{i + 1}' for i in range(5)]
    for a, b in [('linsvc', 'nb'), ('linsvc', 'mlp'), ('mlp', 'nb')]:
        d = scores[a] - scores[b]
        print(f'  配对差 {MODEL_NAMES[a]} - {MODEL_NAMES[b]}: mean={d.mean():+.4f} std={d.std():.4f} '
              f'正差折数={int((d > 0).sum())}/5')
    return df


def error_analysis(cache, prep, preds):
    y_va = cache.y('val').values
    X_va_raw = cache.splits['val'][0].values
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    per_class = {}
    for ax, (m, p) in zip(axes, preds.items()):
        cm = confusion_matrix(y_va, p)
        ax.imshow(cm, cmap='Blues')
        for i in range(10):
            for j in range(10):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7,
                            color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xticks(range(10))
        ax.set_xticklabels(LABELS, rotation=60, fontsize=7)
        ax.set_yticks(range(10))
        ax.set_yticklabels(LABELS, fontsize=7)
        ax.set_xlabel('predicted')
        ax.set_ylabel('true')
        ax.set_title(f'{MODEL_NAMES[m]} (val acc={accuracy_score(y_va, p):.4f})')
        rep = classification_report(y_va, p, output_dict=True, zero_division=0)
        per_class[MODEL_NAMES[m]] = [rep[str(c)]['f1-score'] for c in range(10)]
    fig.tight_layout()
    fig.savefig(FIGURES / 'stage4_confusion.png', dpi=150)
    plt.close(fig)

    pc = pd.DataFrame(per_class, index=[f'{c} {LABELS[c]}' for c in range(10)]).round(4)
    print('\n逐类 F1（验证集）:\n' + pc.to_string())
    pc.to_csv(RESULTS / 'stage4_per_class_f1.csv', encoding='utf-8-sig')

    pairs = []
    for m, p in preds.items():
        cm = confusion_matrix(y_va, p)
        np.fill_diagonal(cm, 0)
        for i, j in zip(*np.unravel_index(np.argsort(cm, axis=None)[::-1][:5], cm.shape)):
            pairs.append(dict(model=MODEL_NAMES[m], true=f'{i} {LABELS[i]}', pred=f'{j} {LABELS[j]}', count=int(cm[i, j])))
    pairs = pd.DataFrame(pairs)
    print('\n各模型最常见的 5 个错分方向:\n' + pairs.to_string(index=False))
    pairs.to_csv(RESULTS / 'stage4_confused_pairs.csv', index=False, encoding='utf-8-sig')

    p = preds['linsvc']
    wrong = np.where(p != y_va)[0]
    both_wrong = [i for i in wrong if preds['nb'][i] != y_va[i] and preds['mlp'][i] != y_va[i]]
    print(f'\n线性 SVM 错分 {len(wrong)} 条，其中三个模型全错 {len(both_wrong)} 条')
    samples = []
    for i in both_wrong[:6]:
        body = Preprocessor(header=True)(X_va_raw[i])
        snippet = ' '.join(body.split())[:300]
        samples.append(dict(true=LABELS[y_va[i]], svm=LABELS[p[i]], nb=LABELS[preds['nb'][i]],
                            mlp=LABELS[preds['mlp'][i]], n_words=len(body.split()), text=snippet))
        print(f'--- true={LABELS[y_va[i]]} svm={LABELS[p[i]]} nb={LABELS[preds["nb"][i]]} '
              f'mlp={LABELS[preds["mlp"][i]]} words={len(body.split())}\n{snippet}')
    pd.DataFrame(samples).to_csv(RESULTS / 'stage4_error_samples.csv', index=False, encoding='utf-8-sig')
    return pc


def top_words(cache, sel, k=10):
    rows = []
    for tag, prep in [('P0 keep header', Preprocessor()), ('P1 strip header', Preprocessor(**PREPROCESS))]:
        pipe = build_pipeline(VEC_KIND, VEC_PARAMS, 'linsvc', sel['linsvc'])
        pipe.fit(cache.get(prep, 'train'), cache.y('train'))
        vocab = pipe['vec'].get_feature_names_out()
        coef = pipe['clf'].coef_
        print(f'\n线性 SVM 每类权重最高的 {k} 个词 [{tag}]:')
        for c in range(10):
            words = ' '.join(vocab[np.argsort(coef[c])[::-1][:k]])
            rows.append(dict(setting=tag, label=f'{c} {LABELS[c]}', top_words=words))
            print(f'  {c} {LABELS[c]:<9s} {words}')
    pd.DataFrame(rows).to_csv(RESULTS / 'stage4_top_words.csv', index=False, encoding='utf-8-sig')


def innovations(cache, sel, preds, comp, cv):
    """所有候选都有验证集与开发集 5 折 CV 结果，作为最终模型选择的依据。"""
    rows = []
    for m in ('nb', 'linsvc', 'mlp'):
        r = comp[comp['model'] == MODEL_NAMES[m]].iloc[0].to_dict()
        rows.append(dict(r, group='I0', name=f'{MODEL_NAMES[m]} (selected config)', candidate=m,
                         cv_mean=cv[MODEL_NAMES[m]].mean(), cv_std=cv[MODEL_NAMES[m]].std(ddof=0)))

    def add(group, name, candidate, prep, vec_kind, model, params):
        row, _, _ = evaluate(cache, group, name, prep, vec_kind, VEC_PARAMS, model, params, cv=True)
        rows.append(dict(row, candidate=candidate))

    p3 = Preprocessor(**PREPROCESS, subject_repeat=3)
    add('I1', 'Linear SVM + Subject x2', 'svm_s2', Preprocessor(**PREPROCESS, subject_repeat=2), VEC_KIND, 'linsvc', sel['linsvc'])
    add('I1', 'Linear SVM + Subject x3', 'svm_s3', p3, VEC_KIND, 'linsvc', sel['linsvc'])
    add('I1', 'MultinomialNB + Subject x3', 'nb_s3', p3, VEC_KIND, 'nb', sel['nb'])
    add('I2', 'Linear SVM + word+char n-gram', 'svm_char', Preprocessor(**PREPROCESS), 'word+char', 'linsvc', sel['linsvc'])
    add('I2', 'Linear SVM + Subject x3 + word+char', 'svm_s3_char', p3, 'word+char', 'linsvc', sel['linsvc'])
    add('I3', 'Hard vote (SVM, NB, MLP)', 'vote', Preprocessor(**PREPROCESS), VEC_KIND, 'vote',
        {'linsvc': sel['linsvc'], 'nb': sel['nb'], 'mlp': sel['mlp']})

    y_va = cache.y('val').values
    stacked = np.vstack([preds['linsvc'], preds['nb'], preds['mlp']])
    print(f'     三个模型中至少一个正确的比例（投票准确率上限）: {np.mean((stacked == y_va).any(axis=0)):.4f}')
    return save_rows(rows, 'stage4_innovations.csv')


def select_final(df, n_val):
    """以 CV 最高者为参照：CV ≥ 参照均值 - 参照标准差 且 验证集在最优者 1 SE 内 → 等价；等价者中取训练耗时最短。"""
    ref = df.loc[df['cv_mean'].idxmax()]
    se = val_se(df['val_acc'].max(), n_val)
    equiv = df[(df['cv_mean'] >= ref['cv_mean'] - ref['cv_std']) & (df['val_acc'] >= df['val_acc'].max() - se)]
    chosen = equiv.loc[equiv['fit_sec'].idxmin()]
    print(f'\nCV 参照: {ref["name"]} cv={ref["cv_mean"]:.4f}±{ref["cv_std"]:.4f}；验证集最优 {df["val_acc"].max():.4f} (SE≈{se:.4f})')
    print('等价候选:\n' + equiv[['name', 'val_acc', 'cv_mean', 'cv_std', 'n_features', 'fit_sec']].to_string(index=False))
    print(f'→ 最终模型: {chosen["name"]}  (candidate={chosen["candidate"]})')
    return chosen


def final_c_check(cache, n_val):
    """最终配置换成了词级+字符级特征，阶段 3 在纯词级特征上选出的 C 不一定仍合适，按 1-SE 规则复核。"""
    from stage3_models import select
    p3 = Preprocessor(**PREPROCESS, subject_repeat=3)
    rows = []
    for c in FINAL_CS:
        row, _, _ = evaluate(cache, 'I4', f'final features, LinearSVC C={c}', p3, 'word+char', VEC_PARAMS,
                             'linsvc', {'C': c, 'max_iter': 5000}, cv=True)
        rows.append({**row, 'C': c})
    df = pd.DataFrame(rows)
    best, se = select(df, n_val, ('C', True))
    print(f'→ 最终特征下 C 选择 {best["C"]} (val={best["val_acc"]:.4f}, cv={best["cv_mean"]:.4f}, SE≈{se:.4f})')
    save_rows(rows, 'stage4_final_c.csv')
    return float(best['C'])


def main():
    setup_log('stage4')
    splits = split_data()
    cache = TextCache(splits)
    prep = Preprocessor(**PREPROCESS)
    sel = load_selected()

    print('=== 1. 选定配置：验证集表现与复杂度（单进程计时）===')
    comp, preds, _ = complexity_and_preds(cache, prep, sel)
    comp.to_csv(RESULTS / 'stage4_complexity.csv', index=False, encoding='utf-8-sig')

    print('\n=== 2. 开发集 5 折 CV（相同折）===')
    cv = cross_validate_all(cache, prep, sel)
    cv.to_csv(RESULTS / 'stage4_cv.csv', encoding='utf-8-sig')

    print('\n=== 3. 验证集误差分析 ===')
    error_analysis(cache, prep, preds)

    print('\n=== 4. 权重词与捷径特征 ===')
    top_words(cache, sel)

    print('\n=== 5. 创新实验（验证集 + 开发集 CV）===')
    df = innovations(cache, sel, preds, comp, cv)

    print('\n=== 6. 最终模型选择（验证集性能 + 复杂度）===')
    chosen = select_final(df, len(splits['val'][1]))
    choice = {'candidate': chosen['candidate'], 'name': chosen['name']}
    if chosen['candidate'] == 'svm_s3_char':
        print('\n=== 7. 最终配置的 C 复核 ===')
        choice['C'] = final_c_check(cache, len(splits['val'][1]))
    (RESULTS / 'stage4_final_choice.json').write_text(
        json.dumps(choice, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
