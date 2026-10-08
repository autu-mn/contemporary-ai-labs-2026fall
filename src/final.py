"""阶段 5：锁定方案 → 本地测试集评估一次 → 全量重训 → 生成 predictions.csv。

1. 三个基线模型与最终方案都用 dev(train+val) 重训，在本地测试集上评估一次。
   本地测试集结果只用于报告最终性能，不再据此修改任何配置。
2. 最终方案用全部 7368 条训练数据重训，对官方无标签测试集预测，
   输出无表头、单列整数标签的 predictions.csv，只做格式检查。
"""
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from config import BASELINES, FINAL, PREPROCESS, VEC_KIND, VEC_PARAMS
from data import FIGURES, RESULTS, ROOT, load_official_test, load_train, split_data
from experiment import build_pipeline, setup_log
from models import n_params
from preprocess import Preprocessor

LABELS = ['graphics', 'windows', 'pc.hw', 'autos', 'motorcyc', 'baseball', 'crypt', 'electron', 'med', 'mideast']


def configs():
    for name, c in BASELINES.items():
        yield name, PREPROCESS, VEC_KIND, VEC_PARAMS, c['model'], c['params']
    yield FINAL['name'], FINAL['preprocess'], FINAL['vec_kind'], FINAL['vec_params'], FINAL['model'], FINAL['params']


def evaluate_local_test(splits):
    out = RESULTS / 'stage5_local_test.csv'
    if out.exists():
        print('注意：本地测试集此前已评估过；本次为复现运行，配置未改变时结果应完全一致。')
    X_dev, y_dev = splits['dev']
    X_te, y_te = splits['test_local']
    rows, final_pred = [], None
    for name, prep_opts, vec_kind, vec_params, model, params in configs():
        prep = Preprocessor(**prep_opts)
        pipe = build_pipeline(vec_kind, vec_params, model, params)
        t0 = time.time()
        pipe.fit(X_dev.map(prep), y_dev)
        fit_sec = time.time() - t0
        p = pipe.predict(X_te.map(prep))
        rows.append(dict(model=name, test_acc=accuracy_score(y_te, p), test_f1=f1_score(y_te, p, average='macro'),
                         n_features=len(pipe['vec'].get_feature_names_out()), n_params=n_params(pipe['clf']),
                         fit_sec_on_dev=round(fit_sec, 1)))
        print(f'[local-test] {name:<44s} acc={rows[-1]["test_acc"]:.4f} f1={rows[-1]["test_f1"]:.4f} '
              f'params={rows[-1]["n_params"]:,} fit={fit_sec:.1f}s', flush=True)
        if name == FINAL['name']:
            final_pred = p
    pd.DataFrame(rows).to_csv(out, index=False, encoding='utf-8-sig')

    print('\n最终方案在本地测试集上的逐类结果:')
    print(classification_report(y_te, final_pred, target_names=LABELS, digits=4))
    rep = classification_report(y_te, final_pred, target_names=LABELS, output_dict=True)
    pd.DataFrame(rep).T.to_csv(RESULTS / 'stage5_final_per_class.csv', encoding='utf-8-sig')
    cm = confusion_matrix(y_te, final_pred)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(RESULTS / 'stage5_final_confusion.csv', encoding='utf-8-sig')
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
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
    ax.set_title(f'Final model on local test (acc={accuracy_score(y_te, final_pred):.4f})')
    fig.tight_layout()
    fig.savefig(FIGURES / 'stage5_local_test_confusion.png', dpi=150)
    plt.close(fig)


def predict_official():
    X, y = load_train()
    X_test = load_official_test()
    prep = Preprocessor(**FINAL['preprocess'])
    pipe = build_pipeline(FINAL['vec_kind'], FINAL['vec_params'], FINAL['model'], FINAL['params'])
    pipe.fit(X.map(prep), y)
    pred = pipe.predict(X_test.map(prep))

    assert len(pred) == len(X_test) == 2457, '预测条数与测试集不一致'
    assert np.issubdtype(pred.dtype, np.integer) and set(np.unique(pred)) <= set(range(10)), '标签不在 0-9'
    path = ROOT / 'predictions.csv'
    pd.DataFrame(pred).to_csv(path, index=False, header=False)

    back = pd.read_csv(path, header=None)
    assert back.shape == (2457, 1) and back[0].between(0, 9).all(), '写出的文件格式不正确'
    print(f'\n已写出 {path.name}: {back.shape[0]} 行 × {back.shape[1]} 列，无表头，标签范围 '
          f'{back[0].min()}-{back[0].max()}')


def check_locked_config():
    """config.py 中锁定的配置必须与阶段 3、4 的选择结果一致，防止手工抄写出错。"""
    sel = json.loads((RESULTS / 'stage3_selected.json').read_text(encoding='utf-8'))
    for name, c in BASELINES.items():
        expect = dict(sel[c['model']])
        got = {k: list(v) if isinstance(v, tuple) else v for k, v in c['params'].items()}
        assert got == expect, f'{name} 的锁定参数 {got} 与阶段 3 结果 {expect} 不一致'
    choice = json.loads((RESULTS / 'stage4_final_choice.json').read_text(encoding='utf-8'))
    assert choice['candidate'] == FINAL['candidate'], f'最终方案与阶段 4 选择 {choice["name"]} 不一致'
    assert choice.get('C', FINAL['params']['C']) == FINAL['params']['C'], '最终方案的 C 与阶段 4 复核结果不一致'
    print('锁定配置与阶段 3/4 的选择结果一致。')


def main():
    setup_log('stage5_final')
    check_locked_config()
    splits = split_data()
    print('=== 1. 本地测试集评估（仅一次）===')
    evaluate_local_test(splits)
    print('=== 2. 全量重训并预测官方测试集 ===')
    predict_official()


if __name__ == '__main__':
    main()
