"""按顺序复现全部实验：python src/run_all.py

阶段 1~4 的选择结论已固化在 config.py 中；final.py 会核对它们与本次运行输出的选择文件是否一致。
总耗时约 1 小时（主要是 RBF 核 SVM 与 MLP），各阶段日志见 results/logs/。
"""
import runpy
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
STAGES = ['test_preprocess.py', 'stage0_eda_baseline.py', 'stage1_preprocess.py', 'stage2_features.py',
          'stage3_models.py', 'stage4_analysis.py', 'final.py']

if __name__ == '__main__':
    sys.path.insert(0, str(SRC))
    stdout = sys.stdout
    for script in STAGES:
        t0 = time.time()
        print(f'\n########## {script} ##########', file=stdout, flush=True)
        runpy.run_path(str(SRC / script), run_name='__main__')
        sys.stdout = stdout
        print(f'########## {script} 完成，用时 {time.time() - t0:.0f}s ##########', file=stdout, flush=True)
