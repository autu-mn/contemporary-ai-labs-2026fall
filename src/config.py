"""各阶段在验证集上锁定的决定。后续阶段只读取这里的配置。

每项注明来源阶段，对应证据见 results/stage*_*.csv 与 results/logs/。
"""

# 阶段 1：只删除邮件头（保留 Subject）。
# 依据：验证集 1 SE 内且 CV 等价的方案中清洗步骤最少，NB 旁证也最高；
# 删除引用行显著有害（SVM -3.2%，NB -5.3%），已排除。
PREPROCESS = {'header': True}

# 阶段 2：TF-IDF，一元词，min_df=2（34270 维）。
# 依据：与本阶段 CV 最优的 (1,2) 二元词配置（649569 维）在 1 个 CV 标准差内等价，维度约为其 1/19；
# 去掉 IDF（tf）或使用原始词频（count/binary）均显著变差。
VEC_KIND = 'tfidf'
VEC_PARAMS = {'ngram_range': (1, 1), 'sublinear_tf': False, 'min_df': 2, 'max_df': 1.0, 'max_features': None}

# 阶段 3：三个模型按 1-SE 规则选出的超参数（与 results/stage3_selected.json 一致）
BASELINES = {
    'MultinomialNB': {'model': 'nb', 'params': {'alpha': 0.1}},
    'Linear SVM': {'model': 'linsvc', 'params': {'C': 0.3, 'max_iter': 5000}},
    'MLP': {'model': 'mlp', 'params': {'hidden_layer_sizes': (100,), 'alpha': 0.0001, 'early_stopping': True,
                                       'max_iter': 50, 'n_iter_no_change': 5}},
}

# 阶段 4：最终方案（锁定后才允许在本地测试集上评估一次）。
# 线性 SVM + 删除邮件头 + Subject 重复 3 次 + 词级 TF-IDF 与字符 2-5 gram TF-IDF 拼接，C=0.3。
# 依据：所有候选中开发集 CV 最高 (0.9495±0.0034)，且高出其余候选 1 个标准差以上；验证集 0.9548。
FINAL = {
    'candidate': 'svm_s3_char',
    'name': 'Linear SVM + Subject x3 + word+char n-gram',
    'preprocess': {'header': True, 'subject_repeat': 3},
    'vec_kind': 'word+char',
    'vec_params': VEC_PARAMS,
    'model': 'linsvc',
    'params': {'C': 0.3, 'max_iter': 5000},
}
