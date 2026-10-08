"""文本向量化器构造。所有向量化器都放在 Pipeline 中，只在训练部分 fit。"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def make_vectorizer(kind='tfidf', char_max_features=100000, **kw):
    """
    kind:
      count     原始词频词袋
      binary    0/1 词袋
      tf        L2 归一化词频（不含 IDF），用于单独检验 IDF 的作用
      tfidf     TF-IDF
      word+char 词级 TF-IDF 与字符 n-gram TF-IDF 拼接
    """
    if kind == 'count':
        return CountVectorizer(**kw)
    if kind == 'binary':
        return CountVectorizer(binary=True, **kw)
    if kind == 'tf':
        return TfidfVectorizer(use_idf=False, **kw)
    if kind == 'tfidf':
        return TfidfVectorizer(**kw)
    if kind == 'word+char':
        return FeatureUnion([
            ('word', TfidfVectorizer(**kw)),
            ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), sublinear_tf=True,
                                     min_df=2, max_features=char_max_features)),
        ])
    raise ValueError(kind)


def n_features(vectorizer):
    return len(vectorizer.get_feature_names_out())
