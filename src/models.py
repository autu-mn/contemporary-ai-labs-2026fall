"""三个代表性模型：MultinomialNB（生成式）、线性 SVM（判别式最大间隔）、MLP（非线性神经网络）。"""
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC

from data import SEED


def make_model(name, **params):
    if name == 'nb':
        return MultinomialNB(**params)
    if name == 'linsvc':
        return LinearSVC(random_state=SEED, **params)
    if name == 'svc':
        return SVC(random_state=SEED, **params)
    if name == 'mlp':
        return MLPClassifier(random_state=SEED, **params)
    if name == 'vote':
        # params: {'linsvc': {...}, 'nb': {...}, 'mlp': {...}}，硬投票（多数表决）
        return VotingClassifier([(m, make_model(m, **p)) for m, p in params.items()], voting='hard')
    raise ValueError(name)


def n_params(clf):
    """可训练参数量，用于比较模型复杂度。"""
    if isinstance(clf, MultinomialNB):
        return int(clf.feature_log_prob_.size + clf.class_log_prior_.size)
    if isinstance(clf, LinearSVC):
        return int(clf.coef_.size + clf.intercept_.size)
    if isinstance(clf, SVC):
        # 核方法的模型规模由支持向量决定：支持向量个数 × 特征维度 + 对偶系数
        return int(clf.support_vectors_.shape[0] * clf.support_vectors_.shape[1] + clf.dual_coef_.size)
    if isinstance(clf, MLPClassifier):
        return int(sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_))
    if isinstance(clf, VotingClassifier):
        return int(sum(n_params(e) for e in clf.estimators_))
    return -1
