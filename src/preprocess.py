"""逐条文本清洗。

这里的每一步都只依赖单条文本本身，不使用任何语料统计量，
因此可以对所有数据子集统一应用而不会造成泄露。
词表、IDF、min_df/max_df 等统计量只在向量化器中、仅基于训练部分拟合。
"""
import re
from functools import lru_cache

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_SUBJECT = re.compile(r'^Subject:(.*)$', re.M)
_FIELD = re.compile(r'^[A-Za-z][A-Za-z0-9._\[\]-]*:')
_QUOTE_LINE = re.compile(r'^\s*[>|].*$', re.M)
_ATTRIBUTION = re.compile(
    r'^.*\b(writes|wrote|says)\s*:\s*$|^\s*In article\s*<[^>]*>.*$', re.M | re.I)
_EMAIL = re.compile(r'\S+@\S+')
_URL = re.compile(r'https?://\S+|www\.\S+')
_NUMBER = re.compile(r'\b\d+(?:\.\d+)?\b')
_TOKEN = re.compile(r'(?u)\b\w\w+\b')
_ALPHA3 = re.compile(r'^[a-z]{3,}$')

_stemmer = PorterStemmer()


@lru_cache(maxsize=None)
def _stem(word):
    return _stemmer.stem(word)


def split_header(text):
    """头部 = 开头的 "Field: ..." 行、紧跟其后以空白开头的续行，以及夹在字段之间的纯空格行；
    遇到完全为空的行或普通正文行即结束。少数文档头部与正文之间没有空行，
    或头部中夹有纯空格行，因此不能简单按第一个空行切分。"""
    lines = text.split('\n')
    end, prev = 0, 'cont'
    for i, line in enumerate(lines):
        if line == '':
            break
        if _FIELD.match(line):
            prev, end = 'field', i + 1
        elif line.strip() and line[:1] in (' ', '\t') and prev in ('field', 'cont'):
            prev, end = 'cont', i + 1
        elif not line.strip():
            prev = 'blank'
        else:
            break
    return '\n'.join(lines[:end]), '\n'.join(lines[end:])


def strip_header(text, subject_repeat=1):
    """删除邮件头，仅保留 Subject（可重复 subject_repeat 次以提升其权重）。"""
    header, body = split_header(text)
    m = _SUBJECT.search(header)
    subject = m.group(1).strip() if m else ''
    return ' '.join([subject] * subject_repeat) + '\n' + body


def strip_quotes(text):
    text = _ATTRIBUTION.sub(' ', text)
    return _QUOTE_LINE.sub(' ', text)


def strip_noise(text):
    text = _EMAIL.sub(' ', text)
    text = _URL.sub(' ', text)
    return _NUMBER.sub(' ', text)


class Preprocessor:
    """可开关的清洗流水线，顺序固定为：邮件头 → 引用 → 噪声 → 分词级处理。"""

    def __init__(self, header=False, quotes=False, noise=False,
                 stopwords=False, stem=False, alpha3=False, subject_repeat=1):
        self.header = header
        self.quotes = quotes
        self.noise = noise
        self.stopwords = stopwords
        self.stem = stem
        self.alpha3 = alpha3
        self.subject_repeat = subject_repeat

    def __call__(self, text):
        if self.header:
            text = strip_header(text, self.subject_repeat)
        if self.quotes:
            text = strip_quotes(text)
        if self.noise:
            text = strip_noise(text)
        if self.stopwords or self.stem or self.alpha3:
            tokens = _TOKEN.findall(text.lower())
            # 停用词必须在词干化之前过滤，否则 "this" 会变成 "thi" 而漏过停用词表
            if self.stopwords:
                tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
            if self.alpha3:
                tokens = [t for t in tokens if _ALPHA3.match(t)]
            if self.stem:
                tokens = [_stem(t) for t in tokens]
            text = ' '.join(tokens)
        return text

    def options(self):
        return {k: v for k, v in vars(self).items() if v and not (k == 'subject_repeat' and v == 1)}

    def __repr__(self):
        opts = self.options()
        return 'Preprocessor(' + ', '.join(f'{k}={v}' for k, v in opts.items()) + ')'
