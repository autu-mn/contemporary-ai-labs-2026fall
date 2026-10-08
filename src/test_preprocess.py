"""预处理函数的边界用例检查：python src/test_preprocess.py"""
from preprocess import Preprocessor, split_header, strip_header, strip_noise, strip_quotes


def check(cond, msg):
    assert cond, msg
    print('ok:', msg)


def main():
    # 标准头部：按空行结束
    h, b = split_header('From: a@b.edu\nSubject: Hi\nLines: 3\n\nbody text')
    check(h.endswith('Lines: 3') and b.strip() == 'body text', '标准头部')

    # 头部与正文间没有空行
    h, b = split_header('From: a@b.edu\nSubject: Hi\nI am the body')
    check(b == 'I am the body', '头部后无空行')

    # 头部中夹有纯空格行，后面仍是字段
    h, b = split_header('From: a\nKeywords: \n \nOrganization: X\nLines: 2\n\nbody')
    check('Organization: X' in h and b.strip() == 'body', '头部中的纯空格行')

    # 字段名含点号和方括号；续行
    h, b = split_header('From: a\nArticle-I.D.: x.1\n\t<cont@line>\nNntp-Posting-Host-[nntpd-1]: h\n\nbody')
    check('Nntp-Posting-Host-[nntpd-1]' in h and b.strip() == 'body', '特殊字段名与续行')

    # 正文首行缩进不能被当成续行；正文中 "Re: xxx" 不能被当成头部
    h, b = split_header('From: a\nSubject: s\n  \n     To reader,\n')
    check('To reader' in b, '正文缩进行')
    h, b = split_header('From: a\nLines: 2\n\nRe: Waving\nbody')
    check(b.lstrip().startswith('Re: Waving'), '空行后的 "Re:" 属于正文')

    check(strip_header('From: a\nSubject: Car  \n\nbody', subject_repeat=2).startswith('Car Car'), 'Subject 重复加权')
    q = strip_quotes('In article <x@y> bob@z writes:\n> quoted\nmine\n| also quoted')
    check('quoted' not in q and 'mine' in q, '引用行与归属行删除')
    n = strip_noise('mail me a@b.com or http://x.org/y at 486 3.5 now')
    check('@' not in n and 'http' not in n and '486' not in n and 'now' in n, '邮箱/URL/数字删除')
    p = Preprocessor(stopwords=True, stem=True)('This is running quickly')
    check(p == 'run quickli', '停用词在词干化之前过滤')
    check(Preprocessor(alpha3=True)('ab abc a1b2 Hello') == 'abc hello', '仅保留 >=3 个字母的词')


if __name__ == '__main__':
    main()
