
from collections import defaultdict

import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from const import QA_TRAIN_CLEAN_X_PATH, QA_TRAIN_CLEAN_Y_PATH, QA_TEST_CLEAN_X_PATH, QA_WORD2VEC_VOCAB_PATH


def save_word_dict(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in vocab:
            w, i = line
            f.write("%s\t%d\n" % (w, i))


def read_data(path_1, path_2, path_3):
    with open(path_1, 'r', encoding='utf-8') as f1, \
            open(path_2, 'r', encoding='utf-8') as f2, \
            open(path_3, 'r', encoding='utf-8') as f3:
        words = []
        for line in f1:
            words += line.split(' ')

        for line in f2:
            words += line.split(' ')

        for line in f3:
            words += line.split(' ')
    return list(set(words))


def build_vocab(items, sort=True, min_count=0, lower=False):
    """
    构建词典列表
    :param items: list  [item1, item2, ... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
    """
    result = []
    if sort:
        # sort by count
        dic = defaultdict(int)
        for item in items:
            for i in item.split(" "):
                i = i.strip()
                if not i:
                    continue
                i = i if not lower else item.lower()
                dic[i] += 1
        # sort
        dic = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        for i, item in enumerate(dic):
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            result.append(key)
    else:
        # sort by items
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)

    vocab = [(w, i) for i, w in enumerate(result)]
    reverse_vocab = [(i, w) for i, w in enumerate(result)]

    return vocab, reverse_vocab


if __name__ == '__main__':
    lines = read_data(
        QA_TRAIN_CLEAN_X_PATH,
        QA_TRAIN_CLEAN_Y_PATH,
        QA_TEST_CLEAN_X_PATH)
    vocab, reverse_vocab = build_vocab(lines)
    save_word_dict(vocab, QA_WORD2VEC_VOCAB_PATH)
