# coding=utf-8

import pandas as pd
import jieba
from jieba import posseg
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from const import (
    QA_TRAIN_DATA_PATH,
    QA_TEST_DATA_PATH,
    QA_STOPWORDS_PATH,
    QA_TRAIN_CLEAN_X_PATH,
    QA_TRAIN_CLEAN_Y_PATH,
    QA_TEST_CLEAN_X_PATH,
)


def read_stopwords(path):
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines


def segment(sentence, cut_type='word', pos=False):
    if pos:
        if cut_type == 'word':
            word_pos_seq = posseg.lcut(sentence)
            word_seq, pos_seq = [], []
            for w, p in word_pos_seq:
                word_seq.append(w)
                pos_seq.append(p)
            return word_seq, pos_seq
        elif cut_type == 'char':
            word_seq = list(sentence)
            pos_seq = []
            for w in word_seq:
                w_p = posseg.lcut(w)
                pos_seq.append(w_p[0].flag)
            return word_seq, pos_seq
    else:
        if cut_type == 'word':
            return jieba.lcut(sentence)
        elif cut_type == 'char':
            return list(sentence)


def parse_data(train_path, test_path):
    # 读取csv
    train_df = pd.read_csv(train_path, encoding='utf-8')
    # 去除report为空的
    train_df.dropna(subset=['Report'], how='any', inplace=True)
    # 剩余字段是输入，包含Brand,Model,Question,Dialogue，如果有空，填充即可
    train_df.fillna('', inplace=True)
    # 实际的输入X仅仅选择两个字段，将其拼接起来
    train_x = train_df.Question.str.cat(train_df.Dialogue)
    train_y = []
    if 'Report' in train_df.columns:
        train_y = train_df.Report
        assert len(train_x) == len(train_y)

    test_df = pd.read_csv(test_path, encoding='utf-8')
    test_df.fillna('', inplace=True)
    test_x = test_df.Question.str.cat(test_df.Dialogue)
    return train_x, train_y, test_x, []


def save_data(data, path, stopwords=set(), empty_word=''):
    count = 0
    with open(path, 'w', encoding='utf-8') as f1:
        for line in data:
            if isinstance(line, str):
                seg_list = segment(line.strip(), cut_type='word')
                # 考虑stopwords
                seg_list = list(filter(lambda x: x not in stopwords, seg_list))
                if len(seg_list) > 0:
                    seg_line = ' '.join(seg_list)
                    f1.write('%s' % seg_line)
                    f1.write('\n')
                elif empty_word:
                    f1.write(u'随时 联系\n')
                count += 1
    print('train path length is ', path, count)


def preprocess_sentence(sentence):
    seg_list = segment(sentence.strip(), cut_type='word')
    seg_line = ' '.join(seg_list)
    return seg_line


if __name__ == '__main__':
    train_list_src, train_list_trg, test_list_src, _ = parse_data(
        QA_TRAIN_DATA_PATH, QA_TEST_DATA_PATH)
    print(len(train_list_src))
    print(len(train_list_trg))
    stopwords = read_stopwords(QA_STOPWORDS_PATH)
    save_data(train_list_src, QA_TRAIN_CLEAN_X_PATH, stopwords)
    save_data(train_list_trg, QA_TRAIN_CLEAN_Y_PATH, stopwords)
    save_data(test_list_src, QA_TEST_CLEAN_X_PATH, stopwords)
