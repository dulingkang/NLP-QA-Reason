import numpy as np

from optparse import OptionParser

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
# from pt20200419.utils.data_utils import dump_pkl

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from const import (
    QA_TRAIN_CLEAN_X_PATH,
    QA_TRAIN_CLEAN_Y_PATH,
    QA_TEST_CLEAN_X_PATH,
    QA_WORD2VEC_OUT_PATH,
    QA_WORD2VEC_BIN_PATH,
    QA_SENTENCE_PATH,
    QA_WORD2VEC_VOCAB_PATH, QA_WORD2VEC_EMBEDDING_PATH)


def read_lines(path, col_sep=None):
    lines = []
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if col_sep:
                if col_sep in line:
                    lines.append(line)
            else:
                lines.append(line)
    return lines


def extract_sentence(train_x_seg_path, train_y_seg_path, test_seg_path):
    ret = []
    lines = read_lines(train_x_seg_path)
    lines += read_lines(train_y_seg_path)
    lines += read_lines(test_seg_path)
    for line in lines:
        ret.append(line)
    return ret


def save_sentence(lines, sentence_path):
    with open(sentence_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write('%s\n' % line.strip())
    print('save sentence:%s' % sentence_path)


def save_w2v(binary=True, min_count=100):
    sentences = extract_sentence(QA_TRAIN_CLEAN_X_PATH, QA_TRAIN_CLEAN_Y_PATH, QA_TEST_CLEAN_X_PATH)
    save_sentence(sentences, QA_SENTENCE_PATH)
    print('train w2v model...')
    # train model
    w2v = Word2Vec(
        sg=1, sentences=LineSentence(QA_SENTENCE_PATH),
        size=256, window=5, min_count=min_count, iter=5)
    out_path = QA_WORD2VEC_BIN_PATH if binary else QA_WORD2VEC_OUT_PATH
    w2v.wv.save_word2vec_format(out_path, binary=binary)
    print("save w2v model %s ok." % out_path)


def test_similar():
    vector = KeyedVectors.load_word2vec_format(QA_WORD2VEC_BIN_PATH, binary=True)
    sim = vector.similarity('技师', '车主')
    print('技师 vs 车主 similarity score:', sim)

    sim = vector.similarity('福克斯', '车')
    print('福克斯 vs 车 similarity score:', sim)

    sim = vector.similarity('本田', '车')
    print('本田 vs 车 similarity score:', sim)
    print('车 most similar', vector.similar_by_word('车'))


def read_vocab(vocab_path):
    w2i = {}
    i2w = {}
    with open(vocab_path, encoding='utf-8') as f:
        for line in f:
            item = line.strip().split()
            try:
                w2i[item[0]] = int(item[1])
                i2w[int(item[1])] = item[0]
            except:
                print(line)
                continue
    return w2i, i2w


def save_embedding(embedding, embedding_path):
    with open(embedding_path, 'w', encoding='utf-8') as f:
        for i, vector in embedding.items():
            s = str(i) + ' ' + ' '.join(map(str, vector.tolist()))+'\n'
            f.write(s)


def build_embedding(vocab_path, model_path):
    vector = KeyedVectors.load_word2vec_format(QA_WORD2VEC_BIN_PATH, binary=True)
    w2i, _ = read_vocab(vocab_path)
    vocab_size = len(w2i)
    vector_size = vector.vector_size
    embedding = {}
    count = 0
    for v, i in w2i.items():
        try:
            embedding[i] = vector[v]
            count = count + 1
        except Exception:
            # 词表中不存在，随机补充一个数据
            embedding[i] = np.random.uniform(-0.25, 0.25, vector_size).astype(np.float32)

    print('count={count}, vocab_size={vocab_size}, path={model_path}'.format(
        count=count, vocab_size=vocab_size, model_path=model_path))
    return embedding


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-t', '--train', action="store_true", help=u'训练word2vec model')
    parser.add_option('-b', '--binary', type='int', default=1, help=u'是否二进制存储')
    parser.add_option('-e', '--embedding', action="store_true", help=u'构建embedding matrix')
    parser.add_option('-s', '--similary', action="store_true", help=u'similary test')

    options, args = parser.parse_args()

    if options.train:
        save_w2v(binary=options.binary)
    elif options.similary:
        test_similar()
    elif options.embedding:
        embedding = build_embedding(QA_WORD2VEC_VOCAB_PATH, QA_WORD2VEC_BIN_PATH)
        save_embedding(embedding, QA_WORD2VEC_EMBEDDING_PATH)
