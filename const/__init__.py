# coding: utf-8
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

QA_ROOT_DIR = BASE_DIR + '/NLP-QA-Reason'
# QA_ROOT_DIR = '/home/ma-user/work'
# QA_ROOT_DIR = '/content/drive/My Drive'
QA_DATA_DIR = QA_ROOT_DIR + '/data'

QA_TRAIN_DATA_PATH = QA_DATA_DIR + '/AutoMaster_TrainSet.csv'
QA_TEST_DATA_PATH = QA_DATA_DIR + '/AutoMaster_TestSet.csv'
QA_TRAIN_CLEAN_X_PATH = QA_DATA_DIR + '/train_set_seg_x.txt'
QA_TRAIN_CLEAN_Y_PATH = QA_DATA_DIR + '/train_set_seg_y.txt'
QA_TEST_CLEAN_X_PATH = QA_DATA_DIR + '/test_set_seg_x.txt'
QA_STOPWORDS_PATH = QA_DATA_DIR + '/stopwords.txt'
QA_SENTENCE_PATH = QA_DATA_DIR + '/sentences.txt'

QA_WORD2VEC_PKL_OUT_PATH = QA_DATA_DIR + '/word2vec.txt'
QA_WORD2VEC_BIN_PATH = QA_DATA_DIR + '/word2vec.bin'
QA_WORD2VEC_VOCAB_PATH = QA_DATA_DIR + '/vocab.txt'
QA_WORD2VEC_EMBEDDING_PATH = QA_DATA_DIR + '/embedding.txt'
QA_WORD2VEC_EMBEDDING_NPY_PATH = QA_DATA_DIR + '/embedding_matrix.npy'

QA_CKPT_OUTPUT_DIR = QA_ROOT_DIR + '/ckpt'
QA_CKPT_SEQ2SEQ_DIR = QA_CKPT_OUTPUT_DIR + '/seq2seq'
QA_CKPT_PNG_DIR = QA_CKPT_OUTPUT_DIR + '/pgn'
