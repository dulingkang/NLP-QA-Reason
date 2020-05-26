# coding=utf-8
from datetime import datetime
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import tensorflow as tf
import argparse
from seq2seq.train_eval_test import train, predict_result


from const import (
    QA_CKPT_SEQ2SEQ_DIR,
    QA_CKPT_PNG_DIR,
    QA_TRAIN_CLEAN_X_PATH,
    QA_TRAIN_CLEAN_Y_PATH,
    QA_TEST_CLEAN_X_PATH,
    QA_WORD2VEC_VOCAB_PATH,
    QA_WORD2VEC_PKL_OUT_PATH,
    QA_DATA_DIR,
    QA_WORD2VEC_EMBEDDING_NPY_PATH)

NUM_SAMPLES = 82706


# 获取项目根目录
# root = pathlib.Path(os.path.abspath(__file__)).parent.parent

def main():
    print('start', datetime.now())
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument("--max_enc_len", default=200, help="Encoder input max sequence length", type=int)
    parser.add_argument("--max_dec_len", default=40, help="Decoder input max sequence length", type=int)
    parser.add_argument("--max_dec_steps", default=100,
                        help="maximum number of words of the predicted abstract", type=int)
    parser.add_argument("--min_dec_steps", default=30,
                        help="Minimum number of words of the predicted abstract", type=int)
    parser.add_argument("--batch_size", default=64, help="batch size", type=int)
    parser.add_argument("--adagrad_init_acc", default=0.1,
                        help="Adagrad optimizer initial accumulator value. "
                             "Please refer to the Adagrad optimizer API documentation "
                             "on tensorflow site for more details.",
                        type=float)
    parser.add_argument('--rand_unif_init_mag', default=0.02,
                        help='magnitude for lstm cells random uniform inititalization', type=float)
    parser.add_argument('--eps', default=1e-12, help='eps',
                        type=float)

    parser.add_argument('--trunc_norm_init_std', default=1e-4, help='std of trunc norm init, '
                                                                    'used for initializing everything else',
                        type=float)

    parser.add_argument('--cov_loss_wt', default=1.0, help='Weight of coverage loss (lambda in the paper).'
                                                           ' If zero, then no incentive to minimize coverage loss.',
                        type=float)

    parser.add_argument('--max_grad_norm', default=2.0, help='for gradient clipping', type=float)
    parser.add_argument("--learning_rate", default=0.001, help="Learning rate", type=float)

    parser.add_argument("--vocab_size", default=30000, help="max vocab size , None-> Max ", type=int)
    parser.add_argument("--max_vocab_size", default=30000, help="max vocab size , None-> Max ", type=int)

    parser.add_argument("--beam_size", default=2,
                        help="beam size for beam search decoding (must be equal to batch size in decode mode)",
                        type=int)
    parser.add_argument("--embed_size", default=300, help="Words embeddings dimension", type=int)
    parser.add_argument("--enc_units", default=128, help="Encoder GRU cell units number", type=int)
    parser.add_argument("--dec_units", default=256, help="Decoder GRU cell units number", type=int)
    parser.add_argument("--attn_units", default=256, help="[context vector, decoder state, decoder input] feedforward \
                               result dimension - this result is used to compute the attention weights",
                        type=int)
    # path
    # /ckpt/checkpoint/checkpoint
    parser.add_argument("--seq2seq_model_dir", default=QA_CKPT_SEQ2SEQ_DIR, help="seq2seq Model folder")
    parser.add_argument("--pgn_model_dir", default=QA_CKPT_PNG_DIR, help="pgn Model folder")
    parser.add_argument("--model_path", help="Path to a specific model", default="", type=str)
    parser.add_argument("--train_seg_x_dir", default=QA_TRAIN_CLEAN_X_PATH, help="train_seg_x_dir")
    parser.add_argument("--train_seg_y_dir", default=QA_TRAIN_CLEAN_Y_PATH, help="train_seg_y_dir")
    parser.add_argument("--test_seg_x_dir", default=QA_TEST_CLEAN_X_PATH, help="test_seg_x_dir")
    parser.add_argument("--vocab_path", default=QA_WORD2VEC_VOCAB_PATH, help="Vocab path")
    parser.add_argument("--word2vec_output", default=QA_WORD2VEC_PKL_OUT_PATH, help="Vocab txt path")
    parser.add_argument("--log_file", help="File in which to redirect console outputs", default="", type=str)
    parser.add_argument("--test_save_dir", default=QA_DATA_DIR, help="test_save_dir")
    parser.add_argument("--test_x_dir", default=QA_TEST_CLEAN_X_PATH, help="test_x_dir")
    parser.add_argument("--embedding_npy", default=QA_WORD2VEC_EMBEDDING_NPY_PATH, help="embedding_npy")

    # others
    parser.add_argument("--max_train_steps", default=1300, help="max_train_steps", type=int)
    parser.add_argument("--checkpoints_save_steps", default=10, help="Save checkpoints every N steps", type=int)
    parser.add_argument("--num_to_test", default=10, help="Number of examples to test", type=int)
    parser.add_argument("--max_num_to_eval", default=5, help="max_num_to_eval", type=int)
    parser.add_argument("--epochs", default=20, help="train epochs", type=int)

    parser.add_argument('--d_model', default=512, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    parser.add_argument('--dff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--dropout_rate', default=0.1, type=float)

    # mode
    parser.add_argument("--mode", default='test', help="training, eval or test options")
    parser.add_argument("--model", default='PGN', help="which model to be slected")
    parser.add_argument("--pointer_gen", default=True, help="training, eval or test options")
    parser.add_argument("--use_coverage", default=True, help="use_coverage")
    parser.add_argument("--greedy_decode", default=False, help="greedy_decoder")
    parser.add_argument("--transformer", default=False, help="transformer")

    args = parser.parse_args()
    params = vars(args)
    print(params)
    if params["mode"] == "train":
        params["max_train_steps"] = NUM_SAMPLES // params["batch_size"]
        train(params)

    # elif params["mode"] == "eval":
    #     evaluate(params)
    elif params["mode"] == "test":
        params["batch_size"] = params["beam_size"]
        predict_result(params)
    print('end', datetime.now())

if __name__ == '__main__':
    main()
