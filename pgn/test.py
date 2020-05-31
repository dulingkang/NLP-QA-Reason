# -*- coding:utf-8 -*-
# Created by LuoJie at 12/12/19
import tensorflow as tf
import pandas as pd

import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from pgn.batcher import beam_test_batch_generator
from pgn.model import PGN

from pgn.test_helper import beam_decode, greedy_decode

from utils.config import TEST_DATA, PGN_CKPT
from utils.config_gpu import config_gpu
from utils.params import get_params
from utils.saveLoader import Vocab
from pgn.batcher import batcher


def test(params):
    assert params["mode"].lower() in ["test", "eval"], "change training mode to 'test' or 'eval'"
    assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"
    # GPU资源配置
    config_gpu()

    print("Test the model ...")

    model = PGN(params)

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    # ds = batcher(vocab, params)

    print("Creating the checkpoint manager")
    checkpoint = tf.train.Checkpoint(PGN=model)

    checkpoint_manager = tf.train.CheckpointManager(checkpoint, PGN_CKPT, max_to_keep=5)

    # checkpoint_manager = tf.train.CheckpointManager(checkpoint, TEMP_CKPT, max_to_keep=5)
    # temp_ckpt = os.path.join(TEMP_CKPT, "ckpt-5")
    # checkpoint.restore(temp_ckpt)

    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    print("Model restored")

    if params['greedy_decode']:
        params['batch_size'] = 512
        results = predict_result(model, params, vocab, params['result_save_path'])
    else:
        b = beam_test_batch_generator(params["beam_size"])
        results = []
        for batch in b:
            best_hyp = beam_decode(model, batch, vocab, params)
            results.append(best_hyp.abstract)
        save_predict_result(results, params['result_save_path'])
        print('save result to :{}'.format(params['result_save_path']))

    return results


def predict_result(model, params, vocab, result_save_path):

    dataset = batcher(vocab, params)

    # 预测结果
    results = greedy_decode(model, dataset, vocab, params)

    results = list(map(lambda x: x.replace(" ",""), results))
    # 保存结果
    save_predict_result(results, result_save_path)

    return results


def save_predict_result(results, result_save_path):
    # 读取结果
    test_df = pd.read_csv(TEST_DATA)
    # 填充结果
    test_df['Prediction'] = results
    # 　提取ID和预测结果两列
    test_df = test_df[['QID', 'Prediction']]
    # 保存结果.
    test_df.to_csv(result_save_path, index=None, sep=',')


if __name__ == '__main__':
    # 获得参数
    params = get_params()
    params["mode"] = "test"
    params["batch_size"] = params["beam_size"]
    # 获得参数
    results = test(params)
