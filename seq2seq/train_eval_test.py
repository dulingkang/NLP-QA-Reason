import os
import tensorflow as tf
from rouge import Rouge

from seq2seq.model.sequence_to_sequence import SequenceToSequence
from seq2seq.model.pgn import PGN
from seq2seq.batcher import batcher, Vocab
from seq2seq.train_helper import train_model, train_model_pgn
from seq2seq.test_helper import beam_decode
from tqdm import tqdm
from utils.data_utils import get_result_filename
import pandas as pd


def init_checkpoint(dir):
    checkpoint = tf.train.Checkpoint()
    init_path = checkpoint.save(os.path.join(dir, 'init'))
    checkpoint.restore(init_path)


def prepare(params):
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
        tf.config.experimental.set_memory_growth(gpus[0], enable=True)

    vocab = Vocab(params["vocab_path"], params["max_vocab_size"])
    params['vocab_size'] = vocab.count
    print('true vocab is ', vocab)

    print("Creating the batcher ...")
    b = batcher(vocab, params)

    print("Building the model ...")
    if params.get('pointer_gen'):
        model = PGN(params)
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), PGN=model)
        checkpoint_dir = params["pgn_model_dir"]
    else:
        model = SequenceToSequence(params)
        ckpt = tf.train.Checkpoint(SequenceToSequence=model)
        checkpoint_dir = params["seq2seq_model_dir"]
    print("Creating the checkpoint manager", checkpoint_dir)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    return model, vocab, b, params, ckpt_manager


def train(params):
    model, vocab, b, params, ckpt_manager = prepare(params)

    print("Starting the training ...")
    if params.get('pointer_gen'):
        train_model_pgn(model, b, params, ckpt_manager)
    else:
        train_model(model, b, params, ckpt_manager)


def predict_result(params):
    result = []
    model, vocab, b, params, ckpt_manager = prepare(params)
    for batch in b:
        # yield beam_decode(model, batch, vocab, params)
        result.append(beam_decode(model, batch, vocab, params))
    print('predict result', result)


def evaluate(params):
    gen = predict_result(params)
    reals = []
    preds = []
    with tqdm(total=params["max_num_to_eval"],position=0, leave=True) as pbar:
        for i in range(params["max_num_to_eval"]):
            trial = next(gen)
            reals.append(trial.real_abstract)
            preds.append(trial.abstract)
            pbar.update(1)
    r=Rouge()
    scores = r.get_scores(preds, reals, avg=True)
    print("\n\n")
    print(scores)


def save_predict_result(results, params):
    # 读取结果
    test_df = pd.read_csv(params['test_x_dir'])
    # 填充结果
    test_df['Prediction'] = results[:20000]
    # 　提取ID和预测结果两列
    test_df = test_df[['QID', 'Prediction']]
    # 保存结果.
    result_save_path = get_result_filename(params)
    test_df.to_csv(result_save_path, index=None, sep=',')


if __name__ == '__main__':
    pass

