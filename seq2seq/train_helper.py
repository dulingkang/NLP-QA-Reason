import tensorflow as tf
import time
from utils.losses import calc_loss

START_DECODING = '[START]'


def train_model(model, dataset, params, ckpt_manager):
    # optimizer = tf.keras.optimizers.Adagrad(params['learning_rate'],
    #                                         initial_accumulator_value=params['adagrad_init_acc'],
    #                                         clipnorm=params['max_grad_norm'])
    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=params["learning_rate"])
    # from_logits = True: preds is model output before passing it into softmax (so we pass it into softmax)
    # from_logits = False: preds is model output after passing it into softmax (so we skip this step)
    # 要看模型在decoder最后输出是否经过softmax
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

    # 定义损失函数
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 1))
        dec_lens = tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=-1)

        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        loss_ = tf.reduce_sum(loss_, axis=-1)/dec_lens
        return tf.reduce_mean(loss_)

    @tf.function()
    def train_step(enc_inp, dec_tar, dec_inp):
        # loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = model.call_encoder(enc_inp)
            dec_hidden = enc_hidden
            # start index
            pred, _ = model(enc_output,  # shape=(3, 200, 256)
                            dec_inp,  # shape=(3, 256)
                            dec_hidden,  # shape=(3, 200)
                            dec_tar)  # shape=(3, 50) 
            loss = loss_function(dec_tar, pred)
                        
        # variables = model.trainable_variables
        variables = model.encoder.trainable_variables + model.attention.trainable_variables + model.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    best_loss = 20
    epochs = params['epochs']
    for epoch in range(epochs):
        t0 = time.time()
        step = 0
        total_loss = 0
        # for step, batch in enumerate(dataset.take(params['max_train_steps'])):
        for batch in dataset:
        # for batch in dataset.take(params['max_train_steps']):
            loss = train_step(batch[0]["enc_input"],  # shape=(16, 200)
                              batch[1]["dec_target"], # shape=(16, 50)
                              batch[1]["dec_input"])
           
            step += 1
            total_loss += loss
            if step % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, step, total_loss / step))
                # print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, step, loss.numpy()))

        if epoch % 2 == 0:
            if total_loss / step < best_loss:
                best_loss = total_loss / step
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch + 1, ckpt_save_path, best_loss))
                print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / step))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - t0))


def train_model_pgn(model, dataset, params, ckpt_manager):
    epochs = params['epochs']
    optimizer = tf.keras.optimizers.Adagrad(
        params['learning_rate'],
        initial_accumulator_value=params['adagrad_init_acc'],
        clipnorm=params['max_grad_norm'],
        epsilon=params['eps']
    )
    # optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=params["learning_rate"])

    # @tf.function()
    def train_step(enc_inp, extended_enc_input, max_oov_len,
                   dec_input, dec_target,
                   enc_pad_mask, padding_mask):
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = model.call_encoder(enc_inp)
            dec_hidden = enc_hidden
            final_dists, _, attentions, coverages = model(
                dec_hidden,
                enc_output,
                dec_input,
                extended_enc_input,
                max_oov_len,
                enc_pad_mask=enc_pad_mask,
                use_coverage=params['use_coverage'],
                prev_coverage=None
            )
            batch_loss, log_loss, cov_loss = calc_loss(
                dec_target, final_dists, padding_mask, attentions,
                params['cov_loss_wt'],
                params['use_coverage'],
                params['pointer_gen']
            )

        # variables = model.trainable_variables
        variables = model.encoder.trainable_variables + \
                    model.attention.trainable_variables + \
                    model.decoder.trainable_variables + \
                    model.pointer.trainable_variables
        gradients = tape.gradient(batch_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss, log_loss, cov_loss

    max_train_steps = params['max_train_steps']
    iter_num = 0
    best_loss = 20
    for epoch in range(epochs):
        start = time.time()
        total_loss = 0
        total_log_loss = 0
        total_cov_loss = 0
        step = 0
        for encoder_batch_data, decoder_batch_data in dataset:
            batch_loss, log_loss, cov_loss = train_step(
                encoder_batch_data["enc_input"],
                encoder_batch_data["extended_enc_input"],
                encoder_batch_data["max_oov_len"],
                decoder_batch_data["dec_input"],
                decoder_batch_data["dec_target"],
                enc_pad_mask=encoder_batch_data["encoder_pad_mask"],
                padding_mask=decoder_batch_data["decoder_pad_mask"]
            )

            total_loss += batch_loss
            total_log_loss += log_loss
            total_cov_loss += cov_loss
            step += 1
            iter_num += 1
            if step % 10 == 0:
                if params['use_coverage']:
                    msg = 'Epoch {} Batch {} avg_loss {:.4f} log_loss {:.4f} cov_loss {:.4f}'
                    print(msg.format(
                        epoch + 1,
                        step,
                        total_loss / step,
                        total_log_loss / step,
                        total_cov_loss / step)
                    )
                else:
                    print('Epoch {} Batch {} avg_loss {:.4f}'.format(
                        epoch + 1,
                        step,
                        total_loss / step)
                    )
            if iter_num >= max_train_steps:
                break

        # if (epoch + 1) % 1 == 0:
        if total_loss / step < best_loss:
            best_loss = total_loss / step
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch + 1, ckpt_save_path, best_loss))
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / step))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        if iter_num >= max_train_steps:
            break
