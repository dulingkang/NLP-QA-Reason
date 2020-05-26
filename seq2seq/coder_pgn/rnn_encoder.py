import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, embedding_matrix, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units

        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)
        self.gru = tf.keras.layers.GRU(
            self.enc_units, return_sequences=True,
            return_state=True, recurrent_initializer='glorot_uniform')
        self.bidirectional_gru = tf.keras.layers.Bidirectional(self.gru)

    def __call__(self, x, hidden):
        x = self.embedding(x)
        # output, hidden = self.gru(x, initial_state=hidden)
        output, forward_state, backward_state = self.bidirectional_gru(x, initial_state=[hidden, hidden])
        hidden = tf.keras.layers.concatenate([forward_state, backward_state], axis=-1)
        return output, hidden

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))