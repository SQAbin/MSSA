import tensorflow as tf
import tensorflow.keras.backend as K
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


class Attention(tf.keras.layers.Layer):
    def __init__(self,Config, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention, self).__init__()
        self.config=Config
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[2], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(self.config.num_block,1),#(block_num,1)
                                 initializer="zeros")

        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)
