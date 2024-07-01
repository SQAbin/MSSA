import tensorflow as tf
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


class Iterative_network(tf.keras.Model):
    def __init__(self, Config):
        super(Iterative_network, self).__init__()
        self.config = Config
        # 定义模型层
        self.iter_level = self.config.iter_level
        self.wi1 = tf.keras.layers.Dense(128, trainable=True, kernel_initializer='glorot_uniform', )
        self.wi2 = tf.keras.layers.Dense(128, trainable=True, activation='relu', kernel_initializer='glorot_uniform')
        self.wi3 = tf.keras.layers.Dense(self.config.f_dim, trainable=True, kernel_initializer='glorot_uniform')
        self.Embed_Dense = [self.wi1, self.wi2, self.wi3]  # 嵌入Dense层
        self.ww1 = self.config.ww1

    def call(self, inputs):
        # 定义模型的正向传播过程inputs[0] ->mask
        # inputs[1] ->cur_msg1
        # inputs[2] ->node_val1
        for t in range(0, self.iter_level):  ##迭代5次
            Li_t1 = tf.matmul(inputs[0], inputs[1])

            cur_info1 = tf.reshape(Li_t1, [-1, self.ww1])
            cur_info1 = tf.nn.relu(self.Embed_Dense[0](cur_info1))  ##嵌入深度为3
            cur_info1 = self.Embed_Dense[1](cur_info1)
            cur_info1 = self.Embed_Dense[2](cur_info1)

            neighbor_val_t1 = tf.reshape(cur_info1, tf.shape(Li_t1))
            # Adding
            tot_val_t1 = inputs[2] + neighbor_val_t1
            tot_msg_t = tf.nn.tanh(tot_val_t1)
            inputs[1] = tot_msg_t
        return inputs[1]
