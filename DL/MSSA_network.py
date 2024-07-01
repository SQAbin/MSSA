import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import tensorflow.keras.regularizers as regularizers
from gensim import models
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras import metrics as metric
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, recall_score, f1_score, accuracy_score, precision_score
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, Callback, ModelCheckpoint, TensorBoard
from sklearn.utils import shuffle
import sys
import os
from fastevalKeras import eval


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from DL.attetion import Attention
from DL.Iterative_network import Iterative_network

from DL.trans_format import ins_to_index, train_data_loader, test_data_process
from DL.utils import test_MRR_Recall_k, model_load_path


# Siamese Network
class MSSA_nework():
    def __init__(self, Config):
        self.config = Config
        self.f_dim = self.config.f_dim
        self.model = None
        self.LossHistory = LossHistory(self.config)

        # load train data
        train_csv = self.config.train_csv_path
        self.train_df = pd.read_csv(train_csv, index_col=0)
        self.train_df = shuffle(self.train_df)

        # load test data
        test_csv = self.config.test_csv_path
        self.test_df = pd.read_csv(test_csv, index_col=0)
        self.test_df = shuffle(self.test_df)
        self.result = dict()

    def init_layers(self):
        self.w2v_model = models.KeyedVectors.load_word2vec_format(
            self.config.w2v_load_path + 'word2vec.w2v',
            binary=True)
        self.optimizer = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.lr_metric = self.get_lr_metric(self.optimizer)
        # input layer
        self.input_feature1 = tf.keras.layers.Input(shape=[None, self.config.max_block_seq, ])
        self.input_feature2 = tf.keras.layers.Input(shape=[None, self.config.max_block_seq, ])
        self.input_mask1 = tf.keras.layers.Input(shape=[None, None, ])
        self.input_mask2 = tf.keras.layers.Input(shape=[None, None, ])

        # LSTM1
        self.LS1 = Lstm1(self.config)

        # iter network layer
        self.Iterative_network = Iterative_network(self.config)

        self.bat1 = BatchNormalization(name="BatchNormalization1")
        self.bat2 = BatchNormalization(name="BatchNormalization2")

        self.ww1 = self.f_dim  # next LSTM input dim
        # LSTM2
        self.LS2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.ww1, kernel_initializer='glorot_uniform', dropout=0.5,
                                 kernel_regularizer=regularizers.l1_l2(0.001, 0.01),
                                 return_sequences=True, activation='tanh'))
        # atttion layer
        self.attention = Attention(self.config, return_sequences=False)

        # Fully connected layer
        self.Dense_1 = tf.keras.layers.Dense(128, activation="relu")
        # output layer
        self.Dense_2 = tf.keras.layers.Dense(1, activation="sigmoid")

    def build_model(self):
        ##init network
        self.init_layers()

        ###build network
        ##word2vec
        embedding_dim = self.config.w2v_dim
        embeddings = 1 * np.random.randn(len(self.w2v_model.vectors) + 1, embedding_dim)
        embeddings[0] = 0  # So that the padding will be ignored
        for i in range(0, len(self.w2v_model.vectors)):
            embeddings[i + 1] = self.w2v_model.vectors[i]
        embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings],
                                    input_length=self.config.max_seq_length,
                                    trainable=False)

        ##Embedding
        encoded_left = self.bat1(embedding_layer(self.input_feature1))
        encoded_right = self.bat1(embedding_layer(self.input_feature2))

        ##Siamese Network input1
        # node_val1 = tf.reshape(self.LS1(tf.reshape(encoded_left, [-1, self.config.max_block_seq, self.config.w2v_dim])),
        #                        [tf.shape(encoded_left)[0], -1, self.ww1])
        node_val1, cur_msg1 = self.LS1(encoded_left)
        # cur_msg1 = tf.nn.relu(node_val1)  # [1,20,]
        ##iter network
        cur_msg1 = self.Iterative_network([self.input_mask1, cur_msg1, node_val1])

        ##Siamese Network input2
        # node_val2 = tf.reshape(
        #     self.LS1(tf.reshape(encoded_right, [-1, self.config.max_block_seq, self.config.w2v_dim])),
        #     [tf.shape(encoded_right)[0], -1, self.ww1])
        node_val2, cur_msg2 = self.LS1(encoded_right)
        # cur_msg2 = tf.nn.relu(node_val2)
        ##iter network
        cur_msg2 = self.Iterative_network([self.input_mask2, cur_msg2, node_val2])

        graph_embed1 = self.bat2(cur_msg1)
        graph_embed2 = self.bat2(cur_msg2)
        graph_embed1 = self.LS2(graph_embed1)
        graph_embed2 = self.LS2(graph_embed2)

        graph_embed1 = self.attention(graph_embed1)
        graph_embed2 = self.attention(graph_embed2)

        graph_embed = concatenate([graph_embed1, graph_embed2])
        graph_embed = BatchNormalization()(graph_embed)
        graph_embed = Dropout(0.3)(graph_embed)
        graph_embed = self.Dense_1(graph_embed)
        graph_embed = BatchNormalization()(graph_embed)
        graph_embed = Dropout(0.3)(graph_embed)
        graph_embed = self.Dense_2(graph_embed)

        my_model = Model(inputs=[self.input_feature1, self.input_mask1, self.input_feature2, self.input_mask2],
                         outputs=graph_embed)

        # my_model.summary()
        # plot_model(my_model, to_file='model.png', show_shapes=True)

        my_model.compile(optimizer=self.optimizer, loss='binary_crossentropy',
                         metrics=['accuracy', metric.AUC(name='auc'), metric.Precision(name='Precision'),
                                  metric.Recall(name='Recall'), self.lr_metric])

        self.model = my_model

    # trian and test
    def train(self):
        print("start train datasets inst to index")
        self.train_df = ins_to_index(self.train_df, self.config)  # instrution to index

        # train_iter
        Data_loader_train = train_data_loader(self.train_df, self.config)

        # train
        self.model.fit(Data_loader_train, steps_per_epoch=int(len(self.train_df) / self.config.batch_size),
                       shuffle=True, epochs=self.config.epochs, callbacks=self.my_callback(), max_queue_size=10,
                       workers=1, verbose=1)
        # save weights
        self.model.save_weights(self.config.model_save_weights)

        # test
        # self.test()

    # test
    def test(self):
        print("start test datasets inst to index")
        self.test_df = ins_to_index(self.test_df, self.config)  # instrution to index

        # test AUC recall f1 acc pre
        # self.model.load_weights(model_load_path(self.config))
        self.model.load_weights(self.config.model_save_weights)
        b1, g1, b2, g2, Y_test = test_data_process(self.test_df, self.config)
        pred = self.model.predict([b1, g1, b2, g2], batch_size=1)
        fpr, tpr, _ = roc_curve(Y_test, pred, pos_label=1)
        AUC = auc(fpr, tpr)
        roc_auc = "{:.2%}".format(AUC)
        print("roc_auc: ", roc_auc)
        fig = plt.figure()
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % AUC)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        fig.savefig(self.config.model_log_path + "best_test_roc.png")

        pred_labels = np.zeros_like(pred)
        pred_labels[pred > 0.5] = 1
        recall = recall_score(Y_test, pred_labels)
        f1 = f1_score(Y_test, pred_labels)
        acc = accuracy_score(Y_test, pred_labels)
        pre = precision_score(Y_test, pred_labels)
        print("recall : ", recall)
        print("f1 : ", f1)
        print("acc : ", acc)
        print("pre : ", pre)
        self.result["recall"] = recall
        self.result["f1"] = f1
        self.result["acc"] = acc
        self.result["pre"] = pre
        self.result["roc_auc"] = roc_auc
        # test MRR and Recall@k
        # recall_at_k, mrr = test_MRR_Recall_k(self.model, self.config)
        # print("PoolSize({:d})-Recall@{}: {:.4f}".format(self.config.poolsize, self.config.k, recall_at_k))
        # print("PoolSize({:d})-:MRR: {:.4f}".format(self.config.poolsize, mrr))

        # with open(self.config.model_log_path + "test.csv", 'w', newline='') as file:
        #     headers = ['train_nums', 'test_nums', 'AUC', 'Recall', 'F1-Score', 'ACC', 'Pre',
        #                'Recall@1(poolsize-32)', 'MRR(poolsize-32)']
        #     f_csv = csv.writer(file)
        #     f_csv.writerow(headers)
        #
        #     test_result = [len(self.train_df), len(self.test_df), roc_auc, recall, f1, acc, pre, recall_at_k, mrr]
        #
        #     f_csv.writerow(test_result)

    def test_MRR_Recall_k(self):
        # test MRR and Recall@k
        recall_at_k, mrr = test_MRR_Recall_k(self.model, self.config)
        print("PoolSize({:d})-Recall@{}: {:.4f}".format(self.config.poolsize, self.config.k, recall_at_k))
        print("PoolSize({:d})-:MRR: {:.4f}".format(self.config.poolsize, mrr))
        # eval('O2','O3',self.model)

    def test_MRR_Recall_k_all(self):
        self.build_model()
        self.model.load_weights(self.config.model_save_weights)
        # eval_test(self.model)
        mrr_O0_O3, recall_O0_O3 = eval('O0', 'O3', self.model)
        mrr_O1_O3, recall_O1_O3 = eval('O1', 'O3', self.model)
        mrr_O2_O3, recall_O2_O3 = eval('O2', 'O3', self.model)
        mrr_O0_Os, recall_O0_Os = eval('O0', 'Os', self.model)
        mrr_O1_Os, recall_O1_Os = eval('O1', 'Os', self.model)
        mrr_O2_Os, recall_O2_Os = eval('O2', 'Os', self.model)
        self.result["O0-O3"] = [mrr_O0_O3, recall_O0_O3]
        self.result["O1-O3"] = [mrr_O1_O3, recall_O1_O3]
        self.result["O2-O3"] = [mrr_O2_O3, recall_O2_O3]
        self.result["OO-Os"] = [mrr_O0_Os, recall_O0_Os]
        self.result["O1-Os"] = [mrr_O1_Os, recall_O1_Os]
        self.result["O2-Os"] = [mrr_O2_Os, recall_O2_Os]
        self.result["averge"] =[(mrr_O0_O3+mrr_O1_O3+mrr_O2_O3+mrr_O0_Os+mrr_O1_Os+mrr_O2_Os)/6,(recall_O0_O3+recall_O1_O3
                                                                                                 +recall_O2_O3+recall_O0_Os+recall_O1_Os+recall_O2_Os)/6]
        print(self.result)
    # get lr
    def get_lr_metric(self, optimizer):
        def lr(y_true, y_pred):
            return optimizer._decayed_lr(tf.float32)  # use ._decayed_lr method instead of .lr

        return lr

    # callback
    def my_callback(self):
        # Callback
        EarlyStop = EarlyStopping(monitor='loss', patience=self.config.patience, restore_best_weights=True)
        CsvLog = CSVLogger(self.config.model_log_path + 'training.csv', separator=',', append=False)
        checkpoint = ModelCheckpoint(self.config.model_save_weights + 'MSSA_model' + '{epoch:02d}.h5', verbose=0,
                                     save_freq=self.config.save_freq,
                                     save_weights_only=True)

        # tensorboard_callback = TensorBoard(log_dir="./result/logs",write_images=True)
        # too much callback will increase train time
        return [EarlyStop, CsvLog, LossHistory(self.config)]  # ,tensorboard_callback]


# Record loss value of every epoch
class LossHistory(Callback):
    def __init__(self, config):
        super().__init__()
        self.losses = []
        self.config = config

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))

        # Real time curve drawing
        plt.plot(np.arange(len(self.losses)), self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(self.config.model_log_path + 'loss.png')  # Save loss curve .png


class Lstm1(tf.keras.Model):
    def __init__(self, Config):
        super(Lstm1, self).__init__()
        self.config = Config
        self.LS1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(int(self.config.f_dim / 2), kernel_initializer='glorot_uniform', dropout=0.3,
                                 return_sequences=False, activation='tanh'))

    def call(self, inputs):
        # output=self.LS1(inputs)
        node_val = tf.reshape(self.LS1(tf.reshape(inputs, [-1, self.config.max_block_seq, self.config.w2v_dim])),
                              [tf.shape(inputs)[0], -1, self.config.ww1])
        # cur_msg2 = tf.nn.relu(node_val2)
        cur_mag = tf.nn.relu(node_val)
        return node_val, cur_mag
