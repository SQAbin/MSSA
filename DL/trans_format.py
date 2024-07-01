import numpy as np
from gensim import models
import tensorflow as tf
from sklearn.utils import shuffle
from tqdm import tqdm
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from tensorflow.keras.preprocessing.sequence import pad_sequences
import re


def text_to_word_list(text):
    text = str(text)
    text = text.strip()
    text = text.lower()
    text = text.split(' ')
    for i in range(0, len(text)):
        text[i] = text[i].strip()
    return text


def str_to_list(strr):
    tempp = []
    if len(strr) == 0:
        return tempp
    temp = strr.split(', [')
    for j in temp:
        temp1 = []
        num = re.findall('\d+', j)
        if len(num) == 0:
            tempp.append(temp1)
            continue
        for k in num:
            temp1.append(int(k))
        tempp.append(temp1)

    return tempp


def ins_to_index(datasets,config):
    # Replace the instruction with the corresponding index in this vector

    model = models.KeyedVectors.load_word2vec_format(
        config.w2v_load_path + 'word2vec.w2v', binary=True)

    # function1
    batch_f1_blocks = np.array(datasets['f1_blocks'])
    if len(batch_f1_blocks) > 9999:
        for blocks in tqdm(range(0, len(batch_f1_blocks))):
            # blocks ->[block1,block2...]
            bb1 = batch_f1_blocks[blocks].strip('[]').split(', ')
            bb1 = [s.strip('\'').strip(' ') for s in bb1]
            batch_f1_blocks[blocks] = bb1
            for i in range(0, len(bb1)):
                bt = bb1[i].split(' ')
                q2n = []
                for j in range(0, len(bt)):
                    if bt[j] not in model.index_to_key:
                        q2n.append(0)
                        # print("Unknown workd is found!!!")
                    else:
                        q2n.append(model.key_to_index[bt[j]] + 1)

                batch_f1_blocks[blocks][i] = q2n
    else:
        for blocks in range(0, len(batch_f1_blocks)):
            # blocks ->[block1,block2...]
            bb1 = batch_f1_blocks[blocks].strip('[]').split(', ')
            bb1 = [s.strip('\'').strip(' ') for s in bb1]
            batch_f1_blocks[blocks] = bb1
            for i in range(0, len(bb1)):
                bt = bb1[i].split(' ')
                q2n = []
                for j in range(0, len(bt)):
                    if bt[j] not in model.index_to_key:
                        q2n.append(0)
                        # print("Unknown workd is found!!!")
                    else:
                        q2n.append(model.key_to_index[bt[j]] + 1)

                batch_f1_blocks[blocks][i] = q2n
    datasets['f1_blocks'] = batch_f1_blocks

    # function2
    batch_f2_blocks = np.array(datasets['f2_blocks'])
    if len(batch_f2_blocks)>9999:
        for blocks in tqdm(range(0, len(batch_f2_blocks))):
            bb1 = batch_f2_blocks[blocks].strip('[]').split(', ')
            bb1 = [s.strip('\'').strip(' ') for s in bb1]
            batch_f2_blocks[blocks] = bb1

            for i in range(0, len(bb1)):
                bt = bb1[i].split(' ')
                q2n = []
                for j in range(0, len(bt)):

                    # print(bt[j])
                    if bt[j] not in model.index_to_key:
                        q2n.append(0)
                        # print("Unknown workd is found!!!")
                    else:
                        q2n.append(model.key_to_index[bt[j]] + 1)

                batch_f2_blocks[blocks][i] = q2n
    else:
        for blocks in range(0, len(batch_f2_blocks)):
            bb1 = batch_f2_blocks[blocks].strip('[]').split(', ')
            bb1 = [s.strip('\'').strip(' ') for s in bb1]
            batch_f2_blocks[blocks] = bb1

            for i in range(0, len(bb1)):
                bt = bb1[i].split(' ')
                q2n = []
                for j in range(0, len(bt)):

                    # print(bt[j])
                    if bt[j] not in model.index_to_key:
                        q2n.append(0)
                        # print("Unknown workd is found!!!")
                    else:
                        q2n.append(model.key_to_index[bt[j]] + 1)

                batch_f2_blocks[blocks][i] = q2n
    datasets['f2_blocks'] = batch_f2_blocks
    return datasets


class train_data_loader(tf.keras.utils.Sequence):

    def __init__(self, train, config, shuffle=True):
        self.train = train
        self.y = self.train['eq']
        self.config=config
        self.batch_size = config.batch_size
        self.shuffle = shuffle
        self.max_block_seq = config.max_block_seq
        self.num_block = config.num_block

    def __len__(self):
        return int(len(self.train) / self.batch_size)

    def __getitem__(self, idx):
        # print(self.train)
        batch_f1_child = np.array(self.train['f1_child'][idx * self.batch_size:(idx + 1) * self.batch_size])

        batch_f1_blocks = np.array(self.train['f1_blocks'][idx * self.batch_size:(idx + 1) * self.batch_size])
        for i in range(0, len(batch_f1_blocks)):
            batch_f1_blocks[i] = pad_sequences(batch_f1_blocks[i], self.max_block_seq, padding='post',
                                               truncating='post')
        batch_f1_blocks = pad_sequences(batch_f1_blocks, self.num_block, padding='post', truncating='post')

        batch_f2_blocks = np.array(self.train['f2_blocks'][idx * self.batch_size:(idx + 1) * self.batch_size])
        for i in range(0, len(batch_f2_blocks)):
            batch_f2_blocks[i] = pad_sequences(batch_f2_blocks[i], self.max_block_seq, padding='post',
                                               truncating='post')
        batch_f2_blocks = pad_sequences(batch_f2_blocks, self.num_block, padding='post', truncating='post')

        adj_f1 = []
        for f1_child in batch_f1_child:
            # f1_child ->[[1,2],[2,3],[4]...]
            f1_child_padding = []
            f1_child = str_to_list(f1_child)

            for elem_list in f1_child:
                # [1,2]->temp1
                zerolist = [0 for zero in range(0, self.config.num_block)]
                if len(elem_list) <= 0:
                    f1_child_padding.append(zerolist)
                    continue

                for elem in elem_list:
                    # temp2 -> 1
                    if int(elem) >= self.config.num_block:
                        continue
                    zerolist[int(elem)] = 1
                f1_child_padding.append(zerolist)
            for temp in range(0, self.config.num_block):
                if len(f1_child_padding) < self.config.num_block:
                    f1_child_padding.append([0 for zero in range(0, self.config.num_block)])
                else:
                    f1_child_padding = f1_child_padding[0:self.config.num_block]
                    break
            adj_f1.append(f1_child_padding)

        batch_f2_child = np.array(self.train['f2_child'][idx * self.batch_size:(idx + 1) * self.batch_size])
        # batch_f2_feature = np.array(self.train['f2_feature'][idx * batch_size:(idx + 1) * batch_size])
        ###child2
        adj_f2 = []
        for f2_child in batch_f2_child:
            # f1_child ->[[1,2],[2,3],[4]...]
            f2_child_padding = []
            f2_child = str_to_list(f2_child)

            for elem_list in f2_child:
                # [1,2]->temp1
                zerolist = [0 for zero in range(0, self.config.num_block)]
                if len(elem_list) <= 0:
                    f2_child_padding.append(zerolist)
                    continue

                for elem in elem_list:
                    # temp2 -> 1
                    if int(elem) >= self.config.num_block:
                        continue
                    zerolist[int(elem)] = 1
                f2_child_padding.append(zerolist)
            for temp in range(0, self.config.num_block):
                if len(f2_child_padding) < self.config.num_block:
                    f2_child_padding.append([0 for zero in range(0, self.config.num_block)])
                else:
                    f2_child_padding = f2_child_padding[0:self.config.num_block]
                    break
            adj_f2.append(f2_child_padding)

        for i in range(0, len(adj_f1)):
            adj_f1[i] = tf.convert_to_tensor(adj_f1[i], dtype='int32')

        adj_f1 = tf.convert_to_tensor([tf.convert_to_tensor(item) for item in adj_f1])

        for i in range(0, len(adj_f2)):
            adj_f2[i] = tf.convert_to_tensor(adj_f2[i], dtype='int32')

        adj_f2 = tf.convert_to_tensor([tf.convert_to_tensor(item) for item in adj_f2])

        batch_y = self.train['eq'][idx * self.batch_size:(idx + 1) * self.batch_size]

        adj_f1, adj_f2 = np.array(adj_f1), np.array(adj_f2)

        return [batch_f1_blocks, adj_f1, batch_f2_blocks, adj_f2], np.array(batch_y)

    def on_epoch_end(self):

        if self.shuffle == True:
            self.train = shuffle(self.train)
            self.y = self.train['eq']


def test_data_process(test_df,config):
    batch_f1_child = np.array(test_df['f1_child'])
    # batch_f1_feature = np.array(x_y['f1_feature'])

    batch_f1_blocks = np.array(test_df['f1_blocks'])

    for i in range(0, len(batch_f1_blocks)):
        batch_f1_blocks[i] = pad_sequences(batch_f1_blocks[i], config.max_block_seq, padding='post', truncating='post')
    batch_f1_blocks = pad_sequences(batch_f1_blocks, config.num_block, padding='post', truncating='post')

    batch_f2_blocks = np.array(test_df['f2_blocks'])
    for i in range(0, len(batch_f2_blocks)):
        batch_f2_blocks[i] = pad_sequences(batch_f2_blocks[i], config.max_block_seq, padding='post',
                                           truncating='post')
    batch_f2_blocks = pad_sequences(batch_f2_blocks, config.num_block, padding='post', truncating='post')
    # print("A: ", batch_f1_blocks)

    # print(batch_f1_feature)
    ###child1
    adj_f1 = []
    for f1_child in batch_f1_child:
        # f1_child ->[[1,2],[2,3],[4]...]
        f1_child_padding = []
        f1_child = str_to_list(f1_child)

        for elem_list in f1_child:
            # [1,2]->temp1
            zerolist = [0 for zero in range(0, config.num_block)]
            if len(elem_list) <= 0:
                f1_child_padding.append(zerolist)
                continue

            for elem in elem_list:
                # temp2 -> 1
                if int(elem) >= config.num_block:
                    continue
                zerolist[int(elem)] = 1
            f1_child_padding.append(zerolist)
        for temp in range(0, config.num_block):
            if len(f1_child_padding) < config.num_block:
                f1_child_padding.append([0 for zero in range(0, config.num_block)])
            else:
                f1_child_padding = f1_child_padding[0:config.num_block]
                break
        adj_f1.append(f1_child_padding)


    batch_f2_child = np.array(test_df['f2_child'])
    # batch_f2_feature = np.array(x_y['f2_feature'])
    ###child2
    adj_f2 = []
    for f2_child in batch_f2_child:
        # f1_child ->[[1,2],[2,3],[4]...]
        f2_child_padding = []
        f2_child = str_to_list(f2_child)

        for elem_list in f2_child:
            # [1,2]->temp1
            zerolist = [0 for zero in range(0, config.num_block)]
            if len(elem_list) <= 0:
                f2_child_padding.append(zerolist)
                continue

            for elem in elem_list:
                # temp2 -> 1
                if int(elem) >= config.num_block:
                    continue
                zerolist[int(elem)] = 1
            f2_child_padding.append(zerolist)
        for temp in range(0, config.num_block):
            if len(f2_child_padding) < config.num_block:
                f2_child_padding.append([0 for zero in range(0, config.num_block)])
            else:
                f2_child_padding = f2_child_padding[0:config.num_block]
                break
        adj_f2.append(f2_child_padding)

    for i in range(0, len(adj_f1)):
        adj_f1[i] = tf.convert_to_tensor(adj_f1[i], dtype='int32')

    # print("aaa: ", aaa)
    adj_f1 = tf.convert_to_tensor([tf.convert_to_tensor(item) for item in adj_f1])
    # print("aaa: ", aaa)
    for i in range(0, len(adj_f2)):
        adj_f2[i] = tf.convert_to_tensor(adj_f2[i], dtype='int32')
    # print("aaa1: ",aaa1)
    adj_f2 = tf.convert_to_tensor([tf.convert_to_tensor(item) for item in adj_f2])

    adj_f1, adj_f2 = np.array(adj_f1), np.array(adj_f2)
    # batch_x1 = [batch_f1_feature, aaa]
    # batch_x2 = [batch_f2_feature, aaa1]
    pred = np.array(test_df['eq'])

    return (batch_f1_blocks, adj_f1, batch_f2_blocks, adj_f2, pred)
