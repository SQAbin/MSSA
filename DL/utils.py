import os
import sys
import re

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from build_datasets.drug_datasets_from_json import make_test_dataset_MRR_Recall_k

from DL.trans_format import test_data_process, ins_to_index
import pandas as pd
import copy
import numpy as np


# test MRR and Recall@k on test data
def test_MRR_Recall_k( MSSA, config):
    # load weights
    MSSA.load_weights(config.model_load_weights)
    # make temp datasets for MRR and Recall@k
    make_test_dataset_MRR_Recall_k(config.poolsize, config.MRR_Recall_k_temp_path, config.path_test_json)
    temp_df = pd.read_csv(config.MRR_Recall_k_temp_path, index_col=0)

    # instrutions to index
    print("start MRR_Recall@k datasets inst to index")
    temp_df = ins_to_index(temp_df,config)
    b1, g1, b2, g2, Y_test = test_data_process(temp_df,config)

    K = config.k  # 前K个最相关预测
    recall_at_k = 0.0
    mrr = 0.0
    for i in range(0, config.poolsize):
        b1_32 = copy.deepcopy(b1)
        g1_32 = copy.deepcopy(g1)
        b2_32 = b2
        g2_32 = g2
        Y_32 = Y_test
        for j in range(0, config.poolsize):
            b1_32[j] = b1_32[i]
            g1_32[j] = g1_32[i]

            Y_32[j] = j
        pred = MSSA.predict([b1_32, g1_32, b2_32, g2_32], batch_size=1)
        # print(pred)
        pred = pred.flatten()
        idx_sorted = np.argsort(-pred)
        # print("idx_sorted", idx_sorted)
        idex_sorted_recall = idx_sorted[:K]
        # print("idex_sorted_recall: ", idex_sorted_recall)
        rank = np.where(idx_sorted == i)[0][0] + 1
        # print("rank: ", rank)
        mrr += 1 / rank
        if Y_32[i] in idex_sorted_recall:
            recall_at_k += 1
        # print("recall_at_k:", recall_at_k)

    recall_at_k /= len(Y_test)
    mrr /= len(Y_test)
    # print("len", len(Y_test))

    return recall_at_k,mrr
    # print(classification_report(Y_test, pred_labels, digits=4))


def model_load_path(config):
    # 设定模型数据所在的目录
    model_dir = config.model_load_weights

    # 获取模型文件列表
    model_list = os.listdir(model_dir)

    # 定义匹配模式，查找文件名中的数字部分
    pattern = re.compile(r'\d+')

    # 定义变量储存当前最大的数字
    max_num = 0
    for model_file in model_list:
        match = pattern.search(model_file)
        if match:
            num = int(match.group())
            max_num = max(num, max_num)

    # print(max_num)
    target_model_file = config.model_load_weights + f'MSSA_model{max_num:02d}.h5'
    # print(target_model_file)
    return target_model_file
