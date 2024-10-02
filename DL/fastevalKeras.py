import pickle
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import argparse
import json
import DL.config as config
import csv
import pandas as pd
import copy
from DL.trans_format import test_data_process, ins_to_index
from sklearn.metrics import auc, roc_curve, recall_score, f1_score, accuracy_score, precision_score
from random import *



def eval(TYPE1, TYPE2,model):

    with open(config.path_test_json) as file_obj:
        numbers = json.load(file_obj)
    datasets=[]
    # funcarr2 = []
    SIMS = []
    Recall_AT_1 = []

    count = 0
    Names=[]
    for name in numbers:
        Names.append(name)
    shuffle(Names)
    for i in Names:

        if TYPE1 in numbers[i]:
            if TYPE2 in numbers[i]:
                if len(str(numbers[i][TYPE1][0]).split(".")) <= 6 or len(str(numbers[i][TYPE2][0]).split("."))<=6:
                    continue
                #print(numbers[i][TYPE1])
                #print(numbers[i][TYPE1][2])
                #print("111")
                A=[count, i,numbers[i][TYPE1][3], numbers[i][TYPE1][1],numbers[i][TYPE2][3],numbers[i][TYPE2][1]]

                datasets.append(A)
                # funcarr2.append(B)
                count = count + 1
                #if count >33:
                #    break
        else:
            continue
    # count = 0
    #print(datasets)
    for i in tqdm(range(0, int(len(datasets) / config.poolsize))):
        if i % 100 == 0:
            print(i)
        dataset = datasets[i * config.poolsize:(i * config.poolsize + config.poolsize)]
        # temp2 = funcarr2[i * config.poolsize:(i * config.poolsize + config.poolsize)]
        # print(temp1)
        with open(config.MRR_Recall_k_temp_path, 'w', newline='') as f1:


            headers = ['index', 'function_name', 'f1_child', 'f1_blocks', 'f2_child', 'f2_blocks',
                       'eq']
            f_csv = csv.writer(f1)
            f_csv.writerow(headers)
            for j in range(0, len(dataset)):
                temp1 = dataset[j][2]
                for id1 in range(0, len(temp1)):
                    temp1[id1] = str(temp1[id1]).replace('.', ' ')
                #print(dataset[j][2])
                temp2 = dataset[j][4]
                for id2 in range(0, len(temp2)):
                    temp2[id2] = str(temp2[id2]).replace('.', ' ')
                row = [  #
                    [dataset[j][0], dataset[j][1], dataset[j][3],temp1,
                      dataset[j][5],temp2, 1]
                ]


                f_csv.writerows(row)
        mrr = 0
        recall_at_k = 0
        temp_df = pd.read_csv(config.MRR_Recall_k_temp_path, index_col=0)

        
        temp_df = ins_to_index(temp_df, config)
        b1, g1, b2, g2, Y_test = test_data_process(temp_df, config)


        K = config.k  

        for idx in range(0, config.poolsize):
            b1_32 = copy.deepcopy(b1)
            g1_32 = copy.deepcopy(g1)
            b2_32 = b2
            g2_32 = g2
            Y_32 = Y_test
            for j in range(0, config.poolsize):
                b1_32[j] = b1_32[idx]
                g1_32[j] = g1_32[idx]

                Y_32[j] = j
            pred = model.predict([b1_32, g1_32, b2_32, g2_32], batch_size=100)
            #print(idx)
            # print(pred)
            pred = pred.flatten()
            #print(pred)

            loc = 0
            Max = 0
            max_loc = 0
            for index in range(0, len(pred)):
                if pred[index] > pred[idx]:
                    loc = loc + 1
                if pred[index] > Max:
                    Max = pred[index]
                    max_loc = max_loc
            rank1 = loc + 1
            mrr += 1 / rank1
            if pred[idx] == Max:
                recall_at_k += 1


        recall_at_k /= config.poolsize
        mrr /= config.poolsize
        # print(mrr)
        # print(recall_at_k)
        SIMS.append(mrr)
        Recall_AT_1.append(recall_at_k)
        # print(SIMS)
    print(TYPE1, TYPE2, 'MRR{}: '.format(config.poolsize), np.array(SIMS).mean())
    print(TYPE1, TYPE2, 'Recall@1: ', np.array(Recall_AT_1).mean())
    return np.array(SIMS).mean(),np.array(Recall_AT_1).mean()


