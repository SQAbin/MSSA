import os

current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# word2vec
window = 5
vector_size = 100  # 100
min_count = 0
workers = 4
sample = 1e-5
w2v_epoch = 100  # 100

w2v_load_path = project_root + "/w2v/"
w2v_save_path = project_root + "/w2v/"
corpus_path = project_root + "/datasets/corpus_functions_asm.csv"

# datasets_path
path_train_json = project_root + '/datasets' + '/dataset_train.json'
path_test_json = project_root + '/datasets' + '/dataset_test.json'


MRR_Recall_k_temp_path = project_root + "/datasets/temp.csv"
temp_path = project_root + "/datasets/Temp.csv"
train_csv_path = project_root + "/datasets/small_max_train_sets.csv"
test_csv_path = project_root + "/datasets/small_max_test_sets.csv"


# MSSA model  key-parameter
w2v_dim = 100
batch_size = 512
max_block_seq = 20
max_seq_length=max_block_seq+1
num_block = 20
iter_level = 5
f_dim = 128
ww1 = 128
epochs = 100
patience = 6

save_freq = 2
# MRR and Recall@k
poolsize = 32
k = 1

# model path
model_save_weights = project_root + "/DL/result/model_weights/"
model_load_weights = project_root + "/DL/result/model_weights/"
model_log_path = project_root + "/DL/result/"
