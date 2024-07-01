import tensorflow as tf
import sys
from keras.utils.vis_utils import plot_model
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from DL.MSSA_network import MSSA_nework
import DL.config as config
import warnings
import argparse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0',
                    help='visible gpu device')
parser.add_argument('--train', type=bool, default=False,
                    help='train or not')
parser.add_argument('--test', type=bool, default=False,
                    help='test or not')
parser.add_argument('--mrr', type=bool, default=False,
                    help='test or not')
parser.add_argument('--w2v_dim', type=int, default=config.w2v_dim,
                    help='word embedding dimension')
parser.add_argument('--batch_size', type=int, default=config.batch_size,
                    help='bath_size are given to the model for training')
parser.add_argument('--max_block_seq', type=int, default=config.max_block_seq,
                    help='Number of instructions contained in the basic block')
parser.add_argument('--num_block', type=int, default=config.num_block,
                    help='Number of instructions contained in the basic block')
parser.add_argument('--iter_level', type=int, default=config.iter_level,
                    help='iteration times')
parser.add_argument('--epoch', type=int, default=config.epochs,
                    help='epoch number')
parser.add_argument('--poolsize', type=int, default=config.poolsize,
                    help='the datasets number to evaluate MMR and Recall@K')
parser.add_argument('--k', type=int, default=config.k,
                    help='top K')
parser.add_argument('--patience', type=int, default=config.patience,
                    help='max epoch to wait loos decrease')
parser.add_argument('--save_freq', type=int, default=config.save_freq,
                    help='gap epoch to save weights')
parser.add_argument('--load_path', type=str, default=config.model_load_weights,
                    help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--save_path', type=str,
                    default=config.model_save_weights, help='path for model saving')
parser.add_argument('--log_path', type=str, default=config.model_log_path,
                    help='path for log')

if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = tf.float32
    print("=================================")
    print(args)
    print("=================================")
    print("GPU is available: ", tf.test.is_gpu_available())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    config.w2v_dim = args.w2v_dim
    config.batch_size = args.batch_size
    config.max_block_seq = args.max_block_seq
    config.num_block = args.num_block
    config.iter_level = args.iter_level
    config.epochs = args.epoch
    config.poolsize = args.poolsize
    config.k = args.k
    config.patience = args.patience
    config.save_freq = args.save_freq
    config.model_load_weights = args.load_path
    config.model_save_weights = args.save_path
    config.model_log_path = args.log_path

    # make model class
    MSSA_net = MSSA_nework(config)
    # build model
    MSSA_net.build_model()

    MSSA_net.model.summary()

    plot_model(MSSA_net.model, to_file=config.model_log_path + 'model.png', show_shapes=True)

    if args.train == True and args.test == True:
        # train and test
        MSSA_net.train()
        MSSA_net.test()
    elif args.train:
        # only train
        MSSA_net.train()
    elif args.test:
        # only test
        MSSA_net.test()
    if args.mrr == True:
        MSSA_net.test_MRR_Recall_k_all()

