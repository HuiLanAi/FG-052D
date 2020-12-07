import argparse
import pickle
from tqdm import tqdm
import sys
import numpy as np
import os





if __name__ == '__main__':
    parser.add_argument('--data_path', default='../data/ntu/xview/val_data.npy')

    data = np.load('../data/ntu/xview/val_data.npy')