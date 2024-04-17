# system module
import os
import pickle
import random
import time, datetime
from pathlib import Path

# 3rd-Party Modules
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Local modules
import funes.tgan
from funes.datautils.data_loading import real_data_loading, real_xlsx_data_loading
from funes.tgan.misc.utils import str2bool, ARGOBJ

from models.timegan import TimeGAN
from tgan_utils import (
    timegan_trainer_vanilla,
    timegan_trainer_wgan,
    timegan_generator,
    feature_prediction,
    one_step_ahead_prediction,
)

def get_data():
    ##############################################
    # Initialize random seed and CUDA
    ##############################################

    os.environ["PYTHONHASHSEED"] = str(42)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.device == "cuda" and torch.cuda.is_available():
        print("Using CUDA\n")
        torch.device = torch.device("cuda:0")
        # torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Using CPU\n")
        torch.device = torch.device("cpu")

    #########################
    # Load and preprocess data for model
    #########################

    # data_path = "data/oridata3.csv"
    # X, T, _, args.max_seq_len, args.padding_value = data_preprocess(
    #     data_path, args.max_seq_len
    # )

    data_path = '../../data/train-data/2022-08-22/'
    file_list = sorted(os.listdir(data_path))
    # print(file_list[30])

    # variable_list = ["soc", "ocv"]

    X = []
    T = []
    for file in file_list:
        print(f'{file}\t{file_list.index(file)+1}/{len(file_list)}')
        try:
            for idx in range(1,199):
                x, t = real_xlsx_data_loading(data_path+file, 370,
                                              ['soc', 'ocv'], idx)
                X.extend(x)
                T.append(t[0])
                # print('11111111111111',X)
                # print('2222222222222',type(t))
        except Exception as e:
            print(e)
            continue
    X = np.array(X)
    T = np.array(T)

    # print(X.shape)
    print(T.shape)
    print(f"Processed data: {X.shape} (Idx x MaxSeqLen x Features)\n") # (198xN, 370, 4)
    np.save('X_data_0822.npy',X)
    np.save('T_data_0822.npy', T)

import pystore
from funes.datautils.data_loading import read_data_from_pystore

def get_data_pystore():
    store = pystore.store('train_data')
    print(store.list_collections())
    collection = store.collection('2022-09-27')
    # print(collection.items)
    print(len(collection.list_items()))

    vins = []
    for item in list(collection.list_items()):
        if item != 'metadata' and item.split('_')[0] not in vins:
            vins.append(item.split('_')[0])
    X = []
    for vin in vins:
        print(f'{vin}\t{vins.index(vin) + 1}/{len(vins)}')
        for idx in range(1, 199):
            x = read_data_from_pystore(collection, vin, idx, ['soc'])
            X.append(x)
    X = np.array(X)
    T = np.array([X.shape[1] for _ in range(X.shape[0])])

    # print(X.shape)
    print(T.shape, T[:3])
    print(f"Processed data: {X.shape} (Idx x MaxSeqLen x Features)\n")  # (198xN, 370, 4) N:car_nums

# get_data_pystore()




