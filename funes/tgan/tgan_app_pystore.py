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
import pystore
import pandas as pd

# Local modules
import funes.tgan
from funes.datautils.data_loading import real_data_loading, real_xlsx_data_loading,read_data_from_pystore
from funes.tgan.misc.utils import str2bool, ARGOBJ

from models.timegan import TimeGAN
from tgan_utils import (
    timegan_trainer_vanilla,
    timegan_trainer_wgan,
    timegan_generator,
    feature_prediction,
    one_step_ahead_prediction,
)



# from src.tgan import PROJROOT

def train_eval_module(
        device,
        is_train,
        seed,
        feat_pred_no,
        max_seq_len,
        train_rate,
        gradient_type,
        emb_epochs,
        sup_epochs,
        gan_epochs,
        batch_size,
        hidden_dim,
        num_layers,
        dis_thresh,
        optimizer,
        learning_rate,
        dat_path,
        mdl_dir,
        tbd_dir,
):
    ##############################################
    # Initialize args with input parameter
    ##############################################
    args = ARGOBJ()
    args.device = device
    args.is_train = is_train
    args.seed = seed
    args.feat_pred_no = feat_pred_no
    args.max_seq_len = max_seq_len
    args.train_rate = train_rate
    args.gradient_type = gradient_type
    args.emb_epochs = emb_epochs
    args.sup_epochs = sup_epochs
    args.gan_epochs = gan_epochs
    args.batch_size = batch_size
    args.hidden_dim = hidden_dim
    args.num_layers = num_layers
    args.dis_thresh = dis_thresh
    args.optimizer = optimizer
    args.learning_rate = learning_rate
    args.dat_path = dat_path
    args.mdl_dir = mdl_dir
    args.tbd_dir = tbd_dir

    ##############################################
    # Initialize random seed and CUDA
    ##############################################
    # Set random seed
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        print("Using CUDA\n")
        args.device = torch.device("cuda:0")
        # torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Using CPU\n")
        args.device = torch.device("cpu")

    #########################
    # Load and preprocess data for model
    #########################

    # data_path = "data/oridata3.csv"
    # X, T, _, args.max_seq_len, args.padding_value = data_preprocess(
    #     data_path, args.max_seq_len
    # )
    if args.gradient_type == "vanilla":
        X, T = real_xlsx_data_loading(args.dat_path + "/HMZABAAH6MF0144842022-07-07-11-44-16", args.max_seq_len, ["soc", "ocv"])
    else:  # wgan or wgan_gp:
        X, T = real_xlsx_data_loading(args.dat_path + "/HMZABAAH6MF0144842022-07-07-11-44-16", args.max_seq_len, ["soc", "ocv"])

    print(T.shape)
    print(f"Processed data: {X.shape} (Idx x MaxSeqLen x Features)\n")
    # print(f"Original data preview:\n{X[:2, :10, :2]}\n")

    args.feature_dim = X.shape[-1]
    args.Z_dim = X.shape[-1]
    args.padding_value = 0
    # Train-Test Split data and time
    train_data, test_data, train_time, test_time = train_test_split(
        X, T, test_size=args.train_rate, random_state=args.seed
    )

    #########################
    # Initialize and Run model
    #########################

    # Log start time
    start = time.time()
    model = TimeGAN(args)
    if args.is_train == True:
        if args.gradient_type == "vanilla":
            timegan_trainer_vanilla(model, train_data, train_time, args)
        else:  # args.gradient_type == "wasserstein" or "wasserstein_gp"
            timegan_trainer_wgan(model, train_data, train_time, args)
    try:
        generated_data = timegan_generator(model, train_time, args)
    except:
        raise ValueError("Generated data is empty.")
    generated_time = train_time

    # Log end time
    end = time.time()

    # print(f"Generated data preview:\n{generated_data[:2, -10:, :2]}\n")
    print(f"Model Runtime: {(end - start) / 60} mins\n")

    #########################
    # Save train and generated data for visualization
    #########################

    # Save splitted data and generated data
    with open(f"{args.mdl_dir}/train_data.pickle", "wb") as fb:
        pickle.dump(train_data, fb)
    with open(f"{args.mdl_dir}/train_time.pickle", "wb") as fb:
        pickle.dump(train_time, fb)
    with open(f"{args.mdl_dir}/test_data.pickle", "wb") as fb:
        pickle.dump(test_data, fb)
    with open(f"{args.mdl_dir}/test_time.pickle", "wb") as fb:
        pickle.dump(test_time, fb)
    with open(f"{args.mdl_dir}/fake_data.pickle", "wb") as fb:
        pickle.dump(generated_data, fb)
    with open(f"{args.mdl_dir}/fake_time.pickle", "wb") as fb:
        pickle.dump(generated_time, fb)

    #########################
    # Preprocess data for seeker
    #########################

    # Define enlarge data and its labels
    enlarge_data = np.concatenate((train_data, test_data), axis=0)
    enlarge_time = np.concatenate((train_time, test_time), axis=0)
    enlarge_data_label = np.concatenate(
        (np.ones([train_data.shape[0], 1]), np.zeros([test_data.shape[0], 1])), axis=0
    )

    # Mix the order
    idx = np.random.permutation(enlarge_data.shape[0])
    enlarge_data = enlarge_data[idx]
    enlarge_data_label = enlarge_data_label[idx]

    #########################
    # Evaluate the performance
    #########################

    # 1. Feature prediction
    feat_idx = np.random.permutation(train_data.shape[2])[: args.feat_pred_no]
    print("Running feature prediction using original data...")
    ori_feat_pred_perf = feature_prediction(
        (train_data, train_time), (test_data, test_time), feat_idx
    )
    print("Running feature prediction using generated data...")
    new_feat_pred_perf = feature_prediction(
        (generated_data, generated_time), (test_data, test_time), feat_idx
    )

    feat_pred = [ori_feat_pred_perf, new_feat_pred_perf]

    print(
        "Feature prediction results:\n"
        + f"(1) Ori: {str(np.round(ori_feat_pred_perf, 4))}\n"
        + f"(2) New: {str(np.round(new_feat_pred_perf, 4))}\n"
    )

    # 2. One step ahead prediction
    print("Running one step ahead prediction using original data...")
    ori_step_ahead_pred_perf = one_step_ahead_prediction(
        (train_data, train_time), (test_data, test_time)
    )
    print("Running one step ahead prediction using generated data...")
    new_step_ahead_pred_perf = one_step_ahead_prediction(
        (generated_data, generated_time), (test_data, test_time)
    )

    step_ahead_pred = [ori_step_ahead_pred_perf, new_step_ahead_pred_perf]

    print(
        "One step ahead prediction results:\n"
        + f"(1) Ori: {str(np.round(ori_step_ahead_pred_perf, 4))}\n"
        + f"(2) New: {str(np.round(new_step_ahead_pred_perf, 4))}\n"
    )
    return None

def train_eval_cmd(args):
    ##############################################
    # Initialize random seed and CUDA
    ##############################################

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        print("Using CUDA\n")
        args.device = torch.device("cuda:0")
        # torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Using CPU\n")
        args.device = torch.device("cpu")

    #########################
    # Load and preprocess data for model
    #########################

    store = pystore.store('train_data')
    col_path = '2022-08-09'
    # store.delete_collection('2022-08-09')
    if col_path in store.list_collections():
        collection = store.collection(col_path)
    else:
        print('collection doesn\'t exist')

    vins = []
    for item in list(collection.list_items()):
        if item != 'metadata' and item.split('_')[0] not in vins:
            vins.append(item.split('_')[0])
    X = []
    for vin in vins:
        print(f'{vin}\t{vins.index(vin) + 1}/{len(vins)}')
        for idx in range(1, 199):
            x = read_data_from_pystore(collection, vin, idx, variable_list)
            X.append(x)
    X = np.array(X)
    T = np.array([X.shape[1] for _ in range(X.shape[0])])

    # print(X.shape)
    print(T.shape,T[:3])
    print(f"Processed data: {X.shape} (Idx x MaxSeqLen x Features)\n") # (198xN, 370, 4) N:car_nums
    # print(f"Original data preview:\n{X[:2, :10, :2]}\n")

    args.feature_dim = X.shape[-1]
    args.Z_dim = X.shape[-1]
    args.padding_value = 0
    # Train-Test Split data and time
    train_data, test_data, train_time, test_time = train_test_split(
        X, T, train_size=args.train_rate, random_state=args.seed
    )
    print(f"train data: {train_data.shape} test data: {test_data.shape}\n")
    #########################
    # Initialize and Run model
    #########################

    # Log start time
    start = time.time()
    model = TimeGAN(args)
    if args.is_train == True:
        if args.gradient_type == "vanilla":
            timegan_trainer_vanilla(model, train_data, train_time, args)
        else:  # args.gradient_type == "wasserstein" or "wasserstein_gp"
            timegan_trainer_wgan(model, train_data, train_time, args)
    try:
        generated_data = timegan_generator(model, train_time, args)
    except:
        raise ValueError("Generated data is empty.")
    generated_time = train_time

    # Log end time
    end = time.time()

    # print(f"Generated data preview:\n{generated_data[:2, -10:, :2]}\n")
    print(f"Model Runtime: {(end - start) / 60} mins\n")

    #########################
    # Save train and generated data for visualization
    #########################

    # Save splitted data and generated data
    with open(f"{args.mdl_dir}/train_data.pickle", "wb") as fb:
        pickle.dump(train_data, fb)
    with open(f"{args.mdl_dir}/train_time.pickle", "wb") as fb:
        pickle.dump(train_time, fb)
    with open(f"{args.mdl_dir}/test_data.pickle", "wb") as fb:
        pickle.dump(test_data, fb)
    with open(f"{args.mdl_dir}/test_time.pickle", "wb") as fb:
        pickle.dump(test_time, fb)
    with open(f"{args.mdl_dir}/fake_data.pickle", "wb") as fb:
        pickle.dump(generated_data, fb)
    with open(f"{args.mdl_dir}/fake_time.pickle", "wb") as fb:
        pickle.dump(generated_time, fb)

    #########################
    # Preprocess data for seeker
    #########################

    # Define enlarge data and its labels
    enlarge_data = np.concatenate((train_data, test_data), axis=0)
    enlarge_time = np.concatenate((train_time, test_time), axis=0)
    enlarge_data_label = np.concatenate(
        (np.ones([train_data.shape[0], 1]), np.zeros([test_data.shape[0], 1])), axis=0
    )

    # Mix the order
    idx = np.random.permutation(enlarge_data.shape[0])
    enlarge_data = enlarge_data[idx]
    enlarge_data_label = enlarge_data_label[idx]

    #########################
    # Evaluate the performance
    #########################

    # 1. Feature prediction
    feat_idx = np.random.permutation(train_data.shape[2])[: args.feat_pred_no]
    print("Running feature prediction using original data...")
    ori_feat_pred_perf = feature_prediction(
        (train_data, train_time), (test_data, test_time), feat_idx
    )
    print("Running feature prediction using generated data...")
    new_feat_pred_perf = feature_prediction(
        (generated_data, generated_time), (test_data, test_time), feat_idx
    )

    feat_pred = [ori_feat_pred_perf, new_feat_pred_perf]

    print(
        "Feature prediction results:\n"
        + f"(1) Ori: {str(np.round(ori_feat_pred_perf, 4))}\n"
        + f"(2) New: {str(np.round(new_feat_pred_perf, 4))}\n"
    )

    # 2. One step ahead prediction
    print("Running one step ahead prediction using original data...")
    ori_step_ahead_pred_perf = one_step_ahead_prediction(
        (train_data, train_time), (test_data, test_time)
    )
    print("Running one step ahead prediction using generated data...")
    new_step_ahead_pred_perf = one_step_ahead_prediction(
        (generated_data, generated_time), (test_data, test_time)
    )

    step_ahead_pred = [ori_step_ahead_pred_perf, new_step_ahead_pred_perf]

    print(
        "One step ahead prediction results:\n"
        + f"(1) Ori: {str(np.round(ori_step_ahead_pred_perf, 4))}\n"
        + f"(2) New: {str(np.round(new_step_ahead_pred_perf, 4))}\n"
    )
    return None


if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser()

    # Experiment Arguments
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", type=str)
    parser.add_argument("--is_train", type=str2bool, default=True)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--feat_pred_no", default=1, type=int)

    # Data Arguments
    parser.add_argument("--max_seq_len", default=370, type=int)
    parser.add_argument("--train_rate", default=0.8, type=float)

    # Model Arguments
    parser.add_argument("--gradient_type", default="vanilla", type=str)
    parser.add_argument("--emb_epochs", default=2, type=int)
    parser.add_argument("--sup_epochs", default=2, type=int)
    parser.add_argument("--gan_epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--hidden_dim", default=24, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--dis_thresh", default=0.15, type=float)
    parser.add_argument("--optimizer", choices=["adam"], default="adam", type=str)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--last_emb_epochs", default=200, type=int)
    parser.add_argument("--last_sup_epochs", default=1000, type=int)
    parser.add_argument("--last_gan_epochs", default=1000, type=int)
    parser.add_argument("--lr_factor", default=0.99, type=float)
    parser.add_argument("--lr_patience", default=500, type=float)
    parser.add_argument("--lr_min", default=1e-10, type=float)

    # Output subfolder Arguments, should give a time stamp as subfolder name
    parser.add_argument("--out_subdir", default="now", type=str)
    # Runtime directory
    # Data directory
    parser.add_argument("--dat_path", default="now", type=str)

    # variable_list = ["soc", "ocv"]
    variable_list = ["soc"]

    args = parser.parse_args()
    ts_now = '-'.join([i.upper() for i in variable_list]) + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.out_subdir = ts_now+'xxx'

    # Input data directory

    PROJROOT = str(Path(__file__).parent.parent.parent)
    args.dat_path = PROJROOT + "/data/train-data/"  # os.path.abspath("../../data")
    # print(f"PROJROOT:{PROJROOT}")
    print(f"Data path in tgan_app: {args.dat_path}")

    if not os.path.exists(args.dat_path):
        raise ValueError(f"Data path not found at {args.dat_path}.")

    # Output directories
    out_dir = funes.tgan.PROJROOT + "/data/output/" + args.out_subdir
    try:
        os.makedirs(out_dir)
    except FileExistsError:
        print("Output directory already exists. Resume...")


    ##############################################
    # Initialize output directories
    ##############################################
    hyperparam_setting_exp = (
            '-'.join([i.upper() for i in variable_list])
            + "-"
            + args.gradient_type
            + "-"
            + str(args.sup_epochs)
            + "-"
            + str(args.emb_epochs)
            + "-"
            + str(args.gan_epochs)
            + "-"
            + str(args.batch_size)
            + "-"
            + str(args.hidden_dim)
            + "-"
            + str(args.num_layers)
            + "-"
            + str(args.learning_rate)
            + "-"
            + str(args.max_seq_len)
            + "-"
            + str(4000)
    )

    # model directory
    args.mdl_dir = out_dir + "/mdl" + f"/{hyperparam_setting_exp}"
    try:
        os.makedirs(args.mdl_dir, exist_ok=False)
    except FileExistsError:
        print("Model directory already exists. Resume...")

    # TensorBoard directory
    args.tbd_dir = out_dir + "/tbd" + f"/{hyperparam_setting_exp}"
    try:
        os.makedirs(args.tbd_dir, exist_ok=False)
    except FileExistsError:
        print("TensorBoard directory already exists. Resume...")

    start = time.time()
    # Call main function
    train_eval_cmd(args)
    print(f'train_eval_cmd: {(time.time()-start)/60}mins')
