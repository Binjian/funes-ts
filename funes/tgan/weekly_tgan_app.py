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

def load_data(store,collection,max_len=100,min_len=None,padding=True,soc_threshold=98):
    # variable_list = ['soc']
    # max_len = 100
    # store = pystore.store('train_data')
    # col_path = '2022-08-09'  # Time: 0.8754101514816284  24552
    col_path = collection
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
    T = []
    print('vins:',len(vins))
    for vin in vins:
        print(f'{vin}\t{vins.index(vin) + 1}/{len(vins)}')
        for idx in range(1, 199):
            df = collection.item(vin + '_cell_' + str(idx)).to_pandas()
            soc = df['soc'].to_list()

            start = 0
            # end = 0
            # split data with soc [80-100]
            for i in range(len(soc) - 1):
                end = i + 1

                if soc[i + 1] < soc[i]:
                    x = df[['vol', 'current', 'soc']].iloc[start:end].values
                    t = len(soc[start:end])

                    start = i + 1
                    if df['soc'].iloc[end-1] >= soc_threshold:
                        if t <= max_len:
                            if padding:
                                x = np.pad(x, ((0, max_len - t), (0, 0)), 'constant', constant_values=(0, 0))

                            X.append(x)
                            T.append(t)
                        else:
                            # rand_idx = sorted(random.sample(range(0, t), max_len))
                            rand_idx = sorted(random.sample(range(0, t - 20), max_len - 20))
                            X.append(np.concatenate((x[rand_idx], x[-20:]), axis=0))
                            T.append(max_len)

                elif end == len(soc) - 1:
                    x = df[['vol', 'current', 'soc']].iloc[start:end + 1].values
                    t = len(soc[start:end + 1])

                    if df['soc'].iloc[end - 1] >= soc_threshold:
                        if t <= max_len:
                            if padding:
                                x = np.pad(x, ((0, max_len - t), (0, 0)), 'constant', constant_values=(0, 0))

                            X.append(x)
                            T.append(t)
                        else:
                            # rand_idx = sorted(random.sample(range(0, t), max_len))
                            rand_idx = sorted(random.sample(range(0, t - 20), max_len - 20))
                            X.append(np.concatenate((x[rand_idx], x[-20:]), axis=0))
                            T.append(max_len)
    if min_len:
        T = np.array(T)
        idx = np.where(T > min_len)
        if padding:
            X = np.array(X)[idx]
            T = np.array(T)[idx]
        else:
            # X = np.array(X)
            X = np.array(X, dtype='object')[idx]
            T = np.array(T)[idx]
    else:
        if padding:
            X = np.array(X)
            T = np.array(T)
        else:
            X = np.array(X, dtype='object')
            T = np.array(T)
    print(X.shape)
    print(T.shape)

    return X,T

def data_preprocess(X,T):

    clean_X = []
    clean_T = []
    for j in range(len(X)):
        data = True
        v = X[j,:T[j],0]
        c = X[j,:T[j],1]
        count = 0
        start = 0

        for i in range(len(v)-1):
            if v[i] >= v[i + 1]:
                count += 1

            if count > 25:
                data = False
                break
        if data:
            for k in range(len(c)-1):
                if c[k+1] < c[k] or c[k]>=0:
                    continue
                else:
                    start = k
                    break
            x = np.pad(X[j,start:,:], ((0, start), (0, 0)), 'constant', constant_values=(0, 0))

            clean_X.append(x)
            clean_T.append(T[j]-start)

    return np.array(clean_X),np.array(clean_T)


def MinMaxScaler(data):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    numerator = data - np.min(data)

    # print(np.min(data, 1)[:,0],np.max(data, 1)[:,1])
    denominator = np.max(data) - np.min(data)

    norm_data = numerator / (denominator + 1e-7)
    return norm_data

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
    col_list = [sorted(store.list_collections())[-1]]
    X = np.array([])
    T = np.array([])
    for col_path in col_list:
        print(col_path)
        try:
            if col_list.index(col_path)==0:
                X,T = load_data(store,col_path,max_len=100,soc_threshold=100)
            else:
                x,t = load_data(store,col_path,max_len=100,soc_threshold=100)
                X = np.concatenate((X, x),axis=0)
                T = np.concatenate((T, t), axis=0)
        except Exception as e:
            print(e)
            continue

    # col_path = datetime.datetime.now().strftime("%Y-%m-%d")
    # col_path = '2022-10-21'
    # X ,T = load_data(store,col_path,max_len=100,soc_threshold=100)
    print('before preprocess:',X.shape,T.shape)
    X,T = data_preprocess(X,T)
    # print(X.shape)

    print(X[0, :, 0])
    X = MinMaxScaler(X)

    X = np.expand_dims(X[:, :, 0], axis=-1)  # vol


    print(T[:3])

    # print(X.shape)
    print(T.shape)
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
    # print(feat_idx)
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
    parser.add_argument("--max_seq_len", default=100, type=int)
    parser.add_argument("--train_rate", default=0.8, type=float)

    # Model Arguments
    parser.add_argument("--gradient_type", default="vanilla", type=str)
    parser.add_argument("--emb_epochs", default=1200, type=int)
    parser.add_argument("--sup_epochs", default=1200, type=int)
    parser.add_argument("--gan_epochs", default=1200, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--hidden_dim", default=24, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--dis_thresh", default=0.15, type=float)
    parser.add_argument("--optimizer", choices=["adam"], default="adam", type=str)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--last_emb_epochs", default=200, type=int)
    parser.add_argument("--last_sup_epochs", default=1000, type=int)
    parser.add_argument("--last_gan_epochs", default=1000, type=int)

    # Output subfolder Arguments, should give a time stamp as subfolder name
    parser.add_argument("--out_subdir", default="now", type=str)
    # Runtime directory
    # Data directory
    parser.add_argument("--dat_path", default="now", type=str)

    # variable_list = ["soc", "ocv"]
    variable_list = ["vol"]

    args = parser.parse_args()
    ts_now = '-'.join([i.upper() for i in variable_list]) + '-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.out_subdir = ts_now

    # Input data directory

    PROJROOT = str(Path(__file__).parent.parent.parent)
    args.dat_path = PROJROOT + "/data/train-data/"  # os.path.abspath("../../data")
    # print(f"PROJROOT:{PROJROOT}")
    print(f"Data path in tgan_app: {args.dat_path}")

    if not os.path.exists(args.dat_path):
        raise ValueError(f"Data path not found at {args.dat_path}.")

    # Output directories
    out_dir = funes.tgan.PROJROOT + "/data/output/vol_weekly_results/" + args.out_subdir
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
