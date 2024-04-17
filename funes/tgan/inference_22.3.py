# system packages
import pickle
from collections import OrderedDict

# third-party packages
import numpy as np
import torch
from models.timegan import EmbeddingNetwork, DiscriminatorNetwork
import matplotlib.pyplot as plt
import pandas as pd
import logging
import datetime
import subprocess

THRESHOLD = 0.5
CELLNUM = 198

def real_data_loading(data_path, seq_len,variable_list):
    """Load and preprocess real-world datasets.

    Args:
      - data_name: filename
      - seq_len: sequence length
      - variable_list: selected_variables

    Returns:
      - data: preprocessed data.
    """
    # read file
    data_file_path = str(data_path) + ".csv"
    df = pd.read_csv(data_file_path)
    # header
    desired_variable_list = list(df.head(0).columns)
    variable_num = 2 + len(variable_list)
    # new df to store data
    new_df = pd.DataFrame()
    new_df["vol"] = df["vol"]
    new_df["current"] = df["current"]

    # check if variable exists
    for element in variable_list:
        if element in desired_variable_list:
            print("All variable is defined in original data-set")
            new_df[element] = df[element]

        else:
            print(str(element)+ " not exists")
            print("the avaliable variables are",desired_variable_list)

    # number of samples
    sample_no = int(df.shape[0]/seq_len)
    temp_data = []
    for i in range(0, sample_no):
        _x = new_df.iloc[i * seq_len : i * seq_len + seq_len]
        temp_data.append(_x)
    idx = np.random.permutation(len(temp_data))
    data = []
    time = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
        time.append(np.count_nonzero(~np.isnan(temp_data[idx[i]])) / variable_num)
    data = np.array(data)
    time = np.array(time)

    return data,time




def main():

    ##### logger
    logger = logging.getLogger("inference")
    formatter = logging.Formatter(
    "%(asctime)s-%(levelname)s-%(module)s-%(threadName)s-%(funcName)s)-%(lineno)d): %(message)s"
    )
    logfilename = (
    "../../data/logs/battery-inference-"
    + datetime.datetime.now().strftime("%y-%m-%d-%h-%m-%s_%f")[:-3]
    + ".log")

    fh = logging.FileHandler(logfilename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)


    ##### data fetch

    logger.info(f"data scraping start!")
    ## TODO wait till data-fetch finish
    p = subprocess.Popen(['python', '../datautils/data_scrap_2022-3.py'], stdin = subprocess.PIPE, stdout=subprocess.PIPE)
    out = p.stdout.readlines()
    print(out)

    ##### inference
    # define saved output path
    path = "../../data/output/test"
    # load model args
    if f"{path}/args.pickle":
        with open(f"{path}/args.pickle", "rb") as fb:
            args = torch.load(fb)
            logger.info(f"model argument loaded successful")
    else:
        logger.info(f"No model argument path found")
    # initialize embedder and discriminator
    discriminator = DiscriminatorNetwork(args)
    embedder = EmbeddingNetwork(args)
    dis_dict = OrderedDict()
    emb_dict = OrderedDict()
    # load timegan model
    model = torch.load(f"{path}/model.pt")
    # fetch state dict of embedder and that of discriminator in timegan model
    for key, value in model.items():
        # print(key)
        if "discriminator" in key:
            dis_dict[key.replace("discriminator.", "")] = value
        if "embedder" in key:
            emb_dict[key.replace("embedder.", "")] = value
    # load state dict into embedder and discriminator
    discriminator.load_state_dict(dis_dict)
    discriminator.eval()
    embedder.load_state_dict(emb_dict)
    embedder.eval()
    logger.info(f"model loaded successful")
    # if data is saved as csv
    X, T = real_data_loading("../../data/train-data/ocv_HMZABAAH9MF014494_412_512", args.max_seq_len,["soc"])
    # convert data into torch tensor
    X = torch.from_numpy(X)
    T = torch.from_numpy(T)
    # embedding
    H_comb = embedder(X.float(), T.float()).detach()
    # discriminate data
    Y_comb = discriminator(H_comb, T)
    logger.info(f"original battery data loaded successful")
    logger.info(f"inference start")
    # store logit by sigmoid and mean
    logit = []
    for i in range(Y_comb.shape[0]):
        logit.append(float(Y_comb[i][-1].sigmoid()))
    # create warn list
    warn_list = []
    # plot data logits
    plt.figure()
    for i in range(len(logit)):
        if logit[i] < THRESHOLD:
            plt.scatter(i, logit[i], color="orange", label="Warning Data")
            warn_list.append(i)
        else:
            plt.scatter(i, logit[i], color="blue", label="Original Data")
    plt.xlabel("Sample no.")
    plt.ylabel("Logits")
    plt.title("Data Logits")
    logger.info(f"inference result produced")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()

    # summary
    warn_rate = len(warn_list)/CELLNUM
    logger.info(f"The warn rate of this data set is {warn_rate}")
    logger.info(f"The specific warn cell numbers are {warn_list}")


if __name__ == "__main__":
    # Call main function
    main()
