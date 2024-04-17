# system packages
import pickle
from collections import OrderedDict
import os
# third-party packages
import matplotlib.pyplot as plt
import numpy as np
import torch
from models.timegan import EmbeddingNetwork, DiscriminatorNetwork
import pandas as pd
import logging
import datetime
import time
import subprocess
import openpyxl
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pystore
import random

THRESHOLD = 0.5
CELLNUM = 198

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


def load_data(collection,vin,idx,max_len=100,min_len=None,padding=True):
    X = []
    T = []
    # variable_list = ['soc']
    # # max_len = 100
    # store = pystore.store('train_data')
    # # col_path = '2022-08-09'  # Time: 0.8754101514816284  24552
    # col_path = collection
    # # store.delete_collection('2022-08-09')
    # if col_path in store.list_collections():
    #     collection = store.collection(col_path)
    # else:
    #     print('collection doesn\'t exist')
    #
    # vins = []
    # for item in list(collection.list_items()):
    #     if item != 'metadata' and item.split('_')[0] not in vins:
    #         vins.append(item.split('_')[0])
    # X = []
    # T = []
    # print('vins:',len(vins))
    # for vin in vins:
    #     print(f'{vin}\t{vins.index(vin) + 1}/{len(vins)}')
    #     for idx in range(1, 199):

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
            if df['soc'].iloc[end - 1] >= 98:
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

            if df['soc'].iloc[end - 1] >= 98:
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
    # print(X.shape)
    # print(T.shape)

    return X,T


def read_data_from_pystore(collection,vin,idx,desired_var):

    df = collection.item(vin+'_cell_'+str(idx)).to_pandas()
    variables = list(df)
    new_df = pd.DataFrame()
    new_df["vol"] = df["vol"]
    new_df["current"] = df["current"]

    for element in desired_var:
        if element in variables:
            # print("All variable is defined in original data-set")
            new_df[element] = df[element]
        else:
            print(str(element)+ " not exists")
            print("the avaliable variables are",desired_var)

    X = [new_df]
    X = np.array(X)
    # print(X.shape)  # (9702, 370, 4)
    T = np.array([X.shape[1] for _ in range(X.shape[0])])
    # print(T[:3])
    # print(T.shape)  # (9702,)
    return X,T

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



def main():

    ##### logger
    # logger = logging.getLogger("inference")
    # formatter = logging.Formatter(
    # "%(asctime)s-%(levelname)s-%(module)s-%(threadName)s-%(funcName)s)-%(lineno)d): %(message)s"
    # )
    # logfilename = (
    # "../../data/logs/battery-inference-"
    # + datetime.datetime.now().strftime("%y-%m-%d-%h-%m-%s_%f")[:-3]
    # + ".log")
    #
    # fh = logging.FileHandler(logfilename)
    # fh.setLevel(logging.DEBUG)
    # fh.setFormatter(formatter)
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    # ch.setFormatter(formatter)
    # logger.addHandler(fh)
    # logger.addHandler(ch)
    # logger.setLevel(logging.DEBUG)


    ##### data fetch

    logger.info(f"data scraping start!")
    ## TODO wait till data-fetch finish
    p = subprocess.Popen(['python', '../datautils/data_scrap_pystore.py'], stdin = subprocess.PIPE, stdout=subprocess.PIPE)
    out = p.stdout.readlines()
    print(out)

    ##### inference
    # define saved output path
    path = "../../data/output/vol_0928/VOL-20221004-044620/mdl/VOL-vanilla-1200-1200-1200-128-24-3-0.001-100-4000"
    if not os.path.exists(path):
        os.mkdir(path)

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
    # if data is saved as xlsx
    # total_warn_rate = 0
    # train_file_path = '../../data/train-data/'+datetime.datetime.now().strftime("%Y-%m-%d")+'/' # inference with recent data
    store_path = 'train_data'

    # col_path= '2022-10-18'
    store = pystore.store(store_path)
    # col_path = datetime.datetime.now().strftime("%Y-%m-%d")
    col_path= sorted(store.list_collections())[-1] # latest collection
    collection = store.collection(col_path)
    print(f'inference with {col_path} data')
    # if col_path in store.list_collections():
    #     collection = store.collection(col_path)
    # else:
    #     print('collection doesn\'t exist')

    vins = []

    for item in list(collection.list_items()):
        if item != 'metadata' and item.split('_')[0] not in vins:
            vins.append(item.split('_')[0])
    print(vins)
    # vins = list(set([j[:17] for j in [i.split('_')[0] for i in list(collection.list_items())
    #                                   if i != 'metadata']]))
    logger.info(f"original battery data loaded successful")
    logger.info(f"inference start")
    for vin in vins:
    #     total_warn_rate = 0
        warn_cells = []
        warn_list = []
        logit = []
        X = np.array([])
        T = np.array([])
        for idx in range(1,199):
            print(f'Loading {vin} Cell_{idx}')
            # X, T = real_data_loading(train_file_path+file, args.max_seq_len,["soc",'ocv'],idx)
            # x,t = read_data_from_pystore(collection,vin,idx,["soc",'ocv'])
            x,t = load_data(collection,vin,idx,max_len=100)
            if idx==1:
                X = x
                T = t
            else:
                X = np.concatenate((X,x),axis=0)
                T = np.concatenate((T,t),axis=0)
        if X.shape[0] != 198:
            logger.info(f'{vin} {col_path} data error! X.shape {X.shape}, T.shape {T.shape}')
            continue

        X = np.expand_dims(MinMaxScaler(X)[:, :, 0], axis=-1)

        print(X.shape,T.shape)

        # convert data into torch tensor
        X = torch.from_numpy(X)
        T = torch.from_numpy(T)
        # embedding
        H_comb = embedder(X.float(), T.float()).detach()
        # discriminate data
        Y_comb = discriminator(H_comb, T)

        # logger.info(f"original battery data loaded successful")
        # logger.info(f"inference start")
        # store logit by sigmoid and mean
        # logit = []
        for i in range(Y_comb.shape[0]):
            logit.append(float(Y_comb[i][-1].sigmoid()))
        # create warn list
        # warn_list = []
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
        # plt.show()

        # summary
        # total_warn_rate += len(warn_list)
        # if len(warn_list) != 0:
        #     warn_cells.append(idx)
            # logger.info(f"The warn rate of this data set is {warn_rate}")
        # print(len(warn_list))
        # print(total_warn_rate)
        logger.info(f"The warn rate of {vin} is {len(warn_list) / CELLNUM}")
        logger.info(f"The specific warn cell numbers of {vin} are {warn_list}")

        # ES
        # server = ['http://10.0.64.64:32222']
        # # server = ['10.0.64.64']
        # port = 32222
        # login_info = ('elastic', '123456')
        #
        # ## es=Elasticsearch(server,port=port,http_auth=login_info,timeout=30,retry_on_timeout=True,max_retries=10)
        # es = Elasticsearch(hosts=server, basic_auth=login_info, request_timeout=30, retry_on_timeout=True,
        #                    max_retries=10)
        # print(es.info())
        # index_name = 'funes_results'
        # if es.indices.exists(index=index_name):
        #     pass
        # else:
        #     es.indices.create(index=index_name)
        # # print(es.indices.get(index=index_name))
        #
        # es_id = vin + '-' + str(int(time.time()))
        # if len(warn_cells) == 0:
        #     state = 'normal'
        # else:
        #     state = 'abnormal'
        #
        # # write
        # #  {'altitude': 536.0, 'fileName': 'newrizon-20211222-5.dbc', 'gpsDirection': 0.0, 'travelTime': 1652170875010, 'lastActiveTime': 1}
        # dataList = [
        #     {'_index': index_name, '_id': es_id, '_source': {'date': datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        #                                                      'vin': vin, 'Warning_cellNum': warn_cells,
        #                                                      'state': state}}]
        #
        # helpers.bulk(es, dataList, index=index_name, raise_on_error=True)
        # print(dataList)
        # print(len(es.search(index=index_name)['hits']['hits']))
        # # print(es.search(index=index_name))
        # print(es.search(index=index_name)['hits']['hits'])
        print(f'{vin} inference done.')



#{'_index':funes_results,'_id':car+timestamp,
# '_source':{'date':datetime.datetime.now(),'carNum':001,'cellNum':001,'state':normal/abnormal}}

if __name__ == "__main__":
    # Call main function
    main()
