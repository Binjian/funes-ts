import pandas as pd
import openpyxl
import numpy as np
import time
import os

import pystore
import matplotlib.pyplot as plt
import random



def real_xlsx_data_loading(data_path, seq_len,variable_list,sheet_num):
    """
    Load and preprocess real-world datasets.

    Args:
      - data_name: filename
      - seq_len: sequence length
      - variable_list: selected_variables

    Returns:
      - data: preprocessed data.
    """
    # read file
    # data_file_path = str(data_path) + ".xlsx"
    data_file_path = str(data_path)
    df = pd.read_excel(data_file_path,sheet_name='Cell_'+str(sheet_num))
    # df = df.append(df)
    # print(len(df))
    # header
    desired_variable_list = list(df.head(0).columns)
    # print(desired_variable_list)
    variable_num = 2 + len(variable_list)
    # new df to store data
    new_df = pd.DataFrame()
    new_df["vol"] = df["vol"]
    new_df["current"] = df["current"]

    # check if variable exists
    for element in variable_list:
        if element in desired_variable_list:
            # print("All variable is defined in original data-set")
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

# df = pd.read_excel('../../data/train-data/2022-08-22/HMZABAAHXMF0182772022-08-22-16-20-48.xlsx',sheet_name='Cell_1')
# data,t = real_xlsx_data_loading('../../data/train-data/2022-08-22/HMZABAAHXMF0182772022-08-22-16-20-48.xlsx',seq_len=370,variable_list=['soc'],sheet_num=1)
# print(t)
# print(df[['vol','current','soc']][:5].values)
# print(data[0][:5])


s = time.time()

# file_list = ['HMZABAAHXMF0182772022-08-22-16-20-48.xlsx','HMZABAAH1MF0182952022-08-22-16-17-30.xlsx',
#              'HMZABAAH6MF0182892022-08-22-16-21-09.xlsx','HMZABAAH8MF0182762022-08-22-16-31-47.xlsx']

def load_data_from_excel():
    data_path = '../../data/train-data/2022-08-22' # Time: 21.670892854531605  24354
    file_list = sorted(os.listdir(data_path))

    variable_list = ['soc']
    X = []
    T = []
    for file in file_list:
        # print('###############################')
        print(f'{file}\t{file_list.index(file)+1}/{len(file_list)}')
        # print('###############################')
        for idx in range(1,199):
            df = pd.read_excel('../../data/train-data/2022-08-22/'+file,sheet_name='Cell_'+str(idx))
            # print(df.shape[0])

            soc = df['soc'].to_list()
            # soc[-1] = 80.0
            # print(len(soc),soc)

            start = 0
            # end = 0
            for i in range(len(soc)-1):
                end = i+1
                # print(start, end, soc[start], soc[end])
                if soc[i + 1] < soc[i]:
                    x = df[['vol','current','soc']].iloc[start:end].values
                    t = len(soc[start:end])
                    x = np.pad(x, ((0, 370 - t), (0, 0)), 'constant', constant_values=(0, 0))

                    # print(len(soc[start:end]),soc[start:end])
                    # print(start,end)
                    start = i+1
                    X.append(x)
                    T.append(t)
                    # print(start,soc[start])
                elif end == len(soc) - 1:
                    x = df[['vol','current','soc']].iloc[start:end+1].values
                    t = len(soc[start:end+1])
                    x = np.pad(x, ((0, 370 - t), (0, 0)), 'constant', constant_values=(0, 0))
                    # print(len(soc[start:end+1]), soc[start:end+1])
                    X.append(x)
                    T.append(t)

    # print(len(X),X)
    print(len(T),T)
    print(np.array(X).shape)
    print(np.array(T).shape,T)
    # print(X[0][:,2])

    X = np.array(X)
    T = np.array(T)
    # np.save('X_0822_split.npy',X)
    # np.save('T_0822_split.npy',T)


def load_data_from_pystore():
    variable_list = ['soc']
    max_len = 100
    store = pystore.store('train_data')
    col_path = '2022-08-09'  # Time: 0.8754101514816284  24552
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
            # df = read_data_from_pystore(collection, vin, idx, variable_list)
            df = collection.item(vin + '_cell_' + str(idx)).to_pandas()

            soc = df['soc'].to_list()

            start = 0
            # end = 0
            for i in range(len(soc) - 1):
                end = i + 1
                # print(start, end, soc[start], soc[end])
                if soc[i + 1] < soc[i]:
                    x = df[['vol', 'current', 'soc']].iloc[start:end].values
                    t = len(soc[start:end])
                    x = np.pad(x, ((0, 370 - t), (0, 0)), 'constant', constant_values=(0, 0))

                    X.append(x)
                    T.append(t)

                    # print(len(soc[start:end]),soc[start:end])
                    # print(start,end)
                    start = i + 1
                    # if df['soc'].iloc[end-1] >= 98 and t <= 100:
                    #     # x = np.pad(x, ((0, max_len - t), (0, 0)), 'constant', constant_values=(0, 0))
                    #     # print(df['soc'].iloc[end])
                    #     X.append(x)
                    #     T.append(t)
                    # print(start,soc[start])
                elif end == len(soc) - 1:
                    x = df[['vol', 'current', 'soc']].iloc[start:end + 1].values
                    t = len(soc[start:end + 1])
                    x = np.pad(x, ((0, 370 - t), (0, 0)), 'constant', constant_values=(0, 0))
                    X.append(x)
                    T.append(t)

                    # print(len(soc[start:end+1]), soc[start:end+1])

                    # if df['soc'].iloc[end - 1] >= 98 and t <= 100:
                    #     # x = np.pad(x, ((0, max_len - t), (0, 0)), 'constant', constant_values=(0, 0))
                    #     # print(df['soc'].iloc[end - 1])
                    #     X.append(x)
                    #     T.append(t)

    X = np.array(X)
    T = np.array(T)
    print(X[:3])
    print(X.shape,T.shape)

# load_data_from_pystore()

def load_data(collection,max_len=100,min_len=None,padding=True):
    variable_list = ['soc']
    # max_len = 100
    store = pystore.store('train_data')
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
            # df = read_data_from_pystore(collection, vin, idx, variable_list)
            df = collection.item(vin + '_cell_' + str(idx)).to_pandas()

            soc = df['soc'].to_list()

            start = 0
            # end = 0
            # split data with soc [80-100]
            for i in range(len(soc) - 1):
                end = i + 1
                # print(start, end, soc[start], soc[end])
                if soc[i + 1] < soc[i]:
                    x = df[['vol', 'current', 'soc']].iloc[start:end].values
                    t = len(soc[start:end])
                    # x = np.pad(x, ((0, 370 - t), (0, 0)), 'constant', constant_values=(0, 0))

                    # print(len(soc[start:end]),soc[start:end])
                    # print(start,end)
                    start = i + 1
                    if df['soc'].iloc[end-1] >= 98:
                        if t <= max_len:
                            if padding:
                                x = np.pad(x, ((0, max_len - t), (0, 0)), 'constant', constant_values=(0, 0))
                                # print(x)
                            # print(df['soc'].iloc[end])
                            X.append(x)
                            T.append(t)
                        else:

                            # rand_idx = sorted(random.sample(range(0, t), max_len))
                            rand_idx = sorted(random.sample(range(0, t - 20), max_len - 20))
                            X.append(np.concatenate((x[rand_idx], x[-20:]), axis=0))

                            # X.append(x[t-max_len:])
                            # X.append(x[rand_idx])

                            T.append(max_len)


                elif end == len(soc) - 1:
                    x = df[['vol', 'current', 'soc']].iloc[start:end + 1].values
                    t = len(soc[start:end + 1])
                    # x = np.pad(x, ((0, 370 - t), (0, 0)), 'constant', constant_values=(0, 0))

                    # print(len(soc[start:end+1]), soc[start:end+1])

                    if df['soc'].iloc[end - 1] >= 98:
                        if t <= max_len:
                            if padding:
                                x = np.pad(x, ((0, max_len - t), (0, 0)), 'constant', constant_values=(0, 0))
                                # print(x)
                            # print(df['soc'].iloc[end - 1])
                            X.append(x)
                            T.append(t)
                        else:
                            # rand_idx = sorted(random.sample(range(0, t), max_len))
                            rand_idx = sorted(random.sample(range(0, t - 20), max_len - 20))
                            X.append(np.concatenate((x[rand_idx], x[-20:]), axis=0))

                            # X.append(x[t - max_len:])
                            # X.append(x[rand_idx])

                            T.append(max_len)
    if min_len:
        T = np.array(T)
        idx = np.where(T > min_len)
        print(idx)
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
    # np.save('X_es_test.npy', X)
    # np.save('T_es_test.npy', T)
    return X,T

def load_data_with_soc_threshold(collection,max_len=100,min_len=None,padding=True,soc_threshold=98):
    variable_list = ['soc']
    # max_len = 100
    store = pystore.store('train_data')
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
            # df = read_data_from_pystore(collection, vin, idx, variable_list)
            df = collection.item(vin + '_cell_' + str(idx)).to_pandas()

            soc = df['soc'].to_list()

            start = 0
            # end = 0
            # split data with soc [80-100]
            for i in range(len(soc) - 1):
                end = i + 1
                # print(start, end, soc[start], soc[end])
                if soc[i + 1] < soc[i]:
                    x = df[['vol', 'current', 'soc']].iloc[start:end].values
                    t = len(soc[start:end])
                    # x = np.pad(x, ((0, 370 - t), (0, 0)), 'constant', constant_values=(0, 0))

                    start = i + 1
                    if df['soc'].iloc[end-1] >= soc_threshold:

                        if t <= max_len:
                            if padding:
                                x = np.pad(x, ((0, max_len - t), (0, 0)), 'constant', constant_values=(0, 0))
                                # print(x)
                            # print(df['soc'].iloc[end])
                            X.append(x)
                            T.append(t)
                        else:

                            # rand_idx = sorted(random.sample(range(0, t), max_len))
                            rand_idx = sorted(random.sample(range(0, t - 20), max_len - 20)) # last 20 soc data is important
                            X.append(np.concatenate((x[rand_idx], x[-20:]), axis=0))

                            # X.append(x[t-max_len:])
                            # X.append(x[rand_idx])

                            T.append(max_len)


                elif end == len(soc) - 1:
                    x = df[['vol', 'current', 'soc']].iloc[start:end + 1].values
                    t = len(soc[start:end + 1])
                    # x = np.pad(x, ((0, 370 - t), (0, 0)), 'constant', constant_values=(0, 0))

                    # print(len(soc[start:end+1]), soc[start:end+1])

                    if df['soc'].iloc[end - 1] >= soc_threshold:
                        if t <= max_len:
                            if padding:
                                x = np.pad(x, ((0, max_len - t), (0, 0)), 'constant', constant_values=(0, 0))
                                # print(x)
                            # print(df['soc'].iloc[end - 1])
                            X.append(x)
                            T.append(t)
                        else:
                            # rand_idx = sorted(random.sample(range(0, t), max_len))
                            rand_idx = sorted(random.sample(range(0, t - 20), max_len - 20))
                            X.append(np.concatenate((x[rand_idx], x[-20:]), axis=0))

                            # X.append(x[t - max_len:])
                            # X.append(x[rand_idx])

                            T.append(max_len)
    if min_len:
        T = np.array(T)
        idx = np.where(T > min_len)
        print(idx)
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
    # np.save('X_es_test.npy', X)
    # np.save('T_es_test.npy', T)
    return X,T



def data_preprocess(X,T):
    # X = np.load('X_0927_split_100.npy')
    # T = np.load('T_0927_split_100.npy')
    clean_X = []
    clean_T = []
    for j in range(len(X)):
        data = True
        v = X[j,:T[j],0] # v vol
        c = X[j,:T[j],1] # c current
        count = 0
        start = 0
        for i in range(len(v)-1):
            if v[i] >= v[i + 1]:
                count += 1
            # else:
            #     count = 0
            # if v[i] - v[i + 1] > 18 or count > 25:  # 0927 (8,15)  0809 (10,10)
            if count > 25:  # 0927 (8,15)  0809 (10,10)   20 for 100len
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

    # print(len(clean_X),len(clean_T))
    # np.save('X_clean_0927.npy', np.array(clean_X))
    # np.save('T_clean_0927.npy', np.array(clean_T))
    return np.array(clean_X),np.array(clean_T)


X,T = load_data_with_soc_threshold(collection='2022-09-27',max_len=100,soc_threshold=100)
print(X[0,:,0])
print('111',X[:,:,0].shape,T.shape)
X,T = data_preprocess(X,T)
print(X.shape,T.shape)

'''
111 (6732, 100) (6732,)
100    4950
22      198
23      198
27      198
34      198
43      198
49      198
51      198
79      198
82      198
(1047, 100, 3) (1047,) 20     1493 =>25

111 (6732, 50) (6732,)
50    5544
22     198
23     198
27     198
34     198
43     198
49     198
(5266, 50, 3) (5266,) 20

'''



# def data_preprocess2():
#     X1 = np.load('X_0809_split_100_3.npy')
#     T1 = np.load('T_0809_split_100_3.npy')
#     X2 = np.load('X_0927_split_100.npy')
#     T2 = np.load('T_0927_split_100.npy')
#
#     # print(len(X))
#     clean_X = []
#     clean_T = []
#     for j in range(len(X2)):
#         data = True
#         v = X2[j,:T2[j],0]
#         # print(len(X[i,:,0]),len(v),T[i])
#     # a = np.array([1,2,3,1,5,9,4,3,5,10,2,7,11])
#     #     print(v[-2:])
#         count = 0
#         for i in range(len(v)-1):
#             if v[i] >= v[i+1]:
#                 count += 1
#             # else:
#             #     count = 0
#             if v[i] - v[i+1] > 8 or count > 15:  # 0927 (8,15)  0809 (10,10)
#                 data = False
#                 break
#         if data:
#             clean_X.append(X2[j])
#             clean_T.append(T2[j])
#             # print(T[j])
#
#     for a in range(len(X1)):
#         data = True
#         v = X1[a,:T1[a],0]
#         # print(len(X[i,:,0]),len(v),T[i])
#     # a = np.array([1,2,3,1,5,9,4,3,5,10,2,7,11])
#     #     print(v[-2:])
#         count = 0
#         for b in range(len(v)-1):
#             if v[b] >= v[b+1]:
#                 count += 1
#             else:
#                 count = 0
#             if v[b] - v[b+1] > 10 or count > 9:  # 0927 (8,15)  0809 (10,10)
#                 data = False
#                 break
#         if data:
#             clean_X.append(X1[a])
#             clean_T.append(T1[a])
#     # np.save('X_clean_100.npy',np.array(clean_X))
#     # np.save('T_clean_100.npy', np.array(clean_T))
#
#     # return np.array(clean_X),np.array(clean_T)

# data_preprocess2()
# print('###',len(x),len(y))

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


# def MinMaxScaler(data):
#     """Min Max normalizer.
#
#     Args:
#       - data: original data
#
#     Returns:
#       - norm_data: normalized data
#     """
#     # numerator = data - np.min(data, 1)
#     numerator = []
#     for i in range(data.shape[1]):
#         numerator.append(data[:,i,:]-np.min(data, 1))
#     numerator = np.array(numerator)
#     numerator = np.transpose(numerator,(1,0,2))
#     print(numerator.shape)
#
#     # print(np.min(data, 1)[:,0],np.max(data, 1)[:,1])
#     de = np.max(data, 1) - np.min(data, 1)
#     de = np.expand_dims(de,axis=1)
#     denominator = de
#     # for i in range(100):
#     #     np.concatenate((denominator,de),axis=1)
#     print(denominator.shape)
#     # print(denominator[:,0])
#     norm_data = numerator / (denominator + 1e-7)
#     return norm_data


'''
X_0809_split_100_2 random 100 idx
X_0809_split_100 [t-100:]
X_0809_split_100_3  rand_idx = sorted(random.sample(range(0, t-20), max_len-20))  X.append(x[rand_idx]+x[-20:])
'''

# X = np.load('X_0809_split_100_3.npy')
# T = np.load('T_0809_split_100_3.npy')
# X = np.load('X_0927_split_100.npy')
# T = np.load('T_0927_split_100.npy')
# X = np.load('X_clean_100.npy')
# T = np.load('T_clean_100.npy')
# X = np.load('X_clean_0927.npy')
# T = np.load('T_clean_0927.npy')

# X,T = data_preprocess(X,T)
# print(len(X),len(T))

# X_norm1 = MinMaxScaler(X)

# print(X.shape)
print(pd.DataFrame(list(T)).value_counts())


# np.random.seed(2)
# np.random.shuffle(X_norm1)
#
# np.random.seed(2)
# np.random.shuffle(T)
#
# np.random.seed(2)
# np.random.shuffle(X)




def show():
    pic_num = 25
    # figure, ax = plt.subplots()
    # figure.suptitle('soc data')
    # for j in range(1, pic_num + 1):
    #     plt.subplot(5, 5, j)
    #     plt.ylim([min(X[j, :T[j], 2]), max(X[j, :T[j], 2])])
    #     plt.plot(X[j, :T[j], -1], label='soc data')
    #     # print(max(X[j, :, 2]))
    # plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # plt.show()

    figure, ax = plt.subplots()
    figure.suptitle('origin vol data')
    for j in range(1, pic_num + 1):
        plt.subplot(5, 5, j)
        plt.ylim([min(X[j, :T[j], 0]), max(X[j, :T[j], 0])])
        # plt.xlim([0, 100])
        # idx = random.randint(1,10494)
        # print(idx)
        # print(X[idx, -5:, -1])
        plt.plot(X[j, :T[j], 0], label='vol data')
        # print(X[j, :T[j], 0],X[j, :T[j], 2])
        # plt.plot(X[j], label='vol data')
        # print(X[j,:,-1])
        # plt.plot(X[idx, :, 0], label='vol data')
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # plt.show()

    # figure, ax = plt.subplots()
    # figure.suptitle('norm1 vol data')
    # for j in range(1, pic_num + 1):
    #     plt.subplot(5, 5, j)
    #     plt.ylim([min(X_norm1[j, :T[j], 0]), max(X_norm1[j, :T[j], 0])])
    #     # plt.xlim([0, 100])
    #     # idx = random.randint(1,10494)
    #     # print(idx)
    #     # print(X[idx, -5:, -1])
    #     plt.plot(X_norm1[j, :T[j], 0], label='vol data')
    #     # plt.plot(X_norm[j], label='vol data')
    #     # print(X_norm[j,:,-1])
    #     # plt.plot(X[idx, :, 0], label='vol data')
    # plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # plt.show()

    # figure, ax = plt.subplots()
    # figure.suptitle('current data')
    # for j in range(1, pic_num + 1):
    #     plt.subplot(5, 5, j)
    #     plt.ylim([min(X[j, :T[j], 1]), max(X[j, :T[j], 1])])
    #     plt.plot(X[j, :T[j], 1], label='current data')
    #     # print(X[j, :T[j], 1])
    # plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.show()

# show()
for i in range(10):
    # np.random.seed(1)
    # np.random.shuffle(X_norm1)

    np.random.seed(2)
    np.random.shuffle(T)

    np.random.seed(2)
    np.random.shuffle(X)
    show()

print('Time:',(time.time()-s)/60)




