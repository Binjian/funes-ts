"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""

## Necessary Packages
import numpy as np
import pandas as pd
import openpyxl

def MinMaxScaler(data):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def sine_data_generation(no, seq_len, dim):
    """Sine data generation.

    Args:
      - no: the number of samples
      - seq_len: sequence length of the time-series
      - dim: feature dimensions

    Returns:
      - data: generated data
    """
    # Initialize the output
    data = list()

    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)

    return data

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

def read_stock_data(data_path, seq_len,variable_list):
    # read file
    data_file_path = str(data_path)
    df = pd.read_csv(data_file_path)
    # header
    desired_variable_list = list(df.head(0).columns)
    # variable_num = 2 + len(variable_list)
    variable_num = len(variable_list)

    # number of samples
    sample_no = int(df.shape[0] / seq_len)
    temp_data = []
    i = 0
    j = seq_len
    while j <= len(df):
        _x = df.iloc[i:j]
        temp_data.append(_x)
        i += 1
        j += 1

    # for i in range(0, sample_no):
    #     # _x = new_df.iloc[i * seq_len: i * seq_len + seq_len]
    #     _x = df.iloc[i * seq_len: i * seq_len + seq_len]
    #     temp_data.append(_x)
    idx = np.random.permutation(len(temp_data))
    data = []
    time = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
        time.append(np.count_nonzero(~np.isnan(temp_data[idx[i]])) / variable_num)
    data = np.array(data)
    time = np.array(time)

    return data, time

def read_data_from_pystore(collection, vin, idx, desired_var):
    df = collection.item(vin + '_cell_' + str(idx)).to_pandas()
    variables = list(df)
    new_df = pd.DataFrame()
    new_df["vol"] = df["vol"]
    new_df["current"] = df["current"]

    for element in desired_var:
        if element in variables:
            # print("All variable is defined in original data-set")
            new_df[element] = df[element]
        else:
            print(str(element) + " not exists")
            print("the avaliable variables are", desired_var)

    return new_df


if __name__ == "__main__":
    pass

