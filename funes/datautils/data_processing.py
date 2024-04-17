#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Cut segment to defined length
def cutSegment(size, data, segmt):
    segdata = pd.DataFrame()
    length_list = []
    for i in range(len(segmt) - 1):
        length = segmt[i + 1] - segmt[i]
        length_list.append(length)
    max_length = max(length_list)
    for i in range(len(segmt) - 1):
        length = segmt[i + 1] - segmt[i]
        part = pd.DataFrame()
        if length < max_length and length > size:
            n = max_length - length
            part = data.iloc[segmt[i] : (segmt[i + 1]), :]
            part = part.reset_index(drop=True)
            inits = pd.DataFrame(columns=data.columns)
            # insert first value as default to fill the sequence
            for a in range(n):
                inits.loc[a] = float("nan")
            part = pd.concat([part, inits], axis=0, ignore_index=True)
            segdata = segdata.append(part, ignore_index=True)
    print(max_length)
    return segdata


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


def real_data_loading(data_name, seq_len):
    """Load and preprocess real-world datasets.

    Args:
      - data_name: stock or energy
      - seq_len: sequence length

    Returns:
      - data: preprocessed data.
    """

    ori_data = np.loadtxt(str(data_name) + ".csv", delimiter=",", skiprows=1)

    # Flip the data to make chronological data
    # ori_data = ori_data[::-1]
    # Normalize the data
    ori_data = MinMaxScaler(ori_data)
    # Preprocess the dataset
    temp_data = []
    time = []
    # Cut data by sequence length
    for i in range(0, int(len(ori_data) / seq_len)):
        _x = ori_data[i * seq_len : i * seq_len + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
        time.append(seq_len)
    data = np.array(data)
    time = np.array(time)
    return data, time


def main():
    # read xlsx file with # time, current, soc and vol1-198
    wb = pd.read_excel(
        "filt-data/NewCar1.xlsx",
        sheet_name="HMZABAAH9MF014494",
    )
    # store data to data frame
    df = pd.DataFrame(wb)

    ## data processing
    # create new data frame to store processed data
    centerdf = pd.DataFrame()
    # store time, current and soc to data frame
    centerdf["time"] = df.iloc[:, 0]
    centerdf["current"] = df.iloc[:, 1]
    centerdf["soc"] = df.iloc[:, 2]
    # calculate mean of cell voltage
    mean = np.mean(df.iloc[:, 3:], axis=1)
    # store difference between mean of cell voltage for each cell into data frame
    for i in range((int(df.shape[1] - 3))):
        centerdf["vol" + str(i + 1)] = df.iloc[:, i + 3] - mean

    centerdf = centerdf.dropna(axis=0, how="any")
    # visualize
    print(centerdf.shape)

    # separate data by time
    seg = [0]
    for i in range(0, (int(df.shape[0]) - 1)):
        if datetime.strptime(
            df.iloc[i + 1, 0], "%Y-%m-%d %H:%M:%S"
        ) - datetime.strptime(df.iloc[i, 0], "%Y-%m-%d %H:%M:%S") >= timedelta(
            seconds=300
        ):
            seg.append(i + 1)
    seg.append(len(df))
    # insert first value as default to fill the segment
    segdf = cutSegment(370, centerdf, seg)
    segdf.shape
    # if NaN print True

    # save processed data into csv format for further training
    vol = []
    cur = []
    vol_num = []
    soc = []

    for i in range(198):
        vol = vol + segdf.iloc[:, i + 3].values.tolist()
        cur = cur + segdf.iloc[:, 1].values.tolist()
        soc = soc + segdf.iloc[:, 2].values.tolist()

    tsdf = pd.DataFrame()
    tsdf["current"] = np.float64(cur)
    tsdf["vol"] = np.float64(vol)
    tsdf["soc"] = np.float64(soc)
    # tsdf.to_csv("car1-soc.csv",index=False)
    tsdf.to_csv("data-soc-nan.csv", mode="a", index=False, header=None, na_rep="nan")

    test_data = real_data_loading("train-data/car1-soc", 370)
    test_data[0].shape

    ori_data, T = real_data_loading("oridata-soc", 370)
    pic_num = 16
    index = 1
    index = index * pic_num
    # for i in range(1,pic_num+1):
    #     plt.subplot(4,4,i)
    #     plt.plot(ori_data[i+index,:,1])

    plt.figure()
    for i in range(1, pic_num + 1):
        plt.subplot(4, 4, i)
        plt.plot(ori_data[i + index, :, 2])


if __name__ == "__main__":
    main()
