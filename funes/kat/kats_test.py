
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystore
import random
from kats.tsfeatures.tsfeatures import TsFeatures
from kats.consts import TimeSeriesData

import warnings
warnings.filterwarnings('ignore')

import time

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


store = pystore.store('train_data')
print(store.list_collections())

collection = store.collection('2022-09-27')


FEATURES = ["trend_strength","seasonality_strength","spikiness","peak","trough","level_shift_idx","level_shift_size","y_acf1",
        "y_acf5","diff1y_acf1","diff1y_acf5","diff2y_acf1","diff2y_acf5","y_pacf5","diff1y_pacf5","diff2y_pacf5","seas_acf1",
        "seas_pacf1","firstmin_ac","firstzero_ac","holt_alpha","holt_beta","hw_alpha","hw_beta","hw_gamma","length","mean",
        "var","entropy","lumpiness","stability","flat_spots","hurst","std1st_der","crossing_points","binarize_mean","unitroot_kpss",
        "heterogeneity","histogram_mode","linearity","cusum_num","cusum_conf","cusum_cp_index","cusum_delta","cusum_llr",
        "cusum_regression_detected","cusum_stable_changepoint","cusum_p_value","robust_num","robust_metric_mean","bocp_num","bocp_conf_max",
        "bocp_conf_mean","trend_num","trend_num_increasing","trend_avg_abs_tau","nowcast_roc","nowcast_ma","nowcast_mom",
        "nowcast_lag","nowcast_macd","nowcast_macdsign","nowcast_macddiff","seasonal_period","trend_mag","seasonality_mag","residual_std",
        "time_years","time_months","time_monthsofyear","time_weeks","time_weeksofyear","time_days","time_daysofyear","time_avg_timezone_offset",
        "time_length_days","time_freq_Monday","time_freq_Tuesday","time_freq_Wednesday","time_freq_Thursday","time_freq_Friday","time_freq_Saturday",
        "time_freq_Sunday",]  # "outlier_num"


''' 
hw_params_features  WARNING:root:Holt-Winters failed shapes (2,10) and (0,1) not aligned: 10 (dim 1) != 0 (dim 0)
level_shift_features  ERROR:root:Length of time series is shorter than window_size, unable to calculate level shift features
WARNING:root:MACDsign couldn't get computed successfully due to insufficient time series length: 21
WARNING:root:MACDsign couldn't get computed successfully due to insufficient time series length: 22
WARNING:root:MACDsign couldn't get computed successfully due to insufficient time series length: 23
WARNING:root:MACDsign couldn't get computed successfully due to insufficient time series length: 24

acfpacf_features ERROR:root:Length is shorter than period, or constant time series, unable to calculate acf/pacf features
'''

vins = []  # 16,17,24,55
for item in list(collection.list_items()):
    if item != 'metadata' and item.split('_')[0] not in vins:
        vins.append(item.split('_')[0])
vins = sorted(vins)
print(len(vins))
START = time.time()
all_data = []
all_feature = []

for vin in vins:
    print(f'Loading {vin} {vins.index(vin)}/{len(vins)}')
    for idx in range(1,199):
        # print(f'Loading {vin} Cell_{idx}')
        cell = collection.item(vin + '_cell_' + str(idx)).to_pandas()

        df = pd.DataFrame()
        df['time'] = cell.timestamp.values
        df['vol'] = cell.vol.values
        df['soc'] = cell.soc.values

        X = []
        T = []
        soc = df['soc'].to_list()
        max_len = 100
        padding = True

        start = 0

        for i in range(len(soc) - 1):
            end = i + 1

            if soc[i + 1] < soc[i]:
                x = df.iloc[start:end].values
                t = len(soc[start:end])

                start = i + 1
                if df['soc'].iloc[end - 1] >= 98:
                    if t <= max_len:
                        # if padding:
                        #     x = np.pad(x, ((0, max_len - t), (0, 0)), 'constant', constant_values=(0, 0))

                        X.append(x)
                        T.append(t)
                    else:
                        # rand_idx = sorted(random.sample(range(0, t), max_len))
                        rand_idx = sorted(random.sample(range(0, t - 20), max_len - 20))
                        X.append(np.concatenate((x[rand_idx], x[-20:]), axis=0))
                        T.append(max_len)

            elif end == len(soc) - 1:
                x = df.iloc[start:end + 1].values
                t = len(soc[start:end + 1])

                if df['soc'].iloc[end - 1] >= 98:
                    if t <= max_len:
                        # if padding:
                        #     x = np.pad(x, ((0, max_len - t), (0, 0)), 'constant', constant_values=(0, 0))

                        X.append(x)
                        T.append(t)
                    else:
                        # rand_idx = sorted(random.sample(range(0, t), max_len))
                        rand_idx = sorted(random.sample(range(0, t - 20), max_len - 20))
                        X.append(np.concatenate((x[rand_idx], x[-20:]), axis=0))
                        T.append(max_len)
        # print('xxx',X[0].shape,np.array(X).shape)
        ts_model = TsFeatures(selected_features=FEATURES)

        # air = pd.read_csv('air_passengers.csv')
        # air.columns = ["time", "value"]
        # print(air.time)
        # print(air.value)
        # print(air)
        # print(type(air.time),type(air.value),type(air))
        # print(type(air['time'].iloc[0]))
        # ts = TimeSeriesData(air)
        # print(ts)
        for x in X:
            # print(x.shape)
            df_ = pd.DataFrame()
            t = [str(i).split()[0] for i in x[:,0]]
            df_['time'] = t
            df_['value'] = list(x[:,1])

            if len(df_) > 24 and len(df_['value'].unique())>=3: # 15,24
                ts = TimeSeriesData(df_)
                # print(type(ts.time))
                features = ts_model.transform(ts)

                # print(ts.value.unique(),len(ts.value),len(ts.value.unique()))

                # print(features)
                # print(len(features.values()))
                # print(features)
                all_data.append(x[:,1])
                all_feature.append(np.array(list(features.values())))
# print(all_data[:3])
# print(all_feature[:3])
print(len(all_data),len(all_feature))
# print(all_feature)
length = [len(i) for i in all_feature]

# print(pd.DataFrame(length).value_counts())

pic_num = 25
figure, ax = plt.subplots()
figure.suptitle('Most similar cases')
figure.set_size_inches(9, 6)
figure.subplots_adjust(right=0.85)

all_data = np.array(all_data)
all_feature = np.array(all_feature)

np.random.seed(1)
np.random.shuffle(all_data)
np.random.seed(1)
np.random.shuffle(all_feature)

all_data_train = all_data[:int(0.8*len(all_data))]
all_data_test = all_data[int(0.8*len(all_data)):]
all_feature_train = all_feature[:int(0.8*len(all_feature))]
all_feature_test = all_feature[int(0.8*len(all_feature)):]

for i in range(1,pic_num+1):
    mae_list = []
    for j in range(len(all_feature_train)):
        if j != i:
            diff = np.mean(abs(all_feature_test[i] - all_feature_train[j]))
            mae_list.append(diff)
        else:
            mae_list.append(1e+8)

    plt.subplot(5, 5, i)
    min_y = min(min(all_data_test[i]),
                min(all_data_train[mae_list.index(min(mae_list))]))
    max_y = max(max(all_data_test[i]),
                max(all_data_train[mae_list.index(min(mae_list))]))
    plt.ylim([min_y, max_y])
    plt.plot(all_data_train[mae_list.index(min(mae_list))],
             label='most similar', linewidth='2.5')
    plt.plot(all_data_test[i], label='origin data')
END = time.time()
print(f'{(END-START)/60} mins')
plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
plt.show()


'''
18612 data
21.29600567817688 mins
60 cars
'''



