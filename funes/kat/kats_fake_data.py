
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystore
import random
from kats.tsfeatures.tsfeatures import TsFeatures
from kats.consts import TimeSeriesData
import argparse
import warnings
warnings.filterwarnings('ignore')
import torch
import time
from funes.tgan.models.timegan import TimeGAN

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


# FEATURES = ["trend_strength","seasonality_strength","spikiness","peak","trough","level_shift_idx","level_shift_size","y_acf1",
#         "y_acf5","diff1y_acf1","diff1y_acf5","diff2y_acf1","diff2y_acf5","y_pacf5","diff1y_pacf5","diff2y_pacf5","seas_acf1",
#         "seas_pacf1","firstmin_ac","firstzero_ac","holt_alpha","holt_beta","hw_alpha","hw_beta","hw_gamma","length","mean",
#         "var","entropy","lumpiness","stability","flat_spots","hurst","std1st_der","crossing_points","binarize_mean","unitroot_kpss",
#         "heterogeneity","histogram_mode","linearity","cusum_num","cusum_conf","cusum_cp_index","cusum_delta","cusum_llr",
#         "cusum_regression_detected","cusum_stable_changepoint","cusum_p_value","robust_num","robust_metric_mean","bocp_num","bocp_conf_max",
#         "bocp_conf_mean","trend_num","trend_num_increasing","trend_avg_abs_tau","nowcast_roc","nowcast_ma","nowcast_mom",
#         "nowcast_lag","nowcast_macd","nowcast_macdsign","nowcast_macddiff","seasonal_period","trend_mag","seasonality_mag","residual_std",
#         "time_years","time_months","time_monthsofyear","time_weeks","time_weeksofyear","time_days","time_daysofyear","time_avg_timezone_offset",
#         "time_length_days","time_freq_Monday","time_freq_Tuesday","time_freq_Wednesday","time_freq_Thursday","time_freq_Friday","time_freq_Saturday",
#         "time_freq_Sunday",]  # "outlier_num"

FEATURES = ["hw_alpha","hw_beta","hw_gamma","length","mean",
        "var","entropy","lumpiness","stability","flat_spots","hurst","std1st_der","crossing_points","binarize_mean","unitroot_kpss",
        "heterogeneity","histogram_mode","linearity"
       ]
print(len(FEATURES))

'''
sine features remove:
time
seasonalities
nowcasting
trend detectors
bocp detector
robust stat detector ** defect

stl_features
level_shift_feature
acfpacf_features * defect
special_ac *** defect
holt_params


'''

# data_path = '../../data/output/vol_0928/VOL-20221004-044620/mdl/VOL-vanilla-1200-1200-1200-128-24-3-0.001-100-4000'
data_path = '/data/output/Sine_1013/Sine-20221013-105200_best/mdl/Sine-vanilla-1000-1000-1000-64-32-3-0.001-50-4000'


def gen_fake(args):
    train_time = pd.read_pickle(f'{data_path}/train_time.pickle')
    train_data = pd.read_pickle(f'{data_path}/train_data.pickle')
    # Ts = pd.read_pickle(f"{data_path}/train_time.pickle")
    # print(type(Ts))
    # print(pd.DataFrame(train_time).value_counts())
    # print(len(set(train_time)))
    if len(set(train_time)) != 1:
        Ts = pd.DataFrame(train_time).value_counts().items()
        T = []
        for val, count in Ts:
            T.append(val[0])

        print(T[:int(0.3 * len(T))], 0.3 * len(T))
        T = T[:int(0.3 * len(T))]
        # args.feature_dim = 5
        # args.Z_dim = 5
        # args.max_seq_len = 50
        # args.padding_value = 0
        T = np.array([np.random.choice(T) for _ in range(train_data.shape[0])])
    else:
        T = np.array([train_time[0] for _ in range(train_time.shape[0])])
    T = torch.from_numpy(T)

    model = TimeGAN(args)
    model.load_state_dict(torch.load(f"{data_path}/model.pt"))

    model.to(args.device)
    model.eval()
    with torch.no_grad():
        # Generate fake data
        torch.manual_seed(1)
        Z = torch.rand((len(T), args.max_seq_len, args.Z_dim))

        # Z_ = []
        # for i in range(len(T)):
        #     tmp_t = list(np.ones(T[i])) + list(np.zeros(args.max_seq_len-T[i]))
        #     r = torch.squeeze(Z[i])*torch.tensor(tmp_t)
        #     Z_.append(r.unsqueeze(axis=-1).numpy())
        # Z_ = torch.from_numpy(np.array(Z_)).float()

        generated_data = model(X=None, T=T, Z=Z, obj="inference")
    fake_time = T.numpy()

    return generated_data.numpy(), train_data, fake_time, train_time


params = data_path.split('/')[-1].split('-')

parser = argparse.ArgumentParser()

parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", type=str)
parser.add_argument("--gradient_type", default="vanilla", type=str)
parser.add_argument("--batch_size", default=params[5], type=int)
parser.add_argument("--hidden_dim", default=params[6], type=int)
parser.add_argument("--num_layers", default=params[7], type=int)
parser.add_argument("--feature_dim", default=1, type=int)
parser.add_argument("--Z_dim", default=1, type=int)
parser.add_argument("--max_seq_len", default=params[9], type=int)
parser.add_argument("--padding_value", default=0, type=float)

args = parser.parse_args()

fake_data, train_data, fake_time, train_time = gen_fake(args)

def kats_similar():
    START = time.time()
    train_no_padding = []
    fake_no_padding = []
    for i in range(fake_data.shape[0]):
        fake_no_padding.append(fake_data[i][:fake_time[i]])

    for i in range(train_data.shape[0]):
        train_no_padding.append(train_data[i][:train_time[i]])

    print(len(train_no_padding),len(fake_no_padding))

    # train_no_padding = train_no_padding[:2000]

    ts_model = TsFeatures(selected_features=FEATURES)
    # time_stamp = int(time.time())
    # for i in range(5,0,-1):
    #     ts1 = time_stamp-2*i
    #     ts2 = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(ts1))
    #     print(ts1,ts2)
    train_feature = []

    for i in range(len(train_no_padding)):
        time_stamps = []
        ts_now = int(time.time())
        for j in range(len(train_no_padding[i]),0,-1): # make timestamp
            ts1 = ts_now-2*i
            ts2 = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(ts1))
            time_stamps.append(ts2)
        df = pd.DataFrame()
        df['time'] = time_stamps
        df['value'] = train_no_padding[i]

        if len(df) > 24 and len(df['value'].unique())>=3:
            ts = TimeSeriesData(df)
            # print(type(ts.time))
            features = ts_model.transform(ts)
            # print(features)
            train_feature.append(np.array(list(features.values())))

        else:
            train_feature.append(np.array([]))
    print(len(train_feature[0]))

    fake_feature = []
    fake_sample_num = 100
    for i in range(fake_sample_num):
        time_stamps = []
        ts_now = int(time.time())
        for j in range(len(fake_no_padding[i]), 0, -1):
            ts1 = ts_now - 2 * i
            ts2 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts1))
            time_stamps.append(ts2)
        df = pd.DataFrame()
        df['time'] = time_stamps
        df['value'] = fake_no_padding[i]

        if len(df) > 24 and len(df['value'].unique()) >= 3:
            ts = TimeSeriesData(df)

            features = ts_model.transform(ts)

            fake_feature.append(np.array(list(features.values())))
    print(len(train_feature),len(fake_feature))
    pic_num = 25
    figure, ax = plt.subplots()
    figure.suptitle('Most similar cases')
    figure.set_size_inches(9, 6)
    figure.subplots_adjust(right=0.85)


    for i in range(1,pic_num+1):
        mae_list = []
        for j in range(len(train_no_padding)):
            if len(train_feature[j]) != 0:
                diff = np.mean(abs(fake_feature[i] - train_feature[j]))
                mae_list.append(diff)
            else:
                mae_list.append(1e8)

        plt.subplot(5, 5, i)
        min_y = min(min(fake_no_padding[i]),
                    min(train_no_padding[mae_list.index(min(mae_list))]))
        max_y = max(max(fake_no_padding[i]),
                    max(train_no_padding[mae_list.index(min(mae_list))]))
        plt.ylim([min_y, max_y])
        plt.plot(train_no_padding[mae_list.index(min(mae_list))],
                 label='most similar', linewidth='2.5')
        plt.plot(fake_no_padding[i], label='fake data')
    END = time.time()
    print(f'{(END-START)/60} mins')
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # plt.show()
kats_similar()

def dis_similar():
    pic_num = 25

    # gen_data_all = fft_new_data
    figure, ax = plt.subplots()
    figure.suptitle('Most similar cases')
    figure.set_size_inches(9, 6)
    figure.subplots_adjust(right=0.85)

    for j in range(1, pic_num + 1):
        # gen_data_case = gen_data_all[j + index]
        mae_list = []
        idx_list = []
        for i in range(train_data.shape[0]):
            if fake_time[j] == train_time[i]:
                # ori_data_case = np.array(fft_ori_data[i])
                diff = np.mean(abs(train_data[i, :train_time[i], 0] - fake_data[j, :fake_time[j], 0]))
                mae_list.append(diff)
                idx_list.append([0, fake_time[j]])
            elif fake_time[j] < train_time[i]:
                left = 0
                right = fake_time[j]
                diff = []
                while right <= train_time[i]:
                    # print('##',left,right,fake_time[j],len(train_data[i, left:right, 0]),len(fake_data[j, :fake_time[j], 0]))
                    d = np.mean(abs(train_data[i, left:right, 0] - fake_data[j, :fake_time[j], 0]))
                    diff.append(d)
                    left += 1
                    right += 1
                mae_list.append(min(diff))
                idx = diff.index(min(diff))
                idx_list.append([idx, idx + fake_time[j]])
            elif fake_time[j] > train_time[i]:
                left = 0
                right = train_time[i]
                diff = []
                while right <= fake_time[j]:
                    d = np.mean(abs(train_data[i, :train_time[i], 0] - fake_data[j, left:right, 0]))
                    diff.append(d)
                    left += 1
                    right += 1
                mae_list.append(min(diff))
                idx = diff.index(min(diff))
                idx_list.append([idx, idx + train_time[i]])

        plt.subplot(5, 5, j)
        min_y = min(min(fake_data[j, :fake_time[j], 0]),
                    min(train_data[mae_list.index(min(mae_list)), :train_time[mae_list.index(min(mae_list))], 0]))
        max_y = max(max(fake_data[j, :fake_time[j], 0]),
                    max(train_data[mae_list.index(min(mae_list)), :train_time[mae_list.index(min(mae_list))], 0]))
        plt.ylim([min_y, max_y])

        match_idx = idx_list[mae_list.index(min(mae_list))]
        x = [_ for _ in range(match_idx[0], match_idx[1])]
        if train_time[mae_list.index(min(mae_list))] >= fake_time[j]:
            plt.plot(train_data[mae_list.index(min(mae_list)), :train_time[mae_list.index(min(mae_list))], 0],
                     label='train data', linewidth='2.5')
            plt.plot(x, fake_data[j, :fake_time[j], 0], label='fake data')
        else:
            plt.plot(x, train_data[mae_list.index(min(mae_list)), :train_time[mae_list.index(min(mae_list))], 0],
                     label='train data', linewidth='2.5')
            plt.plot(fake_data[j, :fake_time[j], 0], label='fake data')

    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)

    plt.show()

# dis_similar()

# gradients = open('gradients.txt','a+')
# gradients.write('xxxxx\n')
# gradients.write('qqqqq\n')
# gradients.close()

