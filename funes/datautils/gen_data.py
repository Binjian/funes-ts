import pandas as pd
import scipy
from scipy.signal import square,sawtooth
import numpy as np
from scipy import signal
import datetime
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import warnings
warnings.filterwarnings('ignore')

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
            freq = np.random.uniform(0, 0.5)
            phase = np.random.uniform(0, 1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)

    return np.array(data)


def square_wave_data(no, seq_len, dim):

    data = list()

    for i in range(no):
        temp = list()
        for k in range(dim):
            p1 = np.random.uniform(0.4, 0.8)
            p2 = np.random.uniform(2, 10)
            # p1 = np.random.uniform(0, 0.05)
            # p2 = np.random.uniform(8, 9)
            temp_data = np.linspace(0, 0.5, seq_len, endpoint=False)
            # temp_data = square(2 * np.pi * p2 * temp_data,duty=p1)
            temp_data = square(2 * np.pi * p2 * temp_data, duty=p1)+1
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)
    # np.save('square_wave_0.5.npy', np.array(data))
    return np.array(data)


# X = []
# for i in range(10):
#     t = np.linspace(0, 1, 500, endpoint=False)
#     print(t)
#
# X = square_wave_data(10000,50,1)
# X = np.load('square_wave_0.5.npy')
# # print(X.shape)
# np.random.shuffle(X)
# for i in range(20):
#     print(X[i,:,0])
#     plt.plot(list(range(X.shape[1])), X[i,:,0])
#     plt.ylim(-2, 2)
#     plt.show()

# X = np.load('square_wave.npy')



def triangle_wave_data(no,seq_len,dim):
    data = list()

    for i in range(no):
        temp = list()
        for k in range(dim):
            p1 = np.random.uniform(0.7, 1) # (0.9,1)
            p2 = np.random.random_integers(6, 10)
            temp_data = np.linspace(0.5, 0.7, seq_len, endpoint=False)
            temp_data = sawtooth(2* np.pi * p2 * temp_data,width=p1)
            temp.append(temp_data)
            # print(p1,p2)


        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)
        # plt.plot(list(range(seq_len)), temp)
        # plt.show()
    # np.save('triangle_wave.npy',np.array(data))
    return np.array(data)

# X1 = square_wave_data(5000,50,1)
X2 = triangle_wave_data(20000,50,1)
X3 = sine_data_generation(20000,50,1)
X = np.concatenate((X2,X3),axis=0)
# np.save('sine+tri_40000data.npy',X)
# # print(X.shape)

# X1 = np.load('sine+tri_data.npy')
# X2 = triangle_wave_data(5000,50,1)
# X = np.concatenate((X1,X2),axis=0)
# np.save('sine+trix2_data.npy',X)
# print(X.shape)
# X = np.load('sine+tri_20000data.npy')
# X = triangle_wave_data(10000,50,1)
# np.random.shuffle(X)
# for i in range(20):
#     print(X[i,:,0])
#     plt.plot(list(range(X.shape[1])), X[i,:,0])
#     plt.ylim(-2, 2)
#     plt.show()


from scipy import interpolate

# x = np.arange(0, 2*np.pi+np.pi/4, 2*np.pi/8)
# y = np.sin(x)
# t = signal.bspline(x,1)
# tck = interpolate.splrep(x, y, s=0)
# print('xxx',tck)
# xnew = np.arange(0, 2*np.pi, np.pi/50)
# ynew = interpolate.splev(xnew, tck, der=0)
# yder = interpolate.splev(xnew, tck, der=1)
# yders = interpolate.spalde(xnew, tck)

# plt.plot(x, y, 'x', xnew, ynew, xnew, np.sin(xnew), x, y, 'b')

from scipy.optimize import curve_fit

# def func(x, a, b):
#     return a*np.power(x,3)+b


# xdata = np.linspace(0, 4, 50)
# y = func(xdata,2,0.5)
# rng = np.random.default_rng()
# np.random.seed(1)
# y_noise = 0.2 * rng.normal(size=xdata.size)
# ydata = y+y_noise
# plt.plot(xdata, ydata, 'b-', label='data')
# popt, pcov = curve_fit(func, xdata, ydata)
# np.random.seed(2)
# # popt = popt + 0.2 * rng.normal(size=popt.size)
# print(popt)
# plt.plot(xdata, func(xdata, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
# plt.plot(xdata, func(xdata+0.05*rng.normal(size=xdata.size), *popt), 'r-',label='fit: a=%5.3f, b=%5.3f' % tuple(popt))


# X = np.load('./X_clean_0927.npy')
# T = np.load('./T_clean_0927.npy')
# print(pd.DataFrame(T).value_counts())
# X = X[T==43] # baseline
# T = T[T==43]
# # X = X[T==49]
# # T = T[T==49]
#
# rng = np.random.default_rng()
#
# np.random.seed(1)
# np.random.shuffle(X)
# np.random.seed(1)
# np.random.shuffle(T)

# pic_num = 25
# figure, ax = plt.subplots()
# figure.suptitle('vol data')
# for i in range(0,1000,25):
#     # print(i)
#     for j in range(1, pic_num + 1):
#         plt.subplot(5, 5, j)
#         plt.ylim([min(X[j+i, :T[j+i], 0]), max(X[j+i, :T[j+i], 0])])
#         plt.plot(X[j+i, :T[j+i], 0], label='vol data')
#     plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
#     plt.show()


def sample_data():
    tmp = []
    tmp_t = []
    list_t = []
    step = 2
    max_len = 50
    for i in X:
        left = 0
        right = left + step
        t = []
        t2 = []
        while right<i.shape[0]:
            t.append(i[left,:])
            if i[left,0] != 0:
                t2.append(left)
            t.append(i[right,:])
            if i[right, 0] != 0:
                t2.append(right)
            left = right+step
            right = left + step
        # print(len(t))
        tmp.append(t[len(t)-max_len:])
        tmp_t.append(len(np.array(t)[:,0].nonzero()[0]))
        list_t.append(t2)
        # print(len(np.array(t)[:,0].nonzero()[0]),len(t2),t2)
        # tmp.append(t[max_len:])
    tmp = np.array(tmp)
    tmp_t = np.array(tmp_t)
    list_t = np.array(list_t)
    print(list_t.shape)
    print(tmp.shape)


def interp_spline():
    # y3 = scipy.interpolate.make_interp_spline(list(range(T[0])), X[0, :T[0], 0], 2)
    # print(y3)
    # y3 = interpolate.splrep(list(range(T[0])), X[0, :T[0], 0], s=0)
    X = np.load('./X_clean_0927.npy')
    T = np.load('./T_clean_0927.npy')
    print(set(T))

    X = X[T == 43]  # baseline
    np.random.shuffle(X)
    T = T[T == 43]
    t = 43
    data = []
    for x in X:
        f = interpolate.interp1d(list(range(t)), x[:t,0])
        xnew = np.arange(0,t-1,0.84)
        print(list(range(t)))
        print(xnew)
        ynew = f(xnew)
        data.append(ynew)
        plt.plot(list(range(t)), x[:t, 0], '--', lw=2.5)
        # plt.plot(list(range(T[0])),y3(list(range(T[0])))+y_noise)
        plt.plot(xnew, ynew)
        plt.show()
    data = np.expand_dims(np.array(data),axis=-1)
    print(data.shape)

# interp_spline()


def B_spline(seq_len):
    print(T[0])
    # t = np.array([0, 6, 12, 18, 24, 28, 32, 36, 40, 42])
    tck = interpolate.splrep(list(range(T[0])), X[0,:T[0],0],k=3)
    print(tck)
    print(len(tck[0]),len(tck[1]),tck[2])

    t = tck[0]
    c = tck[1]
    k = tck[2]
    data = []
    for i in range(10):

    # idx = [0,1,2,8,12,22,30,38,40,41,42,43,44] # k=1  13-4
    # idx = [0,1,2,5,18,21,24,34,37,38,40,41,42,43,44,45] # k=2  16-6
        idx = [0, 1, 2, 3, 18, 20,22,24,35, 36,38, 39, 40, 41, 42, 43, 44, 45,46] # k=3 19-8
        # print('xxxxx',t[idx],c[idx])
        noise = [0,0,0,0] + list(rng.uniform(-0.5, 0.5, size=len(idx)-8)) + [0,0,0,0]
        noise_c = [0, 0, 0, 0] + list(rng.uniform(-20, 20, size=len(idx) - 8)) + [0, 0, 0, 0]
        print(len(noise))
        noise = np.array(noise)

        t_ = t[idx] + noise
        c_ = c[idx] + noise_c
        # c_ = c[idx] + noise*10

        print(t_)
        print(c_)
        # t_noise = rng.uniform(0.1, 0.5, size=len(t))
        # c_noise = rng.uniform(0, 1, size=len(c))

        spl = interpolate.BSpline(t_, c_, k)

        print(len(t_),len(c_),k)
        x_new = np.linspace(0, T[0]-1, seq_len)
        # print(interpolate.BSpline(t,c,k))
        # print(spl(x_new))
        # print(x_new)
        # print(spl(x_new))
        # plt.plot(t_[4:],c_[:-4],'.')
        plt.plot(X[0,:T[0],0],'--')
        plt.plot(x_new,interpolate.BSpline(t[idx], c[idx], k)(x_new),linewidth='2.5',c='red')
        # plt.plot(t,c)
        # print('####',x_new,spl(x_new))
        plt.plot(x_new,spl(x_new))
        data.append(spl(x_new))
    # print(data[0])
    data = np.array(data)
    data = np.expand_dims(data,axis=-1)
    print(data.shape)
    # np.save('vol_gen_data_50.npy',data)
    plt.show()


def gen_other_vol_data(data_len,seq_len):
    all_X = np.load('../datautils/X_clean_0927.npy')
    all_T = np.load('../datautils/T_clean_0927.npy')
    # np.random.shuffle(T)
    print(set(all_T))
    data = []
    for j in set(all_T):
        print(f'###{j}###')
        if j==43:
            continue
        X = all_X[all_T == j][3:] # baseline
        T = all_T[all_T == j][3:]
        # print(T)
        rng = np.random.default_rng()
        # t = np.array([0, 6, 12, 18, 24, 28, 32, 36, 40, 42])
        tck = interpolate.splrep(list(range(T[0])), X[0, :T[0], 0], k=3)
        # print(X[0, :T[0], 0])

        t = tck[0]
        c = tck[1]
        k = tck[2]
        print(len(list(range(len(t)))),len(t))
        # data = []
        for _ in range(data_len):
            # idx = [0,1,2,8,12,22,30,38,40,41,42,43,44] # k=1  13-4
            # idx = [0,1,2,5,18,21,24,34,37,38,40,41,42,43,44,45] # k=2  16-6
            idx = list(range(len(t)))  # k=3 19-8
            # noise_t = [0, 0, 0, 0] + list(rng.uniform(-0.5, 0.5, size=len(idx) - 8)) + [0, 0, 0, 0]
            # noise_c = [0, 0, 0, 0] + list(rng.uniform(-20, 20, size=len(idx) - 8)) + [0, 0, 0, 0]
            noise_t = []
            noise_c = []
            var = 0.45
            # noise = np.array(noise)
            for i in range(len(idx)):
                if i == 0:
                    min_gap_t = abs(t[idx][i + 1] - t[idx][i])
                    min_gap_c = abs(c[idx][i + 1] - c[idx][i])
                elif i == len(idx) - 1:
                    min_gap_t = abs(t[idx][i] - t[idx][i - 1])
                    min_gap_c = abs(c[idx][i] - c[idx][i - 1])
                else:
                    min_gap_t = min(abs(t[idx][i] - t[idx][i - 1]), abs(t[idx][i + 1] - t[idx][i]))
                    min_gap_c = min(abs(c[idx][i] - c[idx][i - 1]), abs(c[idx][i + 1] - c[idx][i]))

                n_t = rng.uniform(-var * min_gap_t, var * min_gap_t, size=1).item()
                n_c = rng.uniform(-var * min_gap_c, var * min_gap_c, size=1).item()
                noise_t.append(n_t)
                noise_c.append(n_c)

            t_ = t[idx] + noise_t
            c_ = c[idx] + noise_c

            spl = interpolate.BSpline(t_, c_, k)
            spl_ori = interpolate.BSpline(t[idx], c[idx], k)

            x_new = np.linspace(0, T[0] - 1, seq_len)

            # origin data
            plt.plot(X[0, :T[0], 0], '--')

            # origin synthetic data
            plt.plot(x_new, spl_ori(x_new), linewidth='2.5', c='red')

            # synthetic data with noise
            plt.plot(x_new, spl(x_new))
            data.append(spl(x_new))

        plt.show()
        # print(data[0])
    data = np.array(data)
    data = np.expand_dims(data, axis=-1)
    print(data.shape)
    # np.save('vol_gen_data_50.npy',data)
    plt.show()
# gen_other_vol_data(5,50)

'''
43    556
15    396
37    396
20    396
24    396
27    396
30    396
39    395
34    395
47    386
42    326
44    285
49    267
32    198
40    198
23    198
22    198
21    198
46    197
51    167
45    152
64    107
60     74
53     39
76     32
74      5
85      1
'''

def gen_vol_data(data_len,seq_len):
    X = np.load('./X_clean_0927.npy')
    T = np.load('./T_clean_0927.npy')
    print(set(T))

    X = X[T == 43] # baseline
    T = T[T == 43]
    rng = np.random.default_rng()
    # t = np.array([0, 6, 12, 18, 24, 28, 32, 36, 40, 42])
    tck = interpolate.splrep(list(range(T[0])), X[0, :T[0], 0], k=3)
    print(X[0, :T[0], 0])

    t = tck[0]
    c = tck[1]
    k = tck[2]
    data = []
    for _ in range(data_len):
        # idx = [0,1,2,8,12,22,30,38,40,41,42,43,44] # k=1  13-4
        # idx = [0,1,2,5,18,21,24,34,37,38,40,41,42,43,44,45] # k=2  16-6
        idx = [0, 1, 2, 3, 18, 20, 22, 24, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46]  # k=3 19-8
        # noise_t = [0, 0, 0, 0] + list(rng.uniform(-0.5, 0.5, size=len(idx) - 8)) + [0, 0, 0, 0]
        # noise_c = [0, 0, 0, 0] + list(rng.uniform(-20, 20, size=len(idx) - 8)) + [0, 0, 0, 0]
        noise_t = []
        noise_c = []
        var = 0.45
        # noise = np.array(noise)
        for i in range(len(idx)):
            if i == 0:
                min_gap_t = abs(t[idx][i + 1] - t[idx][i])
                min_gap_c = abs(c[idx][i + 1] - c[idx][i])
            elif i == len(idx) - 1:
                min_gap_t = abs(t[idx][i] - t[idx][i - 1])
                min_gap_c = abs(c[idx][i] - c[idx][i - 1])
            else:
                min_gap_t = min(abs(t[idx][i] - t[idx][i - 1]), abs(t[idx][i + 1] - t[idx][i]))
                min_gap_c = min(abs(c[idx][i] - c[idx][i - 1]), abs(c[idx][i + 1] - c[idx][i]))

            n_t = rng.uniform(-var * min_gap_t, var * min_gap_t, size=1).item()
            n_c = rng.uniform(-var * min_gap_c, var * min_gap_c, size=1).item()
            noise_t.append(n_t)
            noise_c.append(n_c)

        t_ = t[idx] + noise_t
        c_ = c[idx] + noise_c

        spl = interpolate.BSpline(t_, c_, k)
        spl_ori = interpolate.BSpline(t[idx], c[idx], k)

        x_new = np.linspace(0, T[0] - 1, seq_len)

        # origin data
        plt.plot(X[0, :T[0], 0], '--')

        # origin synthetic data
        plt.plot(x_new, spl_ori(x_new), linewidth='2.5', c='red')

        # synthetic data with noise
        plt.plot(x_new, spl(x_new))
        data.append(spl(x_new))
    # print(data[0])
    data = np.array(data)
    data = np.expand_dims(data, axis=-1)
    # print(data[0])
    # np.save('vol_gen_data_50.npy',data)
    plt.show()
# gen_vol_data(10,50)

# l = [1,4,6,7,5,7,8,10,13,12,14]
# noise = []
# for i in range(len(l)):
#     if i==0:
#         min_gap = abs(l[i+1]-l[i])
#     elif i==len(l)-1:
#         min_gap = abs(l[i]-l[i-1])
#     else:
#         min_gap = min(abs(l[i] - l[i - 1]), abs(l[i + 1] - l[i]))
#
#     n = rng.uniform(-0.3*min_gap, 0.3*min_gap, size=1).item()
#     noise.append(n)
#     print(min_gap,n)
# print(noise)

from kats.tsfeatures.tsfeatures import TsFeatures
from kats.consts import TimeSeriesData
import time

FEATURES = ["spikiness","peak","trough","level_shift_idx","level_shift_size","y_acf1",
        "y_acf5","diff1y_acf1","diff1y_acf5","diff2y_acf1","diff2y_acf5","y_pacf5","diff1y_pacf5","diff2y_pacf5","seas_acf1",
        "seas_pacf1","firstmin_ac","firstzero_ac","holt_alpha","holt_beta","hw_alpha","hw_beta","hw_gamma","length","mean",
        "var","entropy","lumpiness","stability","flat_spots","hurst","std1st_der","crossing_points","binarize_mean","unitroot_kpss",
        "heterogeneity","histogram_mode","linearity"]

def get_kats_feature(data):
    ts_model = TsFeatures(selected_features=FEATURES)

    feature_data = []
    for i in range(len(data)):
        time_stamps = []
        ts_now = int(time.time())
        for j in range(data.shape[1], 0, -1):  # make timestamp
            ts1 = ts_now - 2 * i
            ts2 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts1))
            time_stamps.append(ts2)
        df = pd.DataFrame()
        df['time'] = time_stamps
        df['value'] = data[i]
        # print('xxxx',len(df['value'].unique()))

        if len(df) > 24 and len(df['value'].unique()) >= 3:
        # if len(df) > 24 and len(df['value'].unique()) >= 1: # square
            ts = TimeSeriesData(df)
            features = ts_model.transform(ts)
            feature_data.append(list(features.values()))
        else:
            # train_feature.append(np.array([]))
            feature_data.append([])
    return np.array(feature_data)



from sklearn.decomposition import PCA

def multimodal_vol_pca():
    def g(data_len, seq_len, mode=43):
        X = np.load('../datautils/X_clean_0927.npy')
        T = np.load('../datautils/T_clean_0927.npy')

        X = X[T == mode]
        T = T[T == mode]
        rng = np.random.default_rng()
        # t = np.array([0, 6, 12, 18, 24, 28, 32, 36, 40, 42])
        tck = interpolate.splrep(list(range(T[0])), X[0, :T[0], 0], k=3)
        # print(X[0, :T[0], 0])

        t = tck[0]
        c = tck[1]
        k = tck[2]
        data = []
        for i in range(data_len):
            # idx = [0,1,2,8,12,22,30,38,40,41,42,43,44] # k=1  13-4
            # idx = [0,1,2,5,18,21,24,34,37,38,40,41,42,43,44,45] # k=2  16-6
            if mode == 43:
                idx = [0, 1, 2, 3, 18, 20, 22, 24, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46]  # k=3 19-8
            else:
                idx = list(range(len(t)))
            # noise_t = [0, 0, 0, 0] + list(rng.uniform(-0.5, 0.5, size=len(idx) - 8)) + [0, 0, 0, 0]
            # noise_c = [0, 0, 0, 0] + list(rng.uniform(-20, 20, size=len(idx) - 8)) + [0, 0, 0, 0]
            noise_t = []
            noise_c = []
            var = 0.45
            # noise = np.array(noise)
            for i in range(len(idx)):
                if i == 0:
                    min_gap_t = abs(t[idx][i + 1] - t[idx][i])
                    min_gap_c = abs(c[idx][i + 1] - c[idx][i])
                elif i == len(idx) - 1:
                    min_gap_t = abs(t[idx][i] - t[idx][i - 1])
                    min_gap_c = abs(c[idx][i] - c[idx][i - 1])
                else:
                    min_gap_t = min(abs(t[idx][i] - t[idx][i - 1]), abs(t[idx][i + 1] - t[idx][i]))
                    min_gap_c = min(abs(c[idx][i] - c[idx][i - 1]), abs(c[idx][i + 1] - c[idx][i]))

                n_t = rng.uniform(-var * min_gap_t, var * min_gap_t, size=1).item()
                n_c = rng.uniform(-var * min_gap_c, var * min_gap_c, size=1).item()
                noise_t.append(n_t)
                noise_c.append(n_c)

            t_ = t[idx] + noise_t
            c_ = c[idx] + noise_c

            spl = interpolate.BSpline(t_, c_, k)
            spl_ori = interpolate.BSpline(t[idx], c[idx], k)

            x_new = np.linspace(0, T[0] - 1, seq_len)

            # origin data
            # plt.plot(X[0, :T[0], 0], '--')

            # origin synthetic data
            # plt.plot(x_new, spl_ori(x_new), linewidth='2.5', c='red')

            # synthetic data with noise
            # plt.plot(x_new, spl(x_new))
            data.append(spl(x_new))
        # print(data[0])
        data = np.array(data)
        data = np.expand_dims(data, axis=-1)
        # np.save('vol_gen_data_50.npy',data)
        # plt.show()
        return data

    multimodal_list = [30, 34, 37, 40, 43]
    data = np.array([])
    labels = []
    for m in range(len(multimodal_list)):
        print(multimodal_list[m])
        vol_data = g(1000, 50, mode=multimodal_list[m])
        if m == 0:
            data = vol_data
        else:
            data = np.concatenate((data, vol_data), axis=0)
        labels += [m for _ in range(vol_data.shape[0])]

    labels = np.array(labels)
    print(labels)
    print(data.shape, labels.shape)

    data = get_kats_feature(data)
    print(data.shape)

    colors = ["tab:blue" for i in range(1000)] + [
        "tab:orange" for i in range(1000)
    ]+["tab:red" for i in range(1000)] + [
        "tab:brown" for i in range(1000)
    ] + ["tab:green" for i in range(1000) ]
    pca = PCA()
    pca.fit(data)
    pca_results = pca.transform(data)
    # Plotting
    f, ax = plt.subplots(1)
    for i in range(1,6):
        plt.scatter(
            pca_results[(i-1)*1000:i*1000, 0],
            pca_results[(i-1)*1000:i*1000, 1],
            c=colors[(i-1)*1000:i*1000],
            alpha=0.2,
            label="Original",
        )

    ax.legend()
    plt.title("PCA plot")
    plt.xlabel("x-pca")
    plt.ylabel("y_pca")
    # plt.savefig(
    #         '/home/veos/devel/funes/visualization_results/5_vol_PCA' + '-' + datetime.datetime.now().strftime(
    #             "%Y%m%d-%H%M%S") + '.svg', dpi=300)
    plt.show()


    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)

    f, ax = plt.subplots(1)
    for i in range(1,6):
        plt.scatter(
            tsne_results[(i-1)*1000:i*1000, 0],
            tsne_results[(i-1)*1000:i*1000, 1],
            c=colors[(i-1)*1000:i*1000],
            alpha=0.2,
            label="Original",
        )

    ax.legend()

    plt.title("t-SNE plot")
    plt.xlabel("x-tsne")
    plt.ylabel("y_tsne")
    # plt.savefig(
    #         '/home/veos/devel/funes/visualization_results/5_vol_t-SNE' + '-' + datetime.datetime.now().strftime(
    #             "%Y%m%d-%H%M%S") + '.svg', dpi=300)
    plt.show()


    umap_results = umap.UMAP(n_neighbors=10, min_dist=0.1).fit_transform(data)

    for i in range(1, 6):
        plt.scatter(
            umap_results[(i-1)*1000:i*1000, 0],
            umap_results[(i-1)*1000:i*1000, 1],
            c=colors[(i - 1) * 1000:i * 1000],
            alpha=0.2,
            label="Original",
        )

    plt.legend()
    plt.title("umap plot")
    plt.xlabel("x-umap")
    plt.ylabel("y_umap")
    # plt.savefig(
    #         '/home/veos/devel/funes/visualization_results/5_vol_UMAP' + '-' + datetime.datetime.now().strftime(
    #             "%Y%m%d-%H%M%S") + '.svg', dpi=300)
    plt.show()

# multimodal_vol_pca()
