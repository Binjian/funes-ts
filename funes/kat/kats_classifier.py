import pickle
from collections import OrderedDict
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
import time
import torch
import sklearn
from sklearn.neural_network import MLPClassifier
from scipy.signal import square,sawtooth
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from funes.tgan.models.timegan import TimeGAN
import datetime
from scipy import interpolate

def gen_other_vol_data(data_len,seq_len):
    all_X = np.load('../datautils/X_clean_0927.npy')
    all_T = np.load('../datautils/T_clean_0927.npy')
    # np.random.shuffle(T)
    print(set(all_T))
    data = []
    for j in set(all_T):
        if j==43:
            continue
        X = all_X[all_T == j] # baseline
        T = all_T[all_T == j]
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

def sine_data_generation(no, seq_len, dim):

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
            temp_data = [np.sin(freq * j + phase)+1e-8 for j in range(seq_len)]
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

def triangle_wave_data(no,seq_len,dim):
    data = list()

    for i in range(no):
        temp = list()
        for k in range(dim):
            p1 = np.random.uniform(0.7, 1) # (0.9,1)
            p2 = np.random.random_integers(6, 10)
            temp_data = np.linspace(0.5, 0.7, seq_len, endpoint=False)
            temp_data = sawtooth(2* np.pi * p2 * temp_data,width=p1)
            temp.append(temp_data+np.array([1e-8 for _ in temp_data]))
        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)
    return np.array(data)

def gen_vol_data(data_len,seq_len):
    X = np.load('../datautils/X_clean_0927.npy')
    T = np.load('../datautils/T_clean_0927.npy')

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

# data process, get feature, train, classification
# FEATURES = ["hw_alpha","hw_beta","hw_gamma","length","mean",
#         "var","entropy","lumpiness","stability","flat_spots","hurst","std1st_der","crossing_points","binarize_mean","unitroot_kpss",
#         "heterogeneity","histogram_mode","linearity"
#        ]

FEATURES = ["spikiness","peak","trough","level_shift_idx","level_shift_size","y_acf1",
        "y_acf5","diff1y_acf1","diff1y_acf5","diff2y_acf1","diff2y_acf5","y_pacf5","diff1y_pacf5","diff2y_pacf5","seas_acf1",
        "seas_pacf1","firstmin_ac","firstzero_ac","holt_alpha","holt_beta","hw_alpha","hw_beta","hw_gamma","length","mean",
        "var","entropy","lumpiness","stability","flat_spots","hurst","std1st_der","crossing_points","binarize_mean","unitroot_kpss",
        "heterogeneity","histogram_mode","linearity"]

data_path = '../../data/output/Sine_1013/Sine-20221013-105200_best/mdl/Sine-vanilla-1000-1000-1000-64-32-3-0.001-50-4000'
# data_path = '../../data/output/vol_gen_1027/VOL-20221102-014426/mdl/VOL-vanilla-1200-1200-1200-32-8-3-0.001-100-4000'

def gen_fake(args):
    train_time = pd.read_pickle(f'{data_path}/train_time.pickle')
    train_data = pd.read_pickle(f'{data_path}/train_data.pickle')
    # Ts = pd.read_pickle(f"{data_path}/train_time.pickle")

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

#####  gen data  #####
# fake_data,train_data,_,_ = gen_fake(args)
# print(fake_data.shape,train_data.shape)
# data = np.concatenate((train_data,fake_data),axis=0)
# train_label = [1 for _ in range(train_data.shape[0])]
# fake_label= [0 for _ in range(fake_data.shape[0])]
# labels = np.array(train_label + fake_label)

#####  sine triangle square  #####
# sine_data = sine_data_generation(300,50,1)
# sine_label = [0 for _ in range(sine_data.shape[0])]
# triangle_data = triangle_wave_data(300,50,1)
# triangle_label = [0 for _ in range(triangle_data.shape[0])]
# square_data = square_wave_data(1000,50,1)
# square_label = [0 for _ in range(square_data.shape[0])]

vol_data = gen_other_vol_data(10,50)
# vol_data = gen_vol_data(50,50)
vol_label = [0 for _ in range(vol_data.shape[0])]
# data = np.concatenate((triangle_data,sine_data),axis=0)
# labels = np.array(triangle_label+sine_label)
data = vol_data
labels = np.array(vol_label)
print(data.shape,labels.shape)

START = time.time()
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

        # if len(df) > 24 and len(df['value'].unique()) >= 3:
        if len(df) > 24 and len(df['value'].unique()) >= 1: # square
            ts = TimeSeriesData(df)
            features = ts_model.transform(ts)
            feature_data.append(list(features.values()))
        else:
            # train_feature.append(np.array([]))
            feature_data.append([])
    return np.array(feature_data)
# train_feature = torch.tensor(train_feature,dtype=torch.float)
feature_data = get_kats_feature(data)
# squ = get_kats_feature(square_data)
# print(squ.shape,len(squ[0]),squ[0])
END = time.time()
print(feature_data.shape,labels.shape)
print(f'time: {(END-START)/60}mins')

X_train,X_test,Y_train,Y_test = train_test_split(feature_data,labels,test_size=0.2,shuffle=True)
print(X_train.shape,Y_train.shape)

# def MLP_clf(X_train,Y_train,X_test,Y_test):
#     clf = MLPClassifier(solver='adam',hidden_layer_sizes=(32,16),batch_size=256,learning_rate='constant',
#                         learning_rate_init=0.001,max_iter=1000,verbose=True,early_stopping=False)
#     clf.fit(X_train,Y_train)
#     y_pred = clf.predict(X_test)
#     y_true = list(Y_test)
#     # print(cls.score(X_test,Y_test))
#     print(classification_report(y_true,y_pred))
#     # with open('../../data/output/kats_output/skl_cls_sine_1000epoch.pickle', 'wb') as f:
#     #     pickle.dump(clf,f)

    # cls = pickle.load('./skl_cls_200epoch.pickle')

# MLP_clf(X_train,Y_train,X_test,Y_test)



class KATS_Classifier(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(KATS_Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim,hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, int(hidden_dim/2))
        self.out = torch.nn.Linear(int(hidden_dim/2), 1)
        # self.out = torch.nn.Linear(int(hidden_dim / 2), 3)
        # self.softmax = torch.nn.Softmax()
        self.relu = torch.nn.ReLU()


    def forward(self,X):
        h1 = self.linear1(X)
        h2 = self.relu(self.linear2(h1))
        # out = self.softmax(self.out(h2))
        out = self.out(h2)
        return out

class KATS_Dataset(torch.utils.data.Dataset):
    def __init__(self,data,label):
        self.data = data
        self.label = label

    def __getitem__(self, item):
        data = self.data[item]
        label = self.label[item]
        return data,label

    def __len__(self):
        return len(self.data)

def evaluator(model,dataloader,THRESHOLD):
    model.eval()
    l_true = []
    l_pred = []
    logit = []
    for x, y in dataloader:
        x = x.to('cuda')
        out = model(x)
        # print(out.squeeze(-1).shape,l.shape)
        pred = out.squeeze(-1)
        # print(pred.shape)
        logit += list(out.sigmoid().cpu().detach().numpy())
        # print(out)
        # print(float(out.sigmoid()))
        pred[pred >= THRESHOLD] = 1
        pred[pred < THRESHOLD] = 0
        # print(pred)
        l_pred += list(pred.detach().to('cpu'))
        l_true += list(y)
    # print(l_pred)
    # print(l_true)

    print(classification_report(l_true,l_pred))

    wrong_list = []
    print(len(logit))
    plt.figure()
    for i in range(len(logit)):
        if logit[i] < THRESHOLD:
            plt.scatter(i, logit[i], color="orange", label="Warning Data")
            # warn_list.append(i)
        else:
            plt.scatter(i, logit[i], color="blue", label="Original Data")
            plt.annotate(len(wrong_list),(i,logit[i]))
            # fake_norm_dir[i] = logit[i]
            wrong_list.append(i)
    plt.xlabel("Sample no.")
    plt.ylabel("Logits")
    plt.title("Data Logits")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    # plt.savefig(
    #         '/home/veos/devel/funes/visualization_results/KATS_Data_Logits' + '-' + datetime.datetime.now().strftime(
    #             "%Y%m%d-%H%M%S") + '.svg', dpi=300)
    plt.show()


def train(model,train_dataloader,val_dataloader,EPOCH,lr,save_path):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.functional.binary_cross_entropy_with_logits

    for epoch in range(EPOCH):
        loss = []
        start = time.time()
        for x, y in train_dataloader:
            x = x.to('cuda')
            y = y.to('cuda')
            model.zero_grad()
            out = model(x)
            l = loss_func(out.squeeze(-1), y)
            loss.append(l)
            l.backward()
            opt.step()
        # print(f'epoch: {epoch + 1}, loss: {sum(loss) / len(loss):4f}, time: {(time.time() - start) / 60:3f}s')
        with torch.no_grad():
            val_loss = []
            for x,y in val_dataloader:
                x = x.to('cuda')
                y = y.to('cuda')
                out = model(x)
                l = loss_func(out.squeeze(-1), y)
                val_loss.append(l)
        print(f'epoch: {epoch + 1}, train loss: {sum(loss) / len(loss):4f}, validation loss: {sum(val_loss)/len(val_loss):4f}, time: {(time.time() - start) / 60:3f}s')

        if (epoch + 1) % 10 == 0:
            print(f'saving model... epoch {epoch+1}')
            torch.save(model.state_dict(), save_path)

THRESHOLD = 0.5
input_dim = 38
hidden_dim = 32
EPOCH = 500
batch_size = 256
lr = 0.001
save_path = f'../../data/output/kats_output/kats_clf_all_feature_vol_{EPOCH}_{batch_size}_{hidden_dim}_{lr}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.params'

clf = KATS_Classifier(input_dim,hidden_dim)
clf.to('cuda')
X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size=0.1,shuffle=False)
print(X_train.shape,X_test.shape,X_val.shape)
X_train = torch.tensor(X_train,dtype=torch.float)
Y_train = torch.tensor(Y_train,dtype=torch.float)
trainSet = KATS_Dataset(X_train,Y_train)
trainSet_loader = DataLoader(trainSet,batch_size=batch_size)

X_val = torch.tensor(X_val,dtype=torch.float)
Y_val = torch.tensor(Y_val,dtype=torch.float)
validationSet = KATS_Dataset(X_val,Y_val)
validationSet_loader = DataLoader(validationSet,batch_size=batch_size)

# train(clf,trainSet_loader,validationSet_loader,EPOCH,lr,save_path)

X_test = torch.tensor(X_test,dtype=torch.float)
Y_test = torch.tensor(Y_test,dtype=torch.float)
testSet = KATS_Dataset(X_test,Y_test)
testSet_loader = DataLoader(testSet,batch_size=batch_size)
# clf.load_state_dict(torch.load(save_path))
# evaluator(clf,testSet_loader,THRESHOLD)


######## test with other data ##########
# feature_data = get_kats_feature(feature_data)
# print(feature_data.shape)
feature_data = torch.tensor(feature_data,dtype=torch.float)
labels = torch.tensor(labels,dtype=torch.float)
dataSet = KATS_Dataset(feature_data,labels)
dataSet_loader = DataLoader(dataSet,batch_size=batch_size)
clf.load_state_dict(torch.load('../../data/output/kats_output/kats_clf_all_feature_vol_500_256_32_0.001_20221109-172722.params'))
evaluator(clf,dataSet_loader,THRESHOLD)

