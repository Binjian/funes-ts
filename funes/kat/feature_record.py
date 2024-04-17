import numpy as np
import re
import matplotlib.pyplot as plt

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

FEATURES_dic = {'stl_features':["spikiness","peak","trough"],"level_shift_features": ["level_shift_idx","level_shift_size"],"acfpacf_features":["y_acf1",
        "y_acf5","diff1y_acf1","diff1y_acf5","diff2y_acf1","diff2y_acf5","y_pacf5","diff1y_pacf5","diff2y_pacf5","seas_acf1",
        "seas_pacf1"],"special_ac":["firstmin_ac","firstzero_ac"],"holt_params":["holt_alpha","holt_beta"],"hw_params":["hw_alpha","hw_beta","hw_gamma"],
        "statistics":["length","mean","var","entropy","lumpiness","stability","flat_spots","hurst","std1st_der","crossing_points","binarize_mean","unitroot_kpss",
        "heterogeneity","histogram_mode","linearity"]}
FEATURES = ["spikiness","peak","trough","level_shift_idx","level_shift_size","y_acf1",
        "y_acf5","diff1y_acf1","diff1y_acf5","diff2y_acf1","diff2y_acf5","y_pacf5","diff1y_pacf5","diff2y_pacf5","seas_acf1",
        "seas_pacf1","firstmin_ac","firstzero_ac","holt_alpha","holt_beta","hw_alpha","hw_beta","hw_gamma","length","mean",
        "var","entropy","lumpiness","stability","flat_spots","hurst","std1st_der","crossing_points","binarize_mean","unitroot_kpss",
        "heterogeneity","histogram_mode","linearity"]

with open('feature_record_1122.txt', 'r') as f:
    record = f.read().split('\n\n')
# print(record[0].split(']\n'))
# print(re.sub('[" "]+',' ',record[0]))
# r = record[0].split(']\n')[0] + ']'
# r = re.sub('[" "]+',' ',r)
# print(r)
# print([eval(i) for i in r[1:-1].split(' ')])


feature_grad = []
MinMaxScaler_grad = []
sorted_idx = []
for r in record[:-1]:
    r = re.sub('[" "]+',' ',r)
    # print(r.split(']\n')[2][1:-1])
    # print([eval(i) for i in r.split(']\n')[0][1:].strip().split(' ')])
    feature_grad.append(np.array([eval(i) for i in r.split(']\n')[0][1:].strip().split(' ')]))
    MinMaxScaler_grad.append(np.array([eval(i) for i in r.split(']\n')[1][1:].strip().split(' ')]))
    sorted_idx.append(list([eval(i) for i in r.split(']\n')[2][1:-1].strip().split(' ')]))

score = [0 for _ in range(len(FEATURES))]
for s in sorted_idx:
    print(s[-5:])
    for i in range(len(s)):
        score[s[i]] += i
# print(MinMaxScaler(np.array(score)))
grad = MinMaxScaler(np.array(score))
sorted_index = list(np.argsort(-grad)[:10]) # descent order
print(sorted_index)
print([FEATURES[i] for i in sorted_index])
print(grad[sorted_index])
print([FEATURES[i] for i in sorted_index])

# print(np.sum(np.array(feature_grad),axis=0))
# grad = MinMaxScaler(np.sum(np.array(feature_grad),axis=0))
# # grad = MinMaxScaler(np.sum(np.array(MinMaxScaler_grad),axis=0))
# # MinMaxScaler_grad = MinMaxScaler(np.sum(np.array(MinMaxScaler_grad),axis=0))
# # print(MinMaxScaler_grad)
# print(grad)
# sorted_index = list(np.argsort(-grad)[:10]) # descent order
# print(sorted_index)
# print([FEATURES[i] for i in sorted_index])

# get feature color
color_set = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','	tab:cyan']
colors = []
c = 0
for f in FEATURES_dic:
    colors += [color_set[c] for _ in range(len(FEATURES_dic[f]))]
    c += 1
    if c >= len(color_set):
        print('no color available !')
        break
print(colors)

# get every feature index
feature_idx = []
tmp = 0
for f in FEATURES_dic:
    idx = list(range(len(FEATURES)))[tmp:len(FEATURES_dic[f])+tmp]
    tmp = idx[-1]+1
    feature_idx.append(idx)
print(feature_idx)

plt.figure()
for j in range(len(feature_idx)):
    for i in range(len(grad)):
        if i in feature_idx[j]:
            plt.subplot(1, len(FEATURES_dic), j + 1)
            plt.ylim([-0.05, 1.05])
            plt.scatter(i, grad[i], color=colors[i], linewidths=grad[i] * 10)
            if i in sorted_index:
                plt.annotate(str(sorted_index.index(i)+1) + '-' + FEATURES[i], (i, grad[i]))
            plt.xlabel(list(FEATURES_dic)[j])
            ax = plt.gca()
            ax.axes.xaxis.set_ticks([])
            # ax.axes.yaxis.set_ticks([])

# plt.xlabel("Feature no.")
# plt.ylabel("Grad")
# plt.title("Feature Importance")
# plt.legend()
# plt.show()

# grad = np.reshape(grad,(1,len(FEATURES)))
# plt.matshow(grad, cmap=plt.cm.Blues)
# plt.colorbar()
# plt.title("Ranking")
plt.show()




