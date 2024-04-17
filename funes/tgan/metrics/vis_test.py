"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.
Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.
Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------
visualization_metrics.py
Note: Use PCA or tSNE for generated and original data visualization
"""

# Necessary packages
import os
import pandas as pd
# import torch
import scipy.spatial.distance
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import umap
from scipy import signal,spatial,stats
# from discriminative_metrics import discriminative_score_metrics
# from predictive_metrics import predictive_score_metrics
import datetime
import time
from scipy.fftpack import dct,idct,fft
# from datetime import datetime
__all__ = ["visualization"]

def mixed_visualization(ori_data, analysis, out_list=None, generated_data=None):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      - analysis: tsne or pca
    """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([3000, len(ori_data)]) # 3000
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]


    # Data preprocessing
    ori_data = np.asarray(ori_data)
    ori_data = ori_data[idx]

    # ori = np.load('../../datautils/sine+tri_40000data.npy').squeeze(axis=-1) ##############
    ori = np.load('../../datautils/sine+tri_data.npy').squeeze(axis=-1)
    print(ori.shape, ori_data.shape)
    ori = [list(i) for i in ori]
    data1 = []
    data2 = []
    tmp = [list(i) for i in ori_data.squeeze(axis=-1)]
    for i in range(ori_data.shape[0]):
        if ori.index(tmp[i]) <5000:
        # if ori.index(tmp[i]) < 5000 or ori.index(tmp[i]) >= 10000:
        # if ori.index(tmp[i]) < 20000:
            data1.append(i)
        else:
            data2.append(i)

    if generated_data is not None:
        generated_data = np.asarray(generated_data)
        generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if i == 0:
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            if generated_data is not None:
                prep_data_hat = np.reshape(
                    np.mean(generated_data[0, :, :], 1), [1, seq_len]
                )
        else:
            prep_data = np.concatenate(
                (prep_data, np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len]))
            )
            if generated_data is not None:
                prep_data_hat = np.concatenate(
                    (
                        prep_data_hat,
                        np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len]),
                    )
                )

    # Visualization parameter
    colors = ["tab:blue" for i in range(anal_sample_no)] + [
        "tab:orange" for i in range(anal_sample_no)
    ]
    if analysis == "pca":
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        # Plotting
        f, ax = plt.subplots(1)
        # plt.scatter(
        #     pca_results[:, 0],
        #     pca_results[:, 1],
        #     c=colors[:anal_sample_no],
        #     alpha=0.2,
        #     label="Original",
        # )
        plt.scatter(
            pca_results[:, 0][data1],
            pca_results[:, 1][data1],
            c='tab:blue',
            alpha=0.2,
            label="Original data1",
        )
        plt.scatter(
            pca_results[:, 0][data2],
            pca_results[:, 1][data2],
            c='tab:green',
            alpha=0.2,
            label="Original data2",
        )
        if generated_data is not None:
            pca_hat_results = pca.transform(prep_data_hat)
            plt.scatter(
                pca_hat_results[:, 0],
                pca_hat_results[:, 1],
                c=colors[anal_sample_no:],
                alpha=0.2,
                label="Synthetic",
            )
        if out_list is not None:
            outlier = np.array([pca_results[i, :] for i in out_list])
            plt.scatter(
                outlier[:, 0],
                outlier[:, 1],
                c="black",
                alpha=1,
                marker="D",
                label="Outlier",
            )

        ax.legend()
        plt.title("PCA plot")
        plt.xlabel("x-pca")
        plt.ylabel("y_pca")
        plt.savefig('/home/veos/devel/funes/visualization_results/pca'+'-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.svg', dpi=300)
        plt.show()

    elif analysis == "tsne":
        if generated_data is not None:
            # Do t-SNE Analysis together
            prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
        else:
            prep_data_final = np.concatenate(ori_data, axis=0)
        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        # plt.scatter(
        #     tsne_results[:anal_sample_no, 0],
        #     tsne_results[:anal_sample_no, 1],
        #     c=colors[:anal_sample_no],
        #     alpha=0.2,
        #     label="Original",
        # )
        plt.scatter(
            tsne_results[:anal_sample_no, 0][data1],
            tsne_results[:anal_sample_no, 1][data1],
            c='tab:blue',
            alpha=0.2,
            label="Original data1",
        )
        plt.scatter(
            tsne_results[:anal_sample_no, 0][data2],
            tsne_results[:anal_sample_no, 1][data2],
            c='tab:green',
            alpha=0.2,
            label="Original data2",
        )
        if generated_data is not None:
            plt.scatter(
                tsne_results[anal_sample_no:, 0],
                tsne_results[anal_sample_no:, 1],
                c=colors[anal_sample_no:],
                alpha=0.2,
                label="Synthetic",
            )
        if out_list is not None:
            outlier = np.array([tsne_results[i, :] for i in out_list])
            plt.scatter(
                outlier[:, 0],
                outlier[:, 1],
                c="black",
                alpha=1,
                marker="D",
                label="Outlier",
            )

        ax.legend()

        plt.title("t-SNE plot")
        plt.xlabel("x-tsne")
        plt.ylabel("y_tsne")
        plt.savefig('/home/veos/devel/funes/visualization_results/tsne'+'-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.svg', dpi=300)
        plt.show()

    elif analysis == "umap":

        # UMAP analysis
        umap_results = umap.UMAP(n_neighbors=10, min_dist=0.1).fit_transform(prep_data)

        # Plotting
        # plt.scatter(
        #     umap_results[:, 0],
        #     umap_results[:, 1],
        #     c=colors[:anal_sample_no],
        #     alpha=0.2,
        #     label="Original",
        # )
        plt.scatter(
            umap_results[:, 0][data1],
            umap_results[:, 1][data1],
            c='tab:blue',
            alpha=0.2,
            label="Original data1",
        )
        plt.scatter(
            umap_results[:, 0][data2],
            umap_results[:, 1][data2],
            c='tab:green',
            alpha=0.2,
            label="Original data2",
        )

        if generated_data is not None:
            umap_hat_results = umap.UMAP(n_neighbors=10, min_dist=0.1).fit_transform(
                prep_data_hat
            )
            plt.scatter(
                umap_hat_results[:, 0],
                umap_hat_results[:, 1],
                c=colors[anal_sample_no:],
                alpha=0.2,
                label="Synthetic",
            )
        if out_list is not None:
            outlier = np.array([umap_results[i, :] for i in out_list])
            plt.scatter(
                outlier[:, 0],
                outlier[:, 1],
                c="black",
                alpha=1,
                marker="D",
                label="Outlier",
            )
        plt.legend()
        plt.title("umap plot")
        plt.xlabel("x-umap")
        plt.ylabel("y_umap")
        plt.savefig('/home/veos/devel/funes/visualization_results/umap'+'-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.svg',dpi=300)
        plt.show()


def visualization(ori_data, analysis, out_list=None, generated_data=None):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      - analysis: tsne or pca
    """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([3000, len(ori_data)]) # 3000
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]


    # Data preprocessing
    ori_data = np.asarray(ori_data)
    ori_data = ori_data[idx]
    if generated_data is not None:
        generated_data = np.asarray(generated_data)
        generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if i == 0:
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            if generated_data is not None:
                prep_data_hat = np.reshape(
                    np.mean(generated_data[0, :, :], 1), [1, seq_len]
                )
        else:
            prep_data = np.concatenate(
                (prep_data, np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len]))
            )
            if generated_data is not None:
                prep_data_hat = np.concatenate(
                    (
                        prep_data_hat,
                        np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len]),
                    )
                )

    # Visualization parameter
    colors = ["tab:blue" for i in range(anal_sample_no)] + [
        "tab:orange" for i in range(anal_sample_no)
    ]
    if analysis == "pca":
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(
            pca_results[:, 0],
            pca_results[:, 1],
            c=colors[:anal_sample_no],
            alpha=0.2,
            label="Original",
        )
        if generated_data is not None:
            pca_hat_results = pca.transform(prep_data_hat)
            plt.scatter(
                pca_hat_results[:, 0],
                pca_hat_results[:, 1],
                c=colors[anal_sample_no:],
                alpha=0.2,
                label="Synthetic",
            )
        if out_list is not None:
            outlier = np.array([pca_results[i, :] for i in out_list])
            plt.scatter(
                outlier[:, 0],
                outlier[:, 1],
                c="black",
                alpha=1,
                marker="D",
                label="Outlier",
            )

        ax.legend()
        plt.title("PCA plot")
        plt.xlabel("x-pca")
        plt.ylabel("y_pca")
        # plt.savefig('/home/veos/devel/funes/visualization_results/pca'+'-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.svg', dpi=300)
        plt.show()

    elif analysis == "tsne":
        if generated_data is not None:
            # Do t-SNE Analysis together
            prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
        else:
            prep_data_final = np.concatenate(ori_data, axis=0)
        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(
            tsne_results[:anal_sample_no, 0],
            tsne_results[:anal_sample_no, 1],
            c=colors[:anal_sample_no],
            alpha=0.2,
            label="Original",
        )
        if generated_data is not None:
            plt.scatter(
                tsne_results[anal_sample_no:, 0],
                tsne_results[anal_sample_no:, 1],
                c=colors[anal_sample_no:],
                alpha=0.2,
                label="Synthetic",
            )
        if out_list is not None:
            outlier = np.array([tsne_results[i, :] for i in out_list])
            plt.scatter(
                outlier[:, 0],
                outlier[:, 1],
                c="black",
                alpha=1,
                marker="D",
                label="Outlier",
            )

        ax.legend()

        plt.title("t-SNE plot")
        plt.xlabel("x-tsne")
        plt.ylabel("y_tsne")
        # plt.savefig('/home/veos/devel/funes/visualization_results/tsne'+'-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.svg', dpi=300)
        plt.show()

    elif analysis == "umap":

        # UMAP analysis
        umap_results = umap.UMAP(n_neighbors=10, min_dist=0.1).fit_transform(prep_data)

        # Plotting
        plt.scatter(
            umap_results[:, 0],
            umap_results[:, 1],
            c=colors[:anal_sample_no],
            alpha=0.2,
            label="Original",
        )
        if generated_data is not None:
            umap_hat_results = umap.UMAP(n_neighbors=10, min_dist=0.1).fit_transform(
                prep_data_hat
            )
            plt.scatter(
                umap_hat_results[:, 0],
                umap_hat_results[:, 1],
                c=colors[anal_sample_no:],
                alpha=0.2,
                label="Synthetic",
            )
        if out_list is not None:
            outlier = np.array([umap_results[i, :] for i in out_list])
            plt.scatter(
                outlier[:, 0],
                outlier[:, 1],
                c="black",
                alpha=1,
                marker="D",
                label="Outlier",
            )
        plt.legend()
        plt.title("umap plot")
        plt.xlabel("x-umap")
        plt.ylabel("y_umap")
        # plt.savefig('/home/veos/devel/funes/visualization_results/umap'+'-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.svg',dpi=300)
        plt.show()



def normalization(train_data,fake_data,method='min_max'):
    '''
        method: min_max or z-score (Gaussian distribution)
    '''

    num_features = train_data.shape[2]

    if method == 'min_max':
        for i in range(num_features):
            # train_mean = np.mean(train_data[:,:,i].flatten())
            # fake_mean = np.mean(fake_data[:,:,i].flatten())

            train_data[:, :, i] = (train_data[:, :, i] - min(train_data[:, :, i].flatten())) / (
                    max(train_data[:, :, i].flatten()) - min(train_data[:, :, i].flatten()))
            # print(max(fake_data[:, :, i].flatten()), min(fake_data[:, :, i].flatten()))
            fake_data[:, :, i] = (fake_data[:, :, i] - min(fake_data[:, :, i].flatten()) + 1e-6) / (
                    max(fake_data[:, :, i].flatten()) - min(fake_data[:, :, i].flatten()) + 1e-6)
    elif method == 'z_score':
        for i in range(num_features):
            train_mean = np.mean(train_data[:,:,i].flatten())
            fake_mean = np.mean(fake_data[:,:,i].flatten())
            train_std = np.std(train_data[:,:,i].flatten())
            fake_std = np.std(fake_data[:,:,i].flatten())

            train_data[:, :, i] = (train_data[:, :, i] - train_mean) / train_std
            # print(max(fake_data[:, :, i].flatten()), min(fake_data[:, :, i].flatten()))
            fake_data[:, :, i] = (fake_data[:, :, i] - fake_mean) / fake_std


def SOC():

    # /home/veos/devel/funes_test/data/output/teacher_forcing_len25_default_lr/SOC-20220831-093932/mdl/SOC-vanilla-1500-1500-5000-128-24-3-0.005-25-4000
    # train_data_path = '/home/veos/devel/funes_test/data/output/teacher_forcing_grid_search_len25_0901/SOC-20220905-002039/mdl/SOC-vanilla-2000-2000-4000-256-24-3-0.005-25-4000/train_data.pickle'
    # fake_data_path = '/home/veos/devel/funes_test/data/output/teacher_forcing_grid_search_len25_0901/SOC-20220905-002039/mdl/SOC-vanilla-2000-2000-4000-256-24-3-0.005-25-4000/fake_data.pickle'

    # train_data_path = '/home/veos/devel/funes_test/data/output/without_tf_grid_search_len25_default_lr_0901/SOC-20220904-024500/mdl/SOC-vanilla-2000-2000-4000-256-24-3-0.005-25-4000/train_data.pickle'
    # fake_data_path = '/home/veos/devel/funes_test/data/output/without_tf_grid_search_len25_default_lr_0901/SOC-20220904-024500/mdl/SOC-vanilla-2000-2000-4000-256-24-3-0.005-25-4000/fake_data.pickle'

    train_data_path = '/home/veos/devel/funes/data/output/split_0921/VOL-20220921-141259/mdl/VOL-vanilla-1000-1000-1000-128-24-3-0.001-100-4000/train_data.pickle'
    fake_data_path = '/home/veos/devel/funes/data/output/split_0921/VOL-20220921-141259/mdl/VOL-vanilla-1000-1000-1000-128-24-3-0.001-100-4000/fake_data.pickle'
    train_data = pd.read_pickle(train_data_path)
    fake_data = pd.read_pickle(fake_data_path)
    # print(train_data[:2])
    # print(train_data[:,:,1])
    # print(train_data.shape, fake_data.shape)

    # for file in sorted(os.listdir(file_path)):
    #     print(f'########### {file} ############')
    #     train_data = pd.read_pickle(train_data_path)
    #     fake_data = pd.read_pickle(fake_data_path)



    visualization(train_data, 'pca', generated_data=fake_data)
    visualization(train_data, 'tsne', generated_data=fake_data)
    visualization(train_data, 'umap', generated_data=fake_data)

    from scipy.fftpack import fft

    fft_new_data = fft(fake_data,axis=1)
    # print(f'{fake_data[0]}\n{fft_new_data[0]}')
    fft_ori_data = fft(train_data,axis=1)
    pic_num = 16
    index = 0
    index = index * pic_num
    # print('xxxxx',fft_ori_data[:, :, 1])

    ori_data_all = np.array(fft_ori_data[:, :, 2])
    figure, ax = plt.subplots()
    figure.suptitle('FFT Most similar cases')
    figure.set_size_inches(9, 6)
    figure.subplots_adjust(right=0.85)
    for j in range(1, pic_num + 1):
        # print(index)
        ori_data_case = ori_data_all[j + index]
        mae_list = []
        for i in range(ori_data_all.shape[0]):
            gen_data_case = np.array(fft_new_data[i, :, 2])
            # diff = np.mean(abs(ori_data_case - gen_data_case))
            diff = np.mean(abs(np.abs(ori_data_case) - np.abs(gen_data_case)))
            mae_list.append(diff)
        # print(max(mae_list), min(mae_list))
        # print(fake_data[mae_list.index(min(mae_list))])
        plt.subplot(4, 4, j)
        plt.ylim([0, 1])
        # plt.ylim([-5, 5])
        plt.plot(fake_data[mae_list.index(min(mae_list)), :, 2], label='fake data')
        plt.plot(train_data[j, :, 2], label='train data')
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # plt.savefig(
    #     '/home/veos/devel/funes/visualization_results/fft_most_similar_cases' + '-' + datetime.datetime.now().strftime(
    #         "%Y%m%d-%H%M%S") + '.svg', dpi=300)
    plt.show()

    ############# fft ################
    figure, ax = plt.subplots()
    figure.suptitle('FFT cases')
    figure.set_size_inches(9, 6)
    figure.subplots_adjust(right=0.8)
    for j in range(1, pic_num + 1):
        # print(index)
        ori_data_case = ori_data_all[j + index]
        mae_list = []
        for i in range(ori_data_all.shape[0]):
            gen_data_case = np.array(fft_new_data[i, :, 2])  # soc ocv
            diff = np.mean(abs(np.abs(ori_data_case) - np.abs(gen_data_case)))
            mae_list.append(diff)
        # print(max(mae_list), min(mae_list))
        # print(len(mae_list),min(mae_list))
        # print(fake_data[mae_list.index(min(mae_list))])
        plt.subplot(4, 4, j)
        plt.ylim([-2, 20])
        plt.plot(np.abs(fft_new_data[mae_list.index(min(mae_list)), :, 2]),
                 label='fft fake data')  # abs => amplitude  np.angle(fft_y) => phase
        # print(np.abs(fft_new_data[mae_list.index(min(mae_list)), :, 4]))
        # plt.plot(np.abs(fft_ori_data[mae_list.index(min(mae_list)), :, 4]),label='fft train data')
        plt.plot(np.abs(fft_ori_data[j, :, 2]), label='fft train data')
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # plt.savefig(
    #     '/home/veos/devel/funes/visualization_results/fft_cases' + '-' + datetime.datetime.now().strftime(
    #         "%Y%m%d-%H%M%S") + '.svg', dpi=300)
    plt.show()

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

import argparse
import torch
from funes.tgan.models.timegan import TimeGAN

def VOL_cross_correlation():
    data_path = '../../../data/output/vol_0928/VOL-20220930-133726/mdl/VOL-vanilla-1200-1200-1200-64-32-2-0.001-100-4000'
    def gen_fake(args):
        train_time = pd.read_pickle(f'{data_path}/train_time.pickle')
        train_data = pd.read_pickle(f'{data_path}/train_data.pickle')
        # Ts = pd.read_pickle(f"{data_path}/train_time.pickle")
        # print(type(Ts))
        print(pd.DataFrame(train_time).value_counts())
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

        model = TimeGAN(args)
        T = np.array([np.random.choice(T) for _ in range(1000)])
        T = torch.from_numpy(T)

        model.load_state_dict(torch.load(f"{data_path}/model.pt"))

        model.to(args.device)
        model.eval()
        with torch.no_grad():
            # Generate fake data
            Z = torch.rand((len(T), args.max_seq_len, args.Z_dim))

            # Z_ = []
            # for i in range(len(T)):
            #     tmp_t = list(np.ones(T[i])) + list(np.zeros(args.max_seq_len-T[i]))
            #     r = torch.squeeze(Z[i])*torch.tensor(tmp_t)
            #     Z_.append(r.unsqueeze(axis=-1).numpy())
            # Z_ = torch.from_numpy(np.array(Z_)).float()

            generated_data = model(X=None, T=T, Z=Z, obj="inference")
        fake_time = T

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

    # fake_data, train_data, fake_time, train_time = gen_fake(args)
    data_path = '/home/veos/devel/funes/data/output/vol_0928/VOL-20220930-133726/mdl/VOL-vanilla-1200-1200-1200-64-32-2-0.001-100-4000/'
    # data_path = '/home/veos/devel/funes_test/data/output/Sine_0916/Sine-20220916-191823/mdl/Sine-vanilla-80000-80000-80000-64-32-3-0.001-50-4000/'
    train_data_path = data_path + 'train_data.pickle'
    train_time_path = data_path + 'train_time.pickle'
    fake_data_path = data_path + 'fake_data.pickle'
    fake_time_path = data_path + 'fake_time.pickle'

    train_data = pd.read_pickle(train_data_path)
    fake_data = pd.read_pickle(fake_data_path)
    train_time = pd.read_pickle(train_time_path)
    fake_time = pd.read_pickle(fake_time_path)
    # print(train_data[:2])
    # print(fake_data[:2])

    # plt.plot(MinMaxScaler(train_data[4962,:,0]),'.')
    # plt.plot(MinMaxScaler(train_data[6559, :, 0]),'v')
    # plt.plot(MinMaxScaler(fake_data[2, :, 0]))
    # plt.show()

    # plt.plot(train_data[6559, :, 0])
    # plt.plot(fake_data[4, :, 0])
    # plt.show()

    # cor1 = signal.correlate(MinMaxScaler(train_data[4962,:,0]),MinMaxScaler(fake_data[2, :, 0]))
    # cor2 = signal.correlate(MinMaxScaler(train_data[6559,:,0]),MinMaxScaler(fake_data[2, :, 0]))
    # print(cor1,max(cor1))
    # print(cor2,max(cor2))

    # for i in range(len(train_data)):
    #     train_data[i][train_time[i]:]=train_data[i][train_time[i]-1]
    #     # print(train_data[i])
    #
    # for i in range(len(train_data)):
    #     fake_data[i][fake_time[i]:]=fake_data[i][fake_time[i]-1]
        # print('######',fake_data[i])

    # seed = 3
    # np.random.seed(seed)
    # np.random.shuffle(train_data)
    # np.random.seed(seed)
    # np.random.shuffle(fake_data)
    # np.random.seed(seed)
    # np.random.shuffle(fake_time)
    # np.random.seed(seed)
    # np.random.shuffle(train_time)


    from scipy.fftpack import fft, ifft

    fft_new_data = []
    fft_ori_data = []

    for i in range(fake_data.shape[0]):
        fft_new_data.append(fft(fake_data[i][:fake_time[i]], axis=0))

    for i in range(train_data.shape[0]):
        fft_ori_data.append(fft(train_data[i][:train_time[i]], axis=0))
    fft_new_data = np.array(fft_new_data)
    fft_ori_data = np.array(fft_ori_data)

    # fft_new_data = fft(fake_data, axis=1)
    # fft_ori_data = fft(train_data, axis=1)

    front = 2
    tail = -1
    pic_num = 16
    index = 0
    index = index * pic_num
    # print('xxxxx',fft_ori_data[:, :, 1])

    gen_data_all = fft_new_data
    figure, ax = plt.subplots()
    figure.suptitle('FFT Most similar cases')
    figure.set_size_inches(9, 6)
    figure.subplots_adjust(right=0.85)

    for j in range(1, pic_num + 1):
        gen_data_case = gen_data_all[j + index]
        mae_list = []
        cor_list = []
        max_idx_list = []
        for i in range(fft_ori_data.shape[0]):

            # if train_time[i] == fake_time[j]:
            cor = signal.correlate(MinMaxScaler(train_data[i, :train_time[i], 0]),MinMaxScaler(fake_data[j, :fake_time[j], 0]))  ##########################
            # cor = MinMaxScaler(cor)
            # print(i,list(cor).index(max(cor)),cor.argmax(),len(cor))
            # print(train_time[i],fake_time[j])

            cor_list.append(max(cor))
            max_idx_list.append(cor.argmax())
            # print(len(cor),train_time[i],fake_time[j])

            # print('cor:',max(cor),cor)
            # else:
            #     cor_list.append(-1e5)

            # print('####',scipy.spatial.distance.correlation(train_data[i, :, 0],
            #                                    fake_data[j, :, 0]))
            # diff = scipy.spatial.distance.correlation(train_data[i, :, 0],fake_data[j, :, 0])
            # mae_list.append(diff)

            # if len(gen_data_case) == len(fft_ori_data[i]):
            #     ori_data_case = np.array(fft_ori_data[i])
            #     diff = np.mean(abs(np.abs(ori_data_case) - np.abs(gen_data_case)))
            #     mae_list.append(diff)
            # else:
            #     mae_list.append(1e+5)

        plt.subplot(4, 4, j)
        min_y = min(min(fake_data[j, :fake_time[j], 0]),min(train_data[cor_list.index(max(cor_list)), :train_time[cor_list.index(max(cor_list))], 0]))
        max_y = max(max(fake_data[j, :fake_time[j], 0]),max(train_data[cor_list.index(max(cor_list)), :train_time[cor_list.index(max(cor_list))], 0]))
        plt.ylim([min_y, max_y])
        # plt.ylim([min(fake_data[j, :fake_time[j], 0]), max(fake_data[j, :fake_time[j], 0])])
        # plt.plot(train_data[mae_list.index(min(mae_list)), :train_time[mae_list.index(min(mae_list))], 0],
        #          label='train data', linewidth='2.5')
        plt.plot(train_data[cor_list.index(max(cor_list)), :train_time[cor_list.index(max(cor_list))], 0],
                 label='train data', linewidth='2.5')

        max_val_idx = cor_list.index(max(cor_list))
        # print(max_val_idx)
        print(max_val_idx,max_idx_list[max_val_idx],fake_time[j],train_time[max_val_idx])
        # print(cor_list[max_val_idx-5:max_val_idx+5])
        # print(lags_list[max_val_idx])
        # [_ for _ in range(cor.argmax() - 30 + 1, cor.argmax() + 1)]
        start = max_idx_list[max_val_idx]-fake_time[j]+1
        end = max_idx_list[max_val_idx]+1
        x = [_ for _ in range(start,end)]
        # print(end,start)
        print(x)
        plt.plot(fake_data[j, :fake_time[j], 0], label='fake data 1')
        if sum(np.array(x)<0) == 0:
            plt.plot(x,fake_data[j, :fake_time[j], 0], label='fake data 2')
        else:
            plt.plot(fake_data[j, :fake_time[j], 0], label='fake data 2')

    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # plt.savefig(
    #     '/home/veos/devel/funes/visualization_results/fft_most_similar_cases' + '-' + datetime.datetime.now().strftime(
    #         "%Y%m%d-%H%M%S") + '.svg', dpi=300)
    # plt.show()

    ############# fft ################
    # figure, ax = plt.subplots()
    # figure.suptitle('FFT cases')
    # figure.set_size_inches(9, 6)
    # figure.subplots_adjust(right=0.8)
    # for j in range(1, pic_num + 1):
    #     gen_data_case = gen_data_all[j + index]
    #     mae_list = []
    #     cor_list = []
    #     max_idx_list = []
    #     for i in range(fft_ori_data.shape[0]):
    #         # diff = scipy.spatial.distance.correlation(train_data[i, :, 0], fake_data[j, :, 0])
    #         # mae_list.append(diff)
    #
    #         # if train_time[i]==fake_time[j]:
    #         cor = signal.correlate(train_data[i, :train_time[i], 0], fake_data[j, :fake_time[j], 0])
    #         cor_list.append(max(cor))
    #         max_idx_list.append(cor.argmax())
    #         # print(cor.shape,cor.argmax(),cor.tolist().index(max(cor)))
    #         # else:
    #         #     cor_list.append(-1e5)
    #
    #         # if len(gen_data_case) == len(fft_ori_data[i]):
    #         #     ori_data_case = np.array(fft_ori_data[i])
    #         #     diff = np.mean(abs(np.abs(ori_data_case) - np.abs(gen_data_case)))
    #         #     mae_list.append(diff)
    #         # else:
    #         #     mae_list.append(1e+5)
    #
    #     plt.subplot(4, 4, j)
    #
    #     min_y = min(min(np.abs(fft_new_data[j][front:tail])),
    #                 min(np.abs(fft_ori_data[cor_list.index(max(cor_list))][front:tail])))
    #     max_y = max(max(np.abs(fft_new_data[j][front:tail])),
    #                 max(np.abs(fft_ori_data[cor_list.index(max(cor_list))][front:tail])))
    #     plt.ylim([min_y, max_y])
    #     # plt.ylim([min(np.abs(fft_new_data[j][front:tail])), max(np.abs(fft_new_data[j][front:tail]))])
    #     # plt.plot(np.abs(fft_ori_data[mae_list.index(min(mae_list))][front:tail]), label='fft train data',
    #     #          lw=2.5)  # abs => amplitude  np.angle(fft_y) => phase
    #     plt.plot(np.abs(fft_ori_data[cor_list.index(max(cor_list))][front:tail]), label='fft train data',
    #              lw=2.5)
    #     plt.plot(np.abs(fft_new_data[j][front:tail]), label='fft fake data')
    # plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # plt.savefig(
    #     '/home/veos/devel/funes/visualization_results/fft_cases' + '-' + datetime.datetime.now().strftime(
    #         "%Y%m%d-%H%M%S") + '.svg', dpi=300)
    plt.show()

def mixed_wave():
    data_path = '../../../data/output/sine+tri_1202/MIX-20221205-045631/mdl/MIX-vanilla-1500-1500-1500-64-16-2-4-0.001-50-4000'

    # sine+tri_1021/MIX-20221024-141242_best/mdl/MIX-vanilla-1000-1000-1000-64-24-3-0.001-50-4000 best
    # MIX-20221023-155237/mdl/MIX-vanilla-1000-1000-1000-128-32-3-0.001-50-4000 retrain


    def gen_wave(args):
        train_time = pd.read_pickle(f'{data_path}/train_time.pickle')
        train_data = pd.read_pickle(f'{data_path}/train_data.pickle')
        # print(train_data.shape)

        T = np.array([50 for _ in range(train_data.shape[0])])
        fake_time = np.array(T)

        model = TimeGAN(args)

        model.load_state_dict(torch.load(f"{data_path}/model.pt"))

        # print("\nGenerating Data...")
        # Initialize model to evaluation mode and run without gradients
        model.to(args.device)
        model.eval()
        with torch.no_grad():
            # Generate fake data
            Z = torch.rand((len(T), args.max_seq_len, args.Z_dim))
            # print(args.max_seq_len,Z.shape)
            generated_data = model(X=None, T=T, Z=Z, obj="inference")

        return generated_data.numpy(), train_data, fake_time, train_time

    params = data_path.split('/')[-1].split('-')
    # print(params)

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", type=str)
    parser.add_argument("--gradient_type", default="vanilla", type=str)
    parser.add_argument("--batch_size", default=params[-7], type=int)
    parser.add_argument("--hidden_dim", default=params[-6], type=int)
    parser.add_argument("--num_layers", default=params[-5], type=int)
    parser.add_argument("--feature_dim", default=1, type=int)
    parser.add_argument("--Z_dim", default=1, type=int)
    parser.add_argument("--max_seq_len", default=params[-2], type=int)
    parser.add_argument("--padding_value", default=0, type=float)
    # parser.add_argument("--layer_norm_gru", default=False, type=bool)
    parser.add_argument("--heads", default=params[-4], type=int)
    parser.add_argument("--encode_mode", default='transformer', type=str)
    # if len(params)==12:
    #     parser.add_argument("--max_seq_len", default=params[10], type=int)
    # if len(params)==11:
    #     parser.add_argument("--max_seq_len", default=params[9], type=int)
    # parser.add_argument("--padding_value", default=0, type=float)

    args = parser.parse_args()

    fake_data, train_data, fake_time, train_time = gen_wave(args)


    mixed_visualization(train_data, 'pca', generated_data=fake_data)
    mixed_visualization(train_data, 'tsne', generated_data=fake_data)
    mixed_visualization(train_data, 'umap', generated_data=fake_data)

    from scipy.fftpack import fft

    # fft_new_data = fft(fake_data, axis=1)
    # fft_ori_data = fft(train_data, axis=1)


    cut = 1

    # print(fft_ori_data)
    pic_num = 16

    # gen_data_all = np.array(fft_new_data[:, :, 4])  # soc ocv
    figure, ax = plt.subplots()
    figure.suptitle('Most similar cases')
    figure.set_size_inches(9, 6)
    figure.subplots_adjust(right=0.85)
    for j in range(1, pic_num + 1):
        # gen_data_case = gen_data_all[j, cut:]
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

        plt.subplot(4, 4, j)
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
    plt.savefig(
        '/home/veos/devel/funes/visualization_results/multi_most_similar_cases' + '-' + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S") + '.svg', dpi=300)
    # plt.show()

    ############# fft ################
    # figure, ax = plt.subplots()
    # figure.suptitle('FFT cases')
    # figure.set_size_inches(9, 6)
    # figure.subplots_adjust(right=0.8)
    # for j in range(1, pic_num + 1):
    #     # print(index)
    #     gen_data_case = gen_data_all[j, cut:]
    #     mae_list = []
    #     for i in range(fft_ori_data.shape[0]):
    #         ori_data_case = np.array(fft_ori_data[i, cut:, 4])  # soc ocv
    #         diff = np.mean(abs(np.abs(ori_data_case) - np.abs(gen_data_case)))
    #         mae_list.append(diff)
    #     # print(max(mae_list), min(mae_list))
    #     # print(len(mae_list),min(mae_list))
    #     # print(fake_data[mae_list.index(min(mae_list))])
    #     plt.subplot(5, 5, j)
    #     plt.ylim([min(np.abs(fft_new_data[j, :, 0])), max(np.abs(fft_new_data[j, :, 0]))])
    #     plt.plot(np.abs(fft_ori_data[mae_list.index(min(mae_list)), cut:, 4]),
    #              label='fft train data', lw=2.5)  # abs => amplitude  np.angle(fft_y) => phase
    #     # print(np.abs(fft_new_data[mae_list.index(min(mae_list)), :, 4]))
    #     # plt.plot(np.abs(fft_ori_data[mae_list.index(min(mae_list)), :, 4]),label='fft train data')
    #     plt.plot(np.abs(fft_new_data[j, cut:, 4]), label='fft fake data')
    # plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # # plt.savefig(
    # #     '/home/veos/devel/funes/visualization_results/fft_cases' + '-' + datetime.datetime.now().strftime(
    # #         "%Y%m%d-%H%M%S") + '.svg', dpi=300)
    plt.show()

def single_wave():

    # data_path = '../../../data/output/ttsgan_sine_1202/Sine-20221205-021322/mdl/Sine-vanilla-1000-1000-1000-64-32-3-4-0.001-50-4000'
    data_path = '../../../data/output/sine_1206/Sine-20221207-163456/mdl/Sine-vanilla-1000-1000-1000-64-32-3-4-10-50-4000'

    def gen_wave(args):

        train_time = pd.read_pickle(f'{data_path}/train_time.pickle')
        train_data = pd.read_pickle(f'{data_path}/train_data.pickle')

        T = np.array([50 for _ in range(train_data.shape[0])])
        fake_time = np.array(T)

        model = TimeGAN(args)

        model.load_state_dict(torch.load(f"{data_path}/model.pt"))

        # print("\nGenerating Data...")
        # Initialize model to evaluation mode and run without gradients
        model.to(args.device)
        model.eval()
        with torch.no_grad():
            # Generate fake data
            Z = torch.rand((len(T), args.max_seq_len, args.Z_dim))

            generated_data = model(X=None, T=T, Z=Z, obj="inference")

        return generated_data.numpy(), train_data,fake_time, train_time

    params = data_path.split('/')[-1].split('-')

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", type=str)
    parser.add_argument("--gradient_type", default="vanilla", type=str)
    parser.add_argument("--batch_size", default=params[-7], type=int)
    parser.add_argument("--hidden_dim", default=params[-6], type=int)
    parser.add_argument("--num_layers", default=params[-5], type=int)
    parser.add_argument("--feature_dim", default=1, type=int)
    parser.add_argument("--Z_dim", default=1, type=int)
    parser.add_argument("--max_seq_len", default=params[-2], type=int)
    parser.add_argument("--padding_value", default=0, type=float)
    # parser.add_argument("--layer_norm_gru", default=False, type=bool)
    parser.add_argument("--heads", default=params[-4], type=int)
    parser.add_argument("--encode_mode", default='transformer', type=str)


    args = parser.parse_args()

    fake_data, train_data, fake_time, train_time = gen_wave(args)
    print(train_data.shape,fake_data.shape)

    visualization(train_data, 'pca', generated_data=fake_data)
    visualization(train_data, 'tsne', generated_data=fake_data)
    visualization(train_data, 'umap', generated_data=fake_data)

    from scipy.fftpack import fft

    # fft_new_data = fft(fake_data, axis=1)
    # fft_ori_data = fft(train_data, axis=1)


    cut = 1

    # print(fft_ori_data)
    pic_num = 16

    # gen_data_all = np.array(fft_new_data[:, :, 4])  # soc ocv
    figure, ax = plt.subplots()
    figure.suptitle('Most similar cases')
    figure.set_size_inches(9, 6)
    figure.subplots_adjust(right=0.85)
    for j in range(1, pic_num + 1):
        # gen_data_case = gen_data_all[j, cut:]
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

        plt.subplot(4, 4, j)
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
    # plt.savefig(
    #     '/home/veos/devel/funes/visualization_results/fft_most_similar_cases' + '-' + datetime.datetime.now().strftime(
    #         "%Y%m%d-%H%M%S") + '.svg', dpi=300)
    # plt.show()

    ############# fft ################
    # figure, ax = plt.subplots()
    # figure.suptitle('FFT cases')
    # figure.set_size_inches(9, 6)
    # figure.subplots_adjust(right=0.8)
    # for j in range(1, pic_num + 1):
    #     # print(index)
    #     gen_data_case = gen_data_all[j, cut:]
    #     mae_list = []
    #     for i in range(fft_ori_data.shape[0]):
    #         ori_data_case = np.array(fft_ori_data[i, cut:, 4])  # soc ocv
    #         diff = np.mean(abs(np.abs(ori_data_case) - np.abs(gen_data_case)))
    #         mae_list.append(diff)
    #     # print(max(mae_list), min(mae_list))
    #     # print(len(mae_list),min(mae_list))
    #     # print(fake_data[mae_list.index(min(mae_list))])
    #     plt.subplot(5, 5, j)
    #     plt.ylim([min(np.abs(fft_new_data[j, :, 0])), max(np.abs(fft_new_data[j, :, 0]))])
    #     plt.plot(np.abs(fft_ori_data[mae_list.index(min(mae_list)), cut:, 4]),
    #              label='fft train data', lw=2.5)  # abs => amplitude  np.angle(fft_y) => phase
    #     # print(np.abs(fft_new_data[mae_list.index(min(mae_list)), :, 4]))
    #     # plt.plot(np.abs(fft_ori_data[mae_list.index(min(mae_list)), :, 4]),label='fft train data')
    #     plt.plot(np.abs(fft_new_data[j, cut:, 4]), label='fft fake data')
    # plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # # plt.savefig(
    # #     '/home/veos/devel/funes/visualization_results/fft_cases' + '-' + datetime.datetime.now().strftime(
    # #         "%Y%m%d-%H%M%S") + '.svg', dpi=300)
    plt.show()

def VOL():
    data_path = '../../../data/output/vol_1202/VOL-20221205-085316/mdl/VOL-vanilla-1000-1000-1000-128-8-3-4-0.001-50-4000'
    # data_path = '../../../data/output/baiduyun/VOL-2022-1027/VOL-20221028-050105/mdl/VOL-vanilla-1000-1000-1000-128-32-3-0.001-100-4000'


    def gen_fake(args):
        train_time = pd.read_pickle(f'{data_path}/train_time.pickle')
        train_data = pd.read_pickle(f'{data_path}/train_data.pickle')
        # Ts = pd.read_pickle(f"{data_path}/train_time.pickle")
        # print(type(Ts))
        print(pd.DataFrame(train_time).value_counts())
        print(len(set(train_time)))
        if len(set(train_time)) != 1:
            Ts = pd.DataFrame(train_time).value_counts().items()
            T = []
            for val, count in Ts:
                T.append(val[0])

            # print(T[:int(0.3 * len(T))], 0.3 * len(T))
            # T = T[:int(0.3 * len(T))]

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
    parser.add_argument("--batch_size", default=params[-7], type=int)
    parser.add_argument("--hidden_dim", default=params[-6], type=int)
    parser.add_argument("--num_layers", default=params[-5], type=int)
    parser.add_argument("--feature_dim", default=1, type=int)
    parser.add_argument("--Z_dim", default=1, type=int)
    parser.add_argument("--max_seq_len", default=params[-2], type=int)
    parser.add_argument("--padding_value", default=0, type=float)
    # parser.add_argument("--layer_norm_gru", default=False, type=bool)
    parser.add_argument("--heads", default=params[-4], type=int)
    parser.add_argument("--encode_mode", default='transformer', type=str)

    args = parser.parse_args()

    fake_data, train_data, fake_time, train_time = gen_fake(args)

    visualization(train_data, 'pca', generated_data=fake_data)
    visualization(train_data, 'tsne', generated_data=fake_data)
    visualization(train_data, 'umap', generated_data=fake_data)

    # data_path = '/home/veos/devel/funes/data/output/vol_0928/VOL-20220930-133726/mdl/VOL-vanilla-1200-1200-1200-64-32-2-0.001-100-4000/'
    # data_path = '/home/veos/devel/funes_test/data/output/Sine_0916/Sine-20220916-191823/mdl/Sine-vanilla-80000-80000-80000-64-32-3-0.001-50-4000/'
    # train_data_path = data_path + 'train_data.pickle'
    # train_time_path = data_path + 'train_time.pickle'
    # fake_data_path = data_path + 'fake_data.pickle'
    # fake_time_path = data_path + 'fake_time.pickle'
    #
    # train_data = pd.read_pickle(train_data_path)
    # fake_data = pd.read_pickle(fake_data_path)
    # train_time = pd.read_pickle(train_time_path)
    # fake_time = pd.read_pickle(fake_time_path)
    # # print(train_data[:2])
    # # print(fake_data[:2])
    #
    # seed = 3
    # np.random.seed(seed)
    # np.random.shuffle(train_data)
    # np.random.seed(seed)
    # np.random.shuffle(fake_data)
    # np.random.seed(seed)
    # np.random.shuffle(fake_time)
    # np.random.seed(seed)
    # np.random.shuffle(train_time)

    from scipy.fftpack import fft, ifft

    # fft_new_data = []
    # fft_ori_data = []
    #
    # for i in range(fake_data.shape[0]):
    #     fft_new_data.append(fft(fake_data[i][:fake_time[i]], axis=0))
    #
    # for i in range(train_data.shape[0]):
    #     fft_ori_data.append(fft(train_data[i][:train_time[i]], axis=0))
    # fft_new_data = np.array(fft_new_data)
    # fft_ori_data = np.array(fft_ori_data)

    # fft_new_data = fft(fake_data, axis=1)
    # fft_ori_data = fft(train_data, axis=1)

    front = 2
    tail = -1
    pic_num = 16

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
                diff = np.mean(abs(train_data[i,:train_time[i],0] - fake_data[j,:fake_time[j],0]))
                mae_list.append(diff)
                idx_list.append([0,fake_time[j]])
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
                idx_list.append([idx,idx+fake_time[j]])
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

        plt.subplot(4, 4, j)
        min_y = min(min(fake_data[j, :fake_time[j], 0]),
                    min(train_data[mae_list.index(min(mae_list)), :train_time[mae_list.index(min(mae_list))], 0]))
        max_y = max(max(fake_data[j, :fake_time[j], 0]),
                    max(train_data[mae_list.index(min(mae_list)), :train_time[mae_list.index(min(mae_list))], 0]))
        plt.ylim([min_y, max_y])

        match_idx = idx_list[mae_list.index(min(mae_list))]
        x = [_ for _ in range(match_idx[0],match_idx[1])]
        if train_time[mae_list.index(min(mae_list))] >= fake_time[j]:
            plt.plot(train_data[mae_list.index(min(mae_list)), :train_time[mae_list.index(min(mae_list))], 0],
                     label='train data', linewidth='2.5')
            plt.plot(x,fake_data[j, :fake_time[j], 0], label='fake data')
        else:
            plt.plot(x,train_data[mae_list.index(min(mae_list)), :train_time[mae_list.index(min(mae_list))], 0],
                     label='train data', linewidth='2.5')
            plt.plot(fake_data[j, :fake_time[j], 0], label='fake data')

    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.savefig(
        '/home/veos/devel/funes/visualization_results/most_similar_cases' + '-' + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S") + '.svg', dpi=300)
    # plt.show()

    ############# fft ################
    # figure, ax = plt.subplots()
    # figure.suptitle('FFT cases')
    # figure.set_size_inches(9, 6)
    # figure.subplots_adjust(right=0.8)
    # for j in range(1, pic_num + 1):
    #     # print(index)
    #     gen_data_case = gen_data_all[j + index]
    #     mae_list = []
    #     for i in range(fft_ori_data.shape[0]):
    #         if len(gen_data_case) == len(fft_ori_data[i]):
    #             ori_data_case = np.array(fft_ori_data[i])
    #             diff = np.mean(abs(np.abs(ori_data_case) - np.abs(gen_data_case)))
    #             mae_list.append(diff)
    #         else:
    #             mae_list.append(1e+5)
    #     # M2.append([max(mae_list), mae_list.index(min(mae_list))])
    #     # print(len(mae_list),min(mae_list))
    #     plt.subplot(4, 4, j)
    #     plt.ylim([min(np.abs(fft_new_data[j][front:tail])), max(np.abs(fft_new_data[j][front:tail]))])
    #     plt.plot(np.abs(fft_ori_data[mae_list.index(min(mae_list))][front:tail]), label='fft train data',
    #              lw=2.5)  # abs => amplitude  np.angle(fft_y) => phase
    #     # plt.plot(np.abs(fft_ori_data[mae_list.index(min(mae_list)), :, 4]),label='fft train data')
    #     plt.plot(np.abs(fft_new_data[j][front:tail]), label='fft fake data')
    #
    # plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # # plt.savefig(
    # #     '/home/veos/devel/funes/visualization_results/fft_cases' + '-' + datetime.datetime.now().strftime(
    # #         "%Y%m%d-%H%M%S") + '.svg', dpi=300)
    plt.show()


def VOL_metric():
    data_path = '../../../data/output/vol_gen_1027/VOL-20221106-121653/mdl/VOL-vanilla-1200-1200-1200-128-64-3-0.001-100-4000'
    # data_path = '../../../data/output/baiduyun/VOL-2022-1027/VOL-20221028-050105/mdl/VOL-vanilla-1000-1000-1000-128-32-3-0.001-100-4000'


    def gen_fake(args):
        train_time = pd.read_pickle(f'{data_path}/train_time.pickle')
        train_data = pd.read_pickle(f'{data_path}/train_data.pickle')
        # Ts = pd.read_pickle(f"{data_path}/train_time.pickle")
        # print(type(Ts))
        print(pd.DataFrame(train_time).value_counts())
        print(len(set(train_time)))
        if len(set(train_time)) != 1:
            Ts = pd.DataFrame(train_time).value_counts().items()
            T = []
            for val, count in Ts:
                T.append(val[0])

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

    fake_data, train_data, fake_time, train_time = gen_fake(args)
    print(fake_data.shape,train_data.shape)

    visualization(train_data, 'pca', generated_data=fake_data)
    visualization(train_data, 'tsne', generated_data=fake_data)
    visualization(train_data, 'umap', generated_data=fake_data)

    front = 2
    tail = -1
    pic_num = 16

    # gen_data_all = fft_new_data
    figure, ax = plt.subplots()
    figure.suptitle('Most similar cases')
    figure.set_size_inches(9, 6)
    figure.subplots_adjust(right=0.85)

    MAE = []
    IDX = []
    # ori_index = []
    fake_index = []
    LOSS = []
    # for j in range(fake_data.shape[0]):
    for j in range(train_data.shape[0]):

        # gen_data_case = gen_data_all[j + index]
        mae_list = []
        idx_list = []

        # for i in range(train_data.shape[0]):
        for i in range(fake_data.shape[0]):

            if fake_time[j] == train_time[i]:
                # ori_data_case = np.array(fft_ori_data[i])
                # diff = np.mean(abs(train_data[i, :train_time[i], 0] - fake_data[j, :fake_time[j], 0]))
                diff = np.mean(abs(train_data[j, :train_time[j], 0] - fake_data[i, :fake_time[i], 0]))
                mae_list.append(diff)
                # idx_list.append([0, fake_time[j]])
                idx_list.append([0, train_time[j]])
            # elif fake_time[j] < train_time[i]:
            elif train_time[j] < fake_time[i]:
                left = 0
                # right = fake_time[j]
                right = train_time[j]
                diff = []
                # while right <= train_time[i]:
                while right <= fake_time[i]:

                    # d = np.mean(abs(train_data[i, left:right, 0] - fake_data[j, :fake_time[j], 0]))
                    d = np.mean(abs(train_data[j, left:right, 0] - fake_data[i, :fake_time[i], 0]))
                    diff.append(d)
                    left += 1
                    right += 1
                mae_list.append(min(diff))
                idx = diff.index(min(diff))
                # idx_list.append([idx, idx + fake_time[j]])
                idx_list.append([idx, idx + train_time[j]])
            # elif fake_time[j] > train_time[i]:
            elif train_time[j] > fake_time[i]:
                left = 0
                # right = train_time[i]
                right = fake_time[i]
                diff = []
                # while right <= fake_time[j]:
                while right <= train_time[j]:
                    # d = np.mean(abs(train_data[i, :train_time[i], 0] - fake_data[j, left:right, 0]))
                    d = np.mean(abs(train_data[j, :train_time[j], 0] - fake_data[i, left:right, 0]))
                    diff.append(d)
                    left += 1
                    right += 1
                mae_list.append(min(diff))
                idx = diff.index(min(diff))
                # idx_list.append([idx, idx + train_time[i]])
                idx_list.append([idx, idx + fake_time[i]])
        MAE.append(min(mae_list))
        IDX.append(idx_list[mae_list.index(min(mae_list))])
        # ori_index.append(mae_list.index(min(mae_list)))
        fake_index.append(mae_list.index(min(mae_list)))
        LOSS.append(np.mean(mae_list))


    # sorted_data = sorted(MAE)[:pic_num]
    sorted_data = sorted(MAE)[-pic_num:]
    # sorted_ori_idx = [ori_index[MAE.index(i)] for i in sorted_data]
    sorted_fake_idx = [fake_index[MAE.index(i)] for i in sorted_data]
    print(sorted_data)
    # print(sorted(MAE))
    # print(sorted_fake_idx)
    # print(sorted_ori_idx) # [1134, 1134, 1134, 1134, 1134, 1134, 1134, 1134, 1134, 1134, 1134, 1134, 1134, 1134, 1134, 1134]
    # print([MAE.index(i) for i in sorted_data]) # [2005, 1016, 5708, 3132, 7321, 992, 3195, 3131, 4688, 5952, 5904, 7529, 7483, 4253, 2513, 2491]
    print(len(LOSS))
    print(f'LOSS: {np.mean(LOSS)}') # LOSS: 0.030877381086350915
    for p in range(pic_num):
        # fake_idx = MAE.index(sorted_data[p])
        ori_idx = MAE.index(sorted_data[p])
        # ori_idx = sorted_ori_idx[p]
        fake_idx = sorted_fake_idx[p]
        # print(fake_data[fake_idx, :fake_time[fake_idx],0])
        plt.subplot(4, 4, p+1)
        min_y = min(min(fake_data[fake_idx, :fake_time[fake_idx], 0]),
                    min(train_data[ori_idx, :train_time[ori_idx], 0]))
        max_y = max(max(fake_data[fake_idx, :fake_time[fake_idx], 0]),
                    max(train_data[ori_idx, :train_time[ori_idx], 0]))
        plt.ylim([min_y, max_y])

        match_idx = IDX[fake_idx]
        x = [_ for _ in range(match_idx[0], match_idx[1])]
        if train_time[ori_idx] >= fake_time[fake_idx]:
            plt.plot(train_data[ori_idx, :train_time[ori_idx], 0],
                     label='train data', linewidth='2.5')
            plt.plot(x, fake_data[fake_idx, :fake_time[fake_idx], 0], label='fake data')
        else:
            plt.plot(x, train_data[ori_idx, :train_time[ori_idx], 0],
                     label='train data', linewidth='2.5')
            plt.plot(fake_data[fake_idx, :fake_time[fake_idx], 0], label='fake data')

    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.show()


if __name__ == "__main__":
    # start = time.time()
    # VOL_test() # 7.099074486891428 mins 8000
    # print((time.time()-start)/60,'mins')
    single_wave()
    # dct_single_wave()
    # mixed_wave()
    # VOL()


