from collections import OrderedDict
import os
# third-party packages
import matplotlib.pyplot as plt
import numpy as np
import torch
from funes.tgan.models.timegan import EmbeddingNetwork,DiscriminatorNetwork
import pandas as pd
import logging
import datetime
import time
import subprocess
import openpyxl
from scipy.signal import square,sawtooth
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualization(ori_data, analysis, out_list=None, generated_data=None,out_data=None,fake_norm_list=None):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      - analysis: tsne or pca
    """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([3000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    if fake_norm_list is not None:
        top_3 = fake_norm_list[:3]

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
            if out_data is not None:
                out_data_hat = np.reshape(
                    np.mean(out_data[0, :, :], 1), [1, seq_len]
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
            if out_data is not None and i < out_data.shape[0]:
                out_data_hat = np.concatenate(
                    (
                        out_data_hat,
                        np.reshape(np.mean(out_data[i, :, :], 1), [1, seq_len]),
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
        if len(out_list)!=0 or out_list is not None :
            pca_hat_results = pca.transform(out_data_hat)
            out_list = np.array(out_list)
            out_sample_no = min([3000, len(out_list)])
            out_idx = np.random.permutation(len(out_list))[:out_sample_no]
            out_list = out_list[out_idx]
            # print(len(out_list))
            # outlier = np.array([pca_hat_results[i, :] for i in out_list if i < out_sample_no]) ######################
            outlier = np.array([pca_hat_results[i, :] for i in out_list if i < out_sample_no])

            plt.scatter(
                outlier[:, 0],
                outlier[:, 1],
                c="green",
                alpha=1,
                marker="D",
                label="Outlier",
            )
        if fake_norm_list is not None:
            pca_hat_results = pca.transform(out_data_hat)
            fake_norm_list = np.array(fake_norm_list)
            fake_norm_sample_no = min([3000, len(fake_norm_list)])
            fake_norm_idx = np.random.permutation(len(fake_norm_list))[:fake_norm_sample_no]
            fake_norm_list = fake_norm_list[fake_norm_idx]
            # outlier = np.array([pca_hat_results[i, :] for i in out_list if i < out_sample_no]) ######################
            outlier = np.array([pca_hat_results[i, :] for i in fake_norm_list])
            plt.scatter(
                outlier[:, 0],
                outlier[:, 1],
                c="red",
                alpha=1,
                marker="D",
                label="Outlier",
            )
            top_idx = [list(fake_norm_list).index(k) for k in top_3]

            for k in range(len(top_idx)):
                plt.annotate(k, (outlier[top_idx[k], 0], outlier[top_idx[k], 1]))
            # for k,v in enumerate(list(range(outlier.shape[0]))):
            #     plt.annotate(v,(outlier[k,0],outlier[k,1]))

            # plt.scatter(
            #     outlier[:, 0],
            #     outlier[:, 1],
            #     c="red",
            #     alpha=1,
            #     marker="D",
            #     label="Outlier",
            # )

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
        if generated_data is not None and out_data is not None:
            prep_data_final = np.concatenate((prep_data, prep_data_hat,out_data_hat), axis=0)
        if generated_data is None and out_data is None:
            prep_data_final = np.concatenate(ori_data, axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)
        # print('xx', len(tsne_results), anal_sample_no)

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
                tsne_results[anal_sample_no:2*anal_sample_no, 0],
                tsne_results[anal_sample_no:2*anal_sample_no, 1],
                # tsne_results[anal_sample_no:, 0],
                # tsne_results[anal_sample_no:, 1],
                c=colors[anal_sample_no:],
                alpha=0.2,
                label="Synthetic",
            )
        if len(out_list)!=0 or out_list is not None:
            out_list = np.array(out_list)
            out_sample_no = min([3000, len(out_list)])
            out_idx = np.random.permutation(len(out_list))[:out_sample_no]
            out_list = out_list[out_idx]
            # outlier = np.array([tsne_results[i, :] for i in out_list if i < out_sample_no])
            outlier = np.array([tsne_results[2*anal_sample_no:][i, :] for i in out_list if i < out_sample_no])
            plt.scatter(
                outlier[:, 0],
                outlier[:, 1],
                c="green",
                alpha=1,
                marker="D",
                label="Outlier",
            )
        if fake_norm_list is not None:
            fake_norm_list = np.array(fake_norm_list)
            fake_norm_sample_no = min([3000, len(fake_norm_list)])
            fake_norm_idx = np.random.permutation(len(fake_norm_list))[:fake_norm_sample_no]
            fake_norm_list = fake_norm_list[fake_norm_idx]
            # print(norm_list)
            # outlier = np.array([pca_hat_results[i, :] for i in out_list if i < out_sample_no]) ######################
            outlier = np.array([tsne_results[2*anal_sample_no:][i, :] for i in fake_norm_list])
            plt.scatter(
                outlier[:, 0],
                outlier[:, 1],
                c="red",
                alpha=1,
                marker="D",
                label="Outlier",
            )
            top_idx = [list(fake_norm_list).index(k) for k in top_3]

            for k in range(len(top_idx)):
                plt.annotate(k, (outlier[top_idx[k], 0], outlier[top_idx[k], 1]))
            # for k, v in enumerate(list(range(outlier.shape[0]))):
            #     plt.annotate(v, (outlier[k, 0], outlier[k, 1]))

            # plt.scatter(
            #     outlier[:, 0],
            #     outlier[:, 1],
            #     c="red",
            #     alpha=1,
            #     marker="D",
            #     label="Outlier",
            # )

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
        if len(out_list)!=0 or out_list is not None:
            umap_out_results = umap.UMAP(n_neighbors=10, min_dist=0.1).fit_transform(
                out_data_hat
            )
            out_list = np.array(out_list)
            out_sample_no = min([3000, len(out_list)])
            out_idx = np.random.permutation(len(out_list))[:out_sample_no]
            out_list = out_list[out_idx]
            # outlier = np.array([umap_results[i, :] for i in out_list if i < out_sample_no])
            outlier = np.array([umap_out_results[i, :] for i in out_list if i < out_sample_no])
            plt.scatter(
                outlier[:, 0],
                outlier[:, 1],
                c="green",
                alpha=1,
                marker="D",
                label="Outlier",
            )
        if fake_norm_list is not None :
            umap_out_results = umap.UMAP(n_neighbors=10, min_dist=0.1).fit_transform(
                out_data_hat
            )
            fake_norm_list = np.array(fake_norm_list)
            fake_norm_sample_no = min([3000, len(fake_norm_list)])
            fake_norm_idx = np.random.permutation(len(fake_norm_list))[:fake_norm_sample_no]
            fake_norm_list = fake_norm_list[fake_norm_idx]
            # print(norm_list)
            # outlier = np.array([pca_hat_results[i, :] for i in out_list if i < out_sample_no]) ######################
            outlier = np.array([umap_out_results[i, :] for i in fake_norm_list])
            plt.scatter(
                outlier[:, 0],
                outlier[:, 1],
                c="red",
                alpha=1,
                marker="D",
                label="Outlier",
            )
            top_idx = [list(fake_norm_list).index(k) for k in top_3]

            for k in range(len(top_idx)):
                plt.annotate(k, (outlier[top_idx[k], 0], outlier[top_idx[k], 1]))
            # for k, v in enumerate(list(range(outlier.shape[0]))):
            #     plt.annotate(v, (outlier[k, 0], outlier[k, 1]))
            # plt.scatter(
            #     outlier[:, 0],
            #     outlier[:, 1],
            #     c="red",
            #     alpha=1,
            #     marker="D",
            #     label="Outlier",
            # )

        plt.legend()
        plt.title("umap plot")
        plt.xlabel("x-umap")
        plt.ylabel("y_umap")
        # plt.savefig('/home/veos/devel/funes/visualization_results/umap'+'-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.svg',dpi=300)
        plt.show()


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
            temp_data = square(2 * np.pi * p2 * temp_data, duty=p1)
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)
    # np.save('square_wave.npy', np.array(data))
    return np.array(data)


file_path = '../../data/output/bd_vol_base/VOL-20221026-143718/mdl/VOL-vanilla-10-10-10-128-24-3-0.001-100-4000'

with open(
        f"{file_path}/args.pickle", "rb") as fb:
    args = torch.load(fb)


discriminator = DiscriminatorNetwork(args)
embedder = EmbeddingNetwork(args)
dis_dict = OrderedDict()
emb_dict = OrderedDict()
# load timegan model
model = torch.load(f"{file_path}/model.pt")
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

train_data = pd.read_pickle(
    f'{file_path}/train_data.pickle')
fake_data = pd.read_pickle(
    f'{file_path}/fake_data.pickle')

X = square_wave_data(1000,50,1)

T = np.array([50 for _ in range(len(X))])

X = torch.from_numpy(X)
T = torch.from_numpy(T)
print(X.shape,T.shape)
# for i in range(3):
#     plt.plot(X[i])
#     plt.show()

THRESHOLD = 0.5

H_comb = embedder(X.float(), T.float()).detach()
# # discriminate data
Y_comb = discriminator(H_comb, T)

# logger.info(f"original battery data loaded successful")
# logger.info(f"inference start")
# # store logit by sigmoid and mean
logit = []
for i in range(Y_comb.shape[0]):
    logit.append(float(Y_comb[i][-1].sigmoid()))

print(len(logit))

# # create warn list
warn_list = []
fake_norm_list = []
fake_norm_dir = {}
# # plot data logits
plt.figure()
for i in range(len(logit)):
    if logit[i] < THRESHOLD:
        plt.scatter(i, logit[i], color="orange", label="Warning Data")
        warn_list.append(i)
    else:
        plt.scatter(i, logit[i], color="blue", label="Original Data")
        plt.annotate(len(fake_norm_list),(i,logit[i]))
        fake_norm_dir[i] = logit[i]
        fake_norm_list.append(i)
plt.xlabel("Sample no.")
plt.ylabel("Logits")
plt.title("Data Logits")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()
print(len(warn_list))
print(f'warn rate:{len(warn_list)/len(logit)}') # train_data warning data 4982

print(sorted(fake_norm_dir.items(),key=lambda x:x[1],reverse=True))
figure, ax = plt.subplots()
figure.suptitle('fake normal data')
figure.set_size_inches(9, 6)
figure.subplots_adjust(right=0.85)
# for i in sorted(fake_norm_dir.items(),key=lambda x:x[1],reverse=True)[:3]:
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(X[sorted(fake_norm_dir.items(),key=lambda x:x[1],reverse=True)[i][0]],label=f'fake{i}')
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
plt.show()

fake_norm_list = [i[0] for i in sorted(fake_norm_dir.items(),key=lambda x:x[1],reverse=True)]
print(fake_norm_list)

visualization(train_data, 'pca', generated_data=fake_data,out_list=warn_list,out_data=np.array(X),fake_norm_list=fake_norm_list if len(fake_norm_list)!=0 else None)
visualization(train_data, 'tsne', generated_data=fake_data,out_list=warn_list,out_data=np.array(X),fake_norm_list=fake_norm_list if len(fake_norm_list)!=0 else None)
visualization(train_data, 'umap', generated_data=fake_data,out_list=warn_list,out_data=np.array(X),fake_norm_list=fake_norm_list if len(fake_norm_list)!=0 else None)

print(fake_data.shape,np.array(X).shape)
print(len(warn_list))



