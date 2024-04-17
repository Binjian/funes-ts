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
from scipy import interpolate

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

def mixed_visualization(ori_data, analysis, out_list=None, generated_data=None,out_data=None,fake_norm_list=None):
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

    ori = np.load('../datautils/sine+tri_data.npy').squeeze(axis=-1)  ##############
    print(ori.shape, ori_data.shape)
    ori = [list(i) for i in ori]
    data1 = []
    data2 = []
    tmp = [list(i) for i in ori_data.squeeze(axis=-1)]
    for i in range(ori_data.shape[0]):
        if ori.index(tmp[i]) <5000:
        # if ori.index(tmp[i]) < 5000 or ori.index(tmp[i]) >= 10000:
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
                c="brown",
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
        # plt.savefig('/home/veos/devel/funes/visualization_results/outlier_pca'+'-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.svg', dpi=300)
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
                c="brown",
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
        # plt.savefig('/home/veos/devel/funes/visualization_results/outlier_tsne'+'-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.svg', dpi=300)
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
                c="brown",
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
        # plt.savefig('/home/veos/devel/funes/visualization_results/outlier_umap'+'-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.svg',dpi=300)
        plt.show()


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
            outlier = np.array([pca_hat_results[i, :] for i in fake_norm_list if i < out_sample_no])
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
        # plt.savefig('/home/veos/devel/funes/visualization_results/outlier_pca'+'-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.svg', dpi=300)
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
        # plt.savefig('/home/veos/devel/funes/visualization_results/outlier_tsne'+'-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.svg', dpi=300)
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
        # plt.savefig('/home/veos/devel/funes/visualization_results/outlier_umap'+'-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.svg',dpi=300)
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

def gen_other_vol_data(data_len,seq_len):
    all_X = np.load('../datautils/X_clean_0927.npy')
    all_T = np.load('../datautils/T_clean_0927.npy')
    # np.random.shuffle(T)
    print(set(all_T))
    data = []
    for j in set(all_T):
        # print(j)
        if j==43:
            continue
        print(j)
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
        # print(len(list(range(len(t)))),len(t))
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

def triangle_wave_data(no,seq_len,dim):
    data = list()

    for i in range(no):
        temp = list()
        for k in range(dim):
            p1 = np.random.uniform(0.9, 1)
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


def gen_vol_data(data_len,seq_len):
    X = np.load('../datautils/X_clean_0927.npy')
    T = np.load('../datautils/T_clean_0927.npy')
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

def interp_spline():

    X = np.load('../datautils/X_clean_0927.npy')
    T = np.load('../datautils/T_clean_0927.npy')
    print(set(T))

    X = X[T == 43]  # baseline
    # np.random.shuffle(X)
    T = T[T == 43]
    t = 43
    data = []
    for x in X:
        f = interpolate.interp1d(list(range(t)), x[:t,0])
        xnew = np.arange(0,t-1,0.84)
        ynew = f(xnew)
        data.append(ynew)

    data = np.expand_dims(np.array(data),axis=-1)
    print(data.shape)
    return data


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

file_path = '../../data/output/vol_more_variations_1110/VOL-20221111-194445/mdl/VOL-vanilla-1200-1200-1200-64-8-2-0.001-50-4000'
# file_path = '../../data/output/vol_gen_1027/VOL-20221101-144724/mdl/VOL-vanilla-1200-1200-1200-128-8-2-0.001-100-4000'
# file_path = '../../data/output/sine+tri_improving_train/MIX-20221107-165722_best/mdl/MIX-vanilla-1500-1500-3000-64-24-3-0.0001-50-4000'

with open(
        f"{file_path}/args.pickle", "rb") as fb:
    args = torch.load(fb)

print(args.batch_size,args.hidden_dim)
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

discriminator.load_state_dict(dis_dict)
discriminator.eval()
embedder.load_state_dict(emb_dict)
embedder.eval()

# X = pd.read_pickle(
#     f'../{file_path}/fake_data.pickle')
# T = pd.read_pickle(
#     f'../{file_path}/fake_time.pickle')

train_data = pd.read_pickle(
    f'{file_path}/train_data.pickle')
fake_data = pd.read_pickle(
    f'{file_path}/fake_data.pickle')

# X = square_wave_data(1000,50,1)
# X = np.concatenate((X,sine_data_generation(10,50,1)),axis=0)
# X = triangle_wave_data(1000,100,1)
# X = sine_data_generation(1000,50,1)
# X = np.load('../datautils/vol_gen_data.npy')[:3000]
# X = gen_vol_data(1000,50)
# X = gen_other_vol_data(10,100)
X = np.load('../datautils/X_clean_0927.npy')
T = np.load('../datautils/T_clean_0927.npy')
data = []
print(set(T))
for t in set(T):
    x = X[T==t]
    if t > 50:
        data.extend(MinMaxScaler(x[:,t-50:t,0]))
    else:
        data.extend(np.pad(MinMaxScaler(x[:,:t,0]), ((0, 0), (0, 50-t)), 'constant', constant_values=(0, 0)))

print(np.array(data).shape)
print(np.array(data)[0])
X = np.expand_dims(np.array(data),axis=-1)


# X = X[T==43][:,:43,0]
# X = MinMaxScaler(X)
# X = np.pad(X, ((0, 0), (0, 7)), 'constant', constant_values=(0, 0))
# print(X[0])
# print(X.shape)
# X = np.expand_dims(X,axis=-1)

T = np.array([50 for _ in range(len(X))])

X = torch.from_numpy(X)
T = torch.from_numpy(T)
print(X.shape,T.shape)


THRESHOLD = 0.5

H_comb = embedder(X.float(), T.float()).detach()
Y_comb = discriminator(H_comb, T)


logit = []
for i in range(Y_comb.shape[0]):
    logit.append(float(Y_comb[i][-1].sigmoid()))

print(len(logit))

warn_list = []
fake_norm_list = []
fake_norm_dir = {}

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
# plt.savefig(
#         '/home/veos/devel/funes/visualization_results/Data_Logits' + '-' + datetime.datetime.now().strftime(
#             "%Y%m%d-%H%M%S") + '.svg', dpi=300)
plt.show()
print(len(warn_list))
print(f'warn rate:{len(warn_list)/len(logit)}') # train_data warning data 4982

print(sorted(fake_norm_dir.items(),key=lambda x:x[1],reverse=True))

# for i in sorted(fake_norm_dir.items(),key=lambda x:x[1],reverse=True)[:3]:
if len(fake_norm_dir) != 0:
    figure, ax = plt.subplots()
    figure.suptitle('fake normal data')
    figure.set_size_inches(9, 6)
    figure.subplots_adjust(right=0.85)
    pictures = len(fake_norm_dir) if len(fake_norm_dir) <= 16 else 16
    for i in range(pictures):
        plt.subplot(4, 4, i+1)
        # min_y = min(min(fake_data[j, :fake_time[j], 0]),
        #             min(train_data[mae_list.index(min(mae_list)), :train_time[mae_list.index(min(mae_list))], 0]))
        # max_y = max(max(fake_data[j, :fake_time[j], 0]),
        #             max(train_data[mae_list.index(min(mae_list)), :train_time[mae_list.index(min(mae_list))], 0]))
        # plt.ylim([min_y, max_y])
        # print(sorted(fake_norm_dir.items(),key=lambda x:x[1],reverse=True)[i])
        plt.plot(X[sorted(fake_norm_dir.items(),key=lambda x:x[1],reverse=True)[i][0]],label=f'fake{i}')
        plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.show()

fake_norm_list = [i[0] for i in sorted(fake_norm_dir.items(),key=lambda x:x[1],reverse=True)]
print(fake_norm_list)

visualization(train_data, 'pca', generated_data=fake_data,out_list=warn_list,out_data=np.array(X),fake_norm_list=fake_norm_list if len(fake_norm_list)!=0 else None)
visualization(train_data, 'tsne', generated_data=fake_data,out_list=warn_list,out_data=np.array(X),fake_norm_list=fake_norm_list if len(fake_norm_list)!=0 else None)
visualization(train_data, 'umap', generated_data=fake_data,out_list=warn_list,out_data=np.array(X),fake_norm_list=fake_norm_list if len(fake_norm_list)!=0 else None)
# mixed_visualization(train_data, 'pca', generated_data=fake_data,out_list=warn_list,out_data=np.array(X),fake_norm_list=fake_norm_list if len(fake_norm_list)!=0 else None)
# mixed_visualization(train_data, 'tsne', generated_data=fake_data,out_list=warn_list,out_data=np.array(X),fake_norm_list=fake_norm_list if len(fake_norm_list)!=0 else None)
# mixed_visualization(train_data, 'umap', generated_data=fake_data,out_list=warn_list,out_data=np.array(X),fake_norm_list=fake_norm_list if len(fake_norm_list)!=0 else None)


print(fake_data.shape,np.array(X).shape)
print(len(warn_list))



