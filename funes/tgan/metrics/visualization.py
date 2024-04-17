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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import umap

__all__ = ["visualization"]


def visualization(ori_data, analysis, out_list=None, generated_data=None):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      - analysis: tsne or pca
    """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([3000, len(ori_data)])
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

        plt.title("umap plot")
        plt.xlabel("x-umap")
        plt.ylabel("y_umap")
        plt.show()


if __name__ == "__main__":
    pass
