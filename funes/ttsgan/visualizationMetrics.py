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
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from GANModels import *
from dataLoader import *
from torch.utils import data
print(matplotlib.get_backend())
from scipy.spatial import distance
import scipy
from scipy.fftpack import fft
import umap
import datetime
import argparse


def xxvisualization (ori_data, generated_data, analysis):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
    """  
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)  

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape  

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
        else:
            prep_data = np.concatenate((prep_data, 
                                        np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat, 
                                        np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))
    
    # Visualization parameter        
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]    

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components = 2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)    
        plt.scatter(pca_results[:,0], pca_results[:,1],
                    c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
        plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                    c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")

        ax.legend()  
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
#         plt.show()

    elif analysis == 'tsne':

        # Do t-SNE Analysis together       
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)

        # TSNE anlaysis
        tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                    c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
        plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                    c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")

        ax.legend()

        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
#         plt.show()
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

        plt.title("umap plot")
        plt.xlabel("x-umap")
        plt.ylabel("y_umap")
        plt.show()
        
    # plt.savefig(f'./images/{save_name}.pdf', format="pdf")
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
        plt.savefig('/home/veos/devel/funes/visualization_results/pca_ttsgan'+'-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.svg', dpi=300)
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
        plt.savefig('/home/veos/devel/funes/visualization_results/tsne_ttsgan'+'-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.svg', dpi=300)
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
        plt.savefig('/home/veos/devel/funes/visualization_results/umap_ttsgan'+'-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.svg',dpi=300)
        plt.show()



def get_feature_vector(data):
    # median, mean, standard deviation, variance, root mean square, maximum, and minimum
    fea_vectors = []
    for i in range(data.shape[0]):
        feature = []
        for c in range(data.shape[-1]):
            d = data[i, :, c]
            tmp = []
            tmp.append(np.median(d))
            tmp.append(np.mean(d))
            tmp.append(np.std(d))
            tmp.append(np.var(d))
            tmp.append(np.sqrt(np.mean(np.square(d))))
            tmp.append(np.max(d))
            tmp.append(np.min(d))
            # print(tmp)
            feature += tmp
        fea_vectors.append(feature)
        # print(feature,len(feature))
    print(np.array(fea_vectors).shape)
    return np.array(fea_vectors)

def show_results(seq_len,patch_size,latent_dim):
    # torch.Size([16, 3, 1, 150]) (BatchSize, C, 1, W)

    # x_train shape is (6055, 3, 1, 150), x_test shape is (1524, 3, 1, 150)
    # y_train shape is (6055,), y_test shape is (1524,)
    # file_path = './logs/battery_2022_09_07_18_10_44/Model/checkpoint'  # cos 0.9532487419178071  js
    # file_path = './pre-trained-models/RunningGAN_checkpoint' # cos 0.9617115065704973   js 0.23929891621731514
    file_path = './logs/Running_2022_09_01_15_59_21/Model/checkpoint' # cos 0.9530645168203082   js 0.28462435192799507
    generator = Generator(seq_len=seq_len, patch_size=patch_size, latent_dim=latent_dim)
    # generator = torch.nn.DataParallel(generator)
    print(torch.load(file_path)['gen_state_dict'].keys())
    generator.load_state_dict(torch.load(file_path)['gen_state_dict'])


    # generate fake data
    size = 1500 # dataset size

    fake_noise = torch.FloatTensor(np.random.normal(0, 1, (size, latent_dim)))
    fake_data = generator(fake_noise).detach().numpy()
    fake_data = np.array(fake_data).squeeze(axis=2).transpose(0, 2, 1)

    # get train data
    # train_set = battery_data()
    train_set = unimib_load_dataset(incl_xyz_accel=True, incl_rms_accel=False, incl_val_group=False, is_normalize=True,
                                                                    one_hot_encode=False, data_mode='Train', single_class=True,
                                                                    class_name='Running', augment_times=args.augment_times)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=fake_data.shape[0], num_workers=4, shuffle=False)
    print('##########',len(train_loader))
    origin = []
    for i in train_loader:
        # print(i)
        origin = np.array(i[0])

    train_data = origin.squeeze(axis=2).transpose(0, 2, 1)

    # pca & tsne
    visualization(train_data, fake_data, 'pca')
    visualization(train_data, fake_data, 'tsne')

    print(fake_data.shape)
    print(train_data.shape)


    fft_new_data = fft(fake_data)
    # print(f'{fake_data[0]}\n{fft_new_data[0]}')
    fft_ori_data = fft(train_data)
    pic_num = 16
    index = 0
    index = index * pic_num
    # print('xxxxx',fft_ori_data[:, :, 1])

    # ori_data_all = np.array(fft_ori_data[:, :, 2])  # soc ocv
    figure, ax = plt.subplots()
    figure.suptitle('FFT Most similar cases')
    for c in range(train_data.shape[-1]):
        ori_data_all = np.array(fft_ori_data[:, :, c])
        for j in range(1, pic_num + 1):
            # print(index)
            ori_data_case = ori_data_all[j + index]
            mae_list = []
            for i in range(ori_data_all.shape[0]):
                gen_data_case = np.array(fft_new_data[i, :, c])  # soc ocv
                diff = np.mean(abs(ori_data_case - gen_data_case))
                mae_list.append(diff)
            # print(max(mae_list), min(mae_list))
            # print(fake_data[mae_list.index(min(mae_list))])
            plt.subplot(4, 4, j)
            plt.ylim([-5, 5])
            plt.plot(fake_data[mae_list.index(min(mae_list)), :, c])
            plt.plot(train_data[mae_list.index(min(mae_list)), :, c])

        plt.show()

    mae_list = []
    for i in range(train_data.shape[0]):
        diff = np.mean(abs(fft_ori_data[i] - fft_new_data[i]))
        mae_list.append(diff)
    # print(sorted(mae_list))

    new_train_data = []
    new_fake_data = []
    for i in sorted(mae_list)[:int(size*0.2)]:
        idx = mae_list.index(i)
        new_fake_data.append(fake_data[idx, :, :])
        new_train_data.append(train_data[idx, :, :])

    new_train_data = np.array(new_train_data)
    new_fake_data = np.array(new_fake_data)
    print(new_fake_data.shape, new_train_data.shape)
    fake_data = new_fake_data
    train_data = new_train_data

    # fake_data = fake_data[mae_list.index(min(mae_list)), :, :]
    # train_data = train_data[mae_list.index(min(mae_list)), :, :]
    # print(fake_data.shape,train_data.shape)

    train_fea_vectors = get_feature_vector(train_data)
    fake_fea_vectors = get_feature_vector(fake_data)

    # average cosine similarity
    cos_smi = []
    for i in range(len(fake_fea_vectors)):
        smi = 1 - distance.cosine(train_fea_vectors[i], fake_fea_vectors[i])
        cos_smi.append(smi)
    avg_cos_smi = np.mean(cos_smi)
    print(len(cos_smi))
    print(f'avg_cos_smi:{avg_cos_smi}')

    # average Jensen-Shannon distance
    js_dis = []
    for i in range(len(fake_fea_vectors.transpose())):
        train = train_fea_vectors.transpose()[i]
        fake = fake_fea_vectors.transpose()[i]
        # print(train)
        # print(fake)
        prob_train = scipy.ndimage.histogram(train, np.min(train), np.max(train), 5)
        # print(prob_train)
        prob_fake = scipy.ndimage.histogram(fake, np.min(fake), np.max(fake), 5)
        # print(prob_fake)
        smi = distance.jensenshannon(prob_train, prob_fake)
        js_dis.append(smi)
        # break
    # print(js_dis)
    avg_js_dis = np.mean(js_dis)
    print(len(js_dis))
    print(f'avg_js_dis:{avg_js_dis}')

def show_battery_results(seq_len,patch_size,latent_dim):
    # torch.Size([16, 3, 1, 150]) (BatchSize, C, 1, W)

    # x_train shape is (6055, 3, 1, 150), x_test shape is (1524, 3, 1, 150)
    # y_train shape is (6055,), y_test shape is (1524,)
    file_path = './logs/battery-0913/battery_09_14_09_49_16_100/Model/checkpoint'
    # file_path = './pre-trained-models/RunningGAN_checkpoint'
    # file_path = './logs/Running_2022_09_01_15_59_21/Model/checkpoint'
    generator = Generator(seq_len=seq_len, patch_size=patch_size, latent_dim=latent_dim)
    # generator = torch.nn.DataParallel(generator)
    print(torch.load(file_path)['gen_state_dict']['module.pos_embed'].shape)
    generator.load_state_dict(torch.load(file_path)['gen_state_dict'],False)


    # generate fake data
    size = 6930 # dataset size

    fake_noise = torch.FloatTensor(np.random.normal(0, 1, (size, latent_dim)))
    # print(fake_noise)
    fake_data = generator(fake_noise).detach().numpy()
    # fake_data = generator(fake_noise).to(f'cuda:{0}')

    fake_data = np.array(fake_data).squeeze(axis=2).transpose(0, 2, 1)

    # get train data
    train_set = battery_data()
    # train_set = unimib_load_dataset(incl_xyz_accel=True, incl_rms_accel=False, incl_val_group=False, is_normalize=True,
    #                                                                 one_hot_encode=False, data_mode='Train', single_class=True,
    #                                                                 class_name='Running', augment_times=args.augment_times)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=fake_data.shape[0], num_workers=4, shuffle=False)
    print('##########',len(train_loader))
    origin = []
    for i in train_loader:
        # print(i)
        origin = np.array(i[0])

    train_data = origin.squeeze(axis=2).transpose(0, 2, 1)
    print(train_data)

    # pca & tsne
    visualization(train_data, fake_data, 'pca')
    visualization(train_data, fake_data, 'tsne')
    visualization(train_data, fake_data, 'umap')

    print(fake_data.shape)
    print(train_data.shape)


    fft_new_data = fft(fake_data)
    # print(f'{fake_data[0]}\n{fft_new_data[0]}')
    fft_ori_data = fft(train_data)
    pic_num = 16
    index = 0
    index = index * pic_num
    # print('xxxxx',fft_ori_data[:, :, 1])

    # ori_data_all = np.array(fft_ori_data[:, :, 2])  # soc ocv
    figure, ax = plt.subplots()
    figure.suptitle('FFT Most similar cases')
    for c in range(train_data.shape[-1]):
        ori_data_all = np.array(fft_ori_data[:, :, c])
        for j in range(1, pic_num + 1):
            # print(index)
            ori_data_case = ori_data_all[j + index]
            mae_list = []
            for i in range(ori_data_all.shape[0]):
                gen_data_case = np.array(fft_new_data[i, :, c])  # soc ocv
                diff = np.mean(abs(ori_data_case - gen_data_case))
                mae_list.append(diff)
            # print(max(mae_list), min(mae_list))
            # print(fake_data[mae_list.index(min(mae_list))])
            plt.subplot(4, 4, j)
            plt.ylim([-5, 5])
            plt.plot(fake_data[mae_list.index(min(mae_list)), :, c],label = "Synthetic")
            plt.plot(train_data[mae_list.index(min(mae_list)), :, c],label = "Original")
        plt.legend()
        plt.show()

    mae_list = []
    for i in range(train_data.shape[0]):
        diff = np.mean(abs(fft_ori_data[i] - fft_new_data[i]))
        mae_list.append(diff)
    # print(sorted(mae_list))

    new_train_data = []
    new_fake_data = []
    for i in sorted(mae_list)[:int(size*0.2)]:
        idx = mae_list.index(i)
        new_fake_data.append(fake_data[idx, :, :])
        new_train_data.append(train_data[idx, :, :])

    new_train_data = np.array(new_train_data)
    new_fake_data = np.array(new_fake_data)
    print(new_fake_data.shape, new_train_data.shape)
    fake_data = new_fake_data
    train_data = new_train_data

    # fake_data = fake_data[mae_list.index(min(mae_list)), :, :]
    # train_data = train_data[mae_list.index(min(mae_list)), :, :]
    # print(fake_data.shape,train_data.shape)

    train_fea_vectors = get_feature_vector(train_data)
    fake_fea_vectors = get_feature_vector(fake_data)

    # average cosine similarity
    cos_smi = []
    for i in range(len(fake_fea_vectors)):
        smi = 1 - distance.cosine(train_fea_vectors[i], fake_fea_vectors[i])
        cos_smi.append(smi)
    avg_cos_smi = np.mean(cos_smi)
    print(len(cos_smi))
    print(f'avg_cos_smi:{avg_cos_smi}')

    # average Jensen-Shannon distance
    js_dis = []
    for i in range(len(fake_fea_vectors.transpose())):
        train = train_fea_vectors.transpose()[i]
        fake = fake_fea_vectors.transpose()[i]
        # print(train)
        # print(fake)
        prob_train = scipy.ndimage.histogram(train, np.min(train), np.max(train), 5)
        # print(prob_train)
        prob_fake = scipy.ndimage.histogram(fake, np.min(fake), np.max(fake), 5)
        # print(prob_fake)
        smi = distance.jensenshannon(prob_train, prob_fake)
        js_dis.append(smi)
        # break
    # print(js_dis)
    avg_js_dis = np.mean(js_dis)
    print(len(js_dis))
    print(f'avg_js_dis:{avg_js_dis}')

def show_sine_results():
    # torch.Size([16, 3, 1, 150]) (BatchSize, C, 1, W)

    # x_train shape is (6055, 3, 1, 150), x_test shape is (1524, 3, 1, 150)
    # y_train shape is (6055,), y_test shape is (1524,)
    # file_path = './logs/sine-1125/sine_11-30-16-08_128_64_25_3_4/Model/checkpoint'
    file_path = 'logs/sine-1208/sine_12-08-15-06_64_8_5_2_2/Model/checkpoint'


    batch_size = int(file_path.split('/')[-3].split('_')[-5])
    latent_dim = int(file_path.split('/')[-3].split('_')[-4])
    patch_size = int(file_path.split('/')[-3].split('_')[-3])
    depth = int(file_path.split('/')[-3].split('_')[-2])
    heads = int(file_path.split('/')[-3].split('_')[-1])
    print(f"batch_size:{batch_size},latent_dim:{latent_dim},patch_size:{patch_size},depth:{depth},heads:{heads}")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--max_iter',
        type=int,
        default=None,
        help='set the max iteration number')
    parser.add_argument(
        '--seq_len',
        type=int,
        default=50,
        help='set the max sequence length')
    parser.add_argument(
        '-gen_bs',
        '--gen_batch_size',
        type=int,
        default=batch_size,
        help='size of the batches')
    parser.add_argument(
        '-dis_bs',
        '--dis_batch_size',
        type=int,
        default=batch_size,
        help='size of the batches')
    parser.add_argument(
        '-bs',
        '--batch_size',
        type=int,
        default=batch_size,
        help='size of the batches to load dataset')
    parser.add_argument(
        '--g_lr',
        type=float,
        default=0.001,
        help='adam: gen learning rate')
    parser.add_argument(
        '--d_lr',
        type=float,
        default=0.001,
        help='adam: disc learning rate')
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=latent_dim,
        help='dimensionality of the latent space')
    parser.add_argument(
        '--embed_dim',
        type=int,
        default=latent_dim,
        help='dimensionality of the embed space')
    parser.add_argument(
        '--channels',
        type=int,
        default=1,
        help='number of image channels')
    parser.add_argument('--heads', type=int, default=heads,
                        help='number of heads')
    parser.add_argument('--d_depth', type=int, default=depth,
                        help='Discriminator Depth')
    parser.add_argument('--g_depth', type=int, default=depth,
                        help='Generator Depth')
    parser.add_argument('--patch_size', type=int, default=patch_size,
                        help='Discriminator Depth')
    parser.add_argument('--loss', type=str, default="standard",
                        help='loss function')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='dropout ratio')

    args = parser.parse_args()
    # print(args.embed_dim,args.latent_dim)
    # print(args.gen_model)

    # generator = Generator(seq_len=seq_len, patch_size=patch_size, latent_dim=latent_dim)
    generator = Generator(args)
    generator = torch.nn.DataParallel(generator)

    # generator = torch.nn.DataParallel(generator)
    print(torch.load(file_path)['gen_state_dict']['module.pos_embed'].shape)
    generator.load_state_dict(torch.load(file_path)['gen_state_dict'],True)
    # print(torch.load(file_path)['gen_state_dict'])
    # for n,p in generator.named_parameters():
    #     print(n)
    #     print(p)


    # generate fake data
    size = 10000 # dataset size

    fake_noise = torch.FloatTensor(np.random.normal(0, 1, (size, latent_dim)))
    print(fake_noise.shape)

###########################################################
    # for s in range(size//batch_size):
    #     if s == 0:
    #         fake_data = generator(fake_noise[:batch_size])
    #     else:
    #         data = generator(fake_noise[batch_size*s:batch_size*(s+1)])
    #         fake_data = torch.concat((data,fake_data))
    # fake_data = fake_data.cpu().detach().numpy()
    # print(fake_data.shape)
###########################################################

    fake_data = generator(fake_noise).cpu().detach().numpy()
    # fake_data = generator(fake_noise).to(f'cuda:{0}')


    fake_data = np.array(fake_data).squeeze(axis=2).transpose(0, 2, 1)
    print('fake data:',fake_data.shape)
    # for i in fake_data[:3]:
    #     plt.plot(i)
    #     plt.show()

    # get train data
    train_set = sine_data()
    # train_set = unimib_load_dataset(incl_xyz_accel=True, incl_rms_accel=False, incl_val_group=False, is_normalize=True,
    #                                                                 one_hot_encode=False, data_mode='Train', single_class=True,
    #                                                                 class_name='Running', augment_times=args.augment_times)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=fake_data.shape[0], num_workers=4, shuffle=False)
    print('##########',len(train_loader))
    origin = []
    for i in train_loader:
        # print(i)
        origin = np.array(i[0])

    train_data = origin.squeeze(axis=2).transpose(0, 2, 1)
    # print(train_data)

    # pca & tsne
    # visualization(train_data, 'pca', generated_data=fake_data)
    # visualization(train_data, 'tsne', generated_data=fake_data)
    # visualization(train_data, 'umap', generated_data=fake_data)

    print(fake_data.shape)
    print(train_data.shape)
    train_time = np.array([train_data.shape[1] for _ in range(train_data.shape[0])])
    fake_time = np.array([fake_data.shape[1] for _ in range(fake_data.shape[0])])

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
    #     '/home/veos/devel/funes/visualization_results/ttsgan_most_similar_cases' + '-' + datetime.datetime.now().strftime(
    #         "%Y%m%d-%H%M%S") + '.svg', dpi=300)
    plt.show()

########################################################################
    # mae_list = []
    # for i in range(train_data.shape[0]):
    #     diff = np.mean(abs(fft_ori_data[i] - fft_new_data[i]))
    #     mae_list.append(diff)

    # new_train_data = []
    # new_fake_data = []
    # for i in sorted(mae_list)[:int(size*0.2)]:
    #     idx = mae_list.index(i)
    #     new_fake_data.append(fake_data[idx, :, :])
    #     new_train_data.append(train_data[idx, :, :])
    #
    # new_train_data = np.array(new_train_data)
    # new_fake_data = np.array(new_fake_data)
    # print(new_fake_data.shape, new_train_data.shape)
    # fake_data = new_fake_data
    # train_data = new_train_data
    #
    # # fake_data = fake_data[mae_list.index(min(mae_list)), :, :]
    # # train_data = train_data[mae_list.index(min(mae_list)), :, :]
    # # print(fake_data.shape,train_data.shape)
    #
    # train_fea_vectors = get_feature_vector(train_data)
    # fake_fea_vectors = get_feature_vector(fake_data)
    #
    # # average cosine similarity
    # cos_smi = []
    # for i in range(len(fake_fea_vectors)):
    #     smi = 1 - distance.cosine(train_fea_vectors[i], fake_fea_vectors[i])
    #     cos_smi.append(smi)
    # avg_cos_smi = np.mean(cos_smi)
    # print(len(cos_smi))
    # print(f'avg_cos_smi:{avg_cos_smi}')
    #
    # # average Jensen-Shannon distance
    # js_dis = []
    # for i in range(len(fake_fea_vectors.transpose())):
    #     train = train_fea_vectors.transpose()[i]
    #     fake = fake_fea_vectors.transpose()[i]
    #     # print(train)
    #     # print(fake)
    #     prob_train = scipy.ndimage.histogram(train, np.min(train), np.max(train), 5)
    #     # print(prob_train)
    #     prob_fake = scipy.ndimage.histogram(fake, np.min(fake), np.max(fake), 5)
    #     # print(prob_fake)
    #     smi = distance.jensenshannon(prob_train, prob_fake)
    #     js_dis.append(smi)
    #     # break
    # # print(js_dis)
    # avg_js_dis = np.mean(js_dis)
    # print(len(js_dis))
    # print(f'avg_js_dis:{avg_js_dis}')

if __name__=='__main__':
    # print(args.seq_len,args.patch_size,args.latent_dim)
    # show_results(150,15,100)
    show_sine_results()


