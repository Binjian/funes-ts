import os
import datetime
import subprocess
import torch


def main():
    # TimeGAN Arguments
    seed = 42
    feat_pred_no = 1
    max_seq_len = 370
    train_rate = 0.5
    sup_epochs = 1000
    gan_epochs = sup_epochs
    embed_epochs = 200
    batch_size = 8
    hidden_dim = 8
    num_layers = 2
    dis_thresh = 0.15
    optimizer = "adam"
    learning_rate = 5e-3

    ts_specified = "20220323-175851"
    num_layers_specified = 3
    batch_size_specified = 32
    hidden_dim_specified = 32

    # hidden_dim*batch_size ！= <128*128,256*128,256*64,512*64,512*128,512*32,
    # clear cuda cache before training
    torch.cuda.empty_cache()
    print("CUDA CACHE clear!")
    subprocess.call(
        [
            "python",
            "tgan_app.py",
            "--device",
            "cuda",
            "--is_train",
            "True",
            "--seed",
            str(seed),
            "--feat_pred_no",
            str(feat_pred_no),
            "--max_seq_len",
            str(max_seq_len),
            "--train_rate",
            str(train_rate),
            "--emb_epochs",
            str(embed_epochs),
            "--sup_epochs",
            str(sup_epochs),
            "--gan_epochs",
            str(gan_epochs),
            "--batch_size",
            str(batch_size_specified),
            "--hidden_dim",
            str(hidden_dim_specified),
            "--num_layers",
            str(num_layers_specified),
            "--dis_thresh",
            str(dis_thresh),
            "--optimizer",
            optimizer,
            "--learning_rate",
            str(learning_rate),
            "--out_subdir",
            ts_specified,
        ]
    )

    print("Resumed Training complete!")
    print("===============")
    print("===============")

if __name__ == "__main__":
    # Call main function
    main()
