import os
import datetime
import torch
from tgan_app import train_eval_module
# from src import PROJROOT
from pathlib import Path

PROJROOT = os.path.abspath(str(Path(__file__).parent.parent.parent))
print(f"PROJROOT: {PROJROOT}")

def resume_train():
    # TimeGAN Arguments

    # clear cuda cache before training
    torch.cuda.empty_cache()
    print("CUDA CACHE clear!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    is_train = True
    gradient_type = "vanilla"
    # ts_specified = "20220323-175851"
    ts_specified = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_subdir = ts_specified  #  ts_specified = "20220323-175851"


    seed = 42
    feat_pred_no = 1
    max_seq_len = 370
    train_rate = 0.5
    sup_epochs = 1000
    gan_epochs = sup_epochs
    emb_epochs = 200
    batch_size = 8
    hidden_dim = 8
    num_layers = 2
    dis_thresh = 0.15
    optimizer = "adam"
    learning_rate = 5e-3

    num_layers_specified = 3
    batch_size_specified = 32
    hidden_dim_specified = 32

    # Output directory
    # Input data directory
    dat_path = os.path.abspath(PROJROOT + "/data/")  # os.path.abspath("../../data")
    print(f"dat_path: {dat_path}")
    if not os.path.exists(dat_path):
        raise ValueError(f"Data file not found at {dat_path}.")

    # Output directories
    out_dir = os.path.abspath(PROJROOT + "/output/" + out_subdir)
    print(f"out_dir: {out_dir}")
    try:
        os.makedirs(out_dir)
    except FileExistsError:
        print("Output directory already exists. Resume...")

    ##############################################
    # Initialize output directories
    ##############################################
    hyperparam_setting_exp = (
        "SOC"
        + "-"
        + gradient_type
        + "-"
        + str(sup_epochs)
        + "-"
        + str(emb_epochs)
        + "-"
        + str(batch_size)
        + "-"
        + str(hidden_dim)
        + "-"
        + str(num_layers)
        + "-"
        + str(learning_rate)
        + "-"
        + str(max_seq_len)
        + "-"
        + str(4000)
    )

    # model directory
    mdl_dir = out_dir + "/mdl" + f"/{hyperparam_setting_exp}"
    print(f"mdl_dir: {mdl_dir}")
    try:
        os.makedirs(mdl_dir, exist_ok=False)
    except FileExistsError:
        print("Model directory already exists. Resume...")

    # TensorBoard directory
    tbd_dir = out_dir + "/tbd" + f"/{hyperparam_setting_exp}"
    print(f"tbd_dir: {tbd_dir}")
    try:
        os.makedirs(tbd_dir, exist_ok=False)
    except FileExistsError:
        print("TensorBoard directory already exists. Resume...")
    # Call main function

    # hidden_dim*batch_size ！= <128*128,256*128,256*64,512*64,512*128,512*32,

    train_eval_module(
        device,
        is_train,
        seed,
        feat_pred_no,
        max_seq_len,
        train_rate,
        gradient_type,
        emb_epochs,
        sup_epochs,
        gan_epochs,
        batch_size,
        hidden_dim,
        num_layers,
        dis_thresh,
        optimizer,
        learning_rate,
        dat_path,
        mdl_dir,
        tbd_dir,
    )

    print("Resumed Training complete!")
    print("===============")
    print("===============")


if __name__ == "__main__":
    # Call main function
    resume_train()
