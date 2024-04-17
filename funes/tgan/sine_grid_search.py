# system packages
import datetime
import subprocess

import ml_collections.config_dict.config_dict
import torch
from collections import namedtuple
from absl import app
from absl import flags
from ml_collections import config_flags
import stock_tgan_config

vanilla_config = stock_tgan_config.get_config()
# print('11111111111',config)
_CONFIG = config_flags.DEFINE_config_dict("stock_tgan_config",vanilla_config)

def grid_search(_):
    """
    To run this script, you need to have the following files:
        - tgan_config.py on the same level as this script

    on the command line:
        - python3 grid-search.py --tgan_config=tgan_config.py
    """

    # print(_CONFIG.value)
    # Load config, select gan type from config
    hyperparam = _CONFIG.value
    print('num of layers',hyperparam.num_layers)
    # timestamp
    ts_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # grid search
    for num_layers_element in hyperparam.num_layers:
        for batch_size_element in hyperparam.batch_size:
            for hidden_dim_element in hyperparam.hidden_dim:
                for head in hyperparam.heads:
                    # clear cuda cache before training
                    torch.cuda.empty_cache()
                    print("CUDA CACHE clear!")
                    subprocess.call(
                        [
                            "python",
                            "sine.py",
                            "--device",
                            "cuda",
                            "--is_train",
                            "True",
                            "--seed",
                            str(hyperparam.seed),
                            "--feat_pred_no",
                            str(hyperparam.feat_pred_no),
                            "--max_seq_len",
                            str(hyperparam.max_seq_len),
                            "--train_rate",
                            str(hyperparam.train_rate),
                            "--emb_epochs",
                            str(hyperparam.embed_epochs),
                            "--sup_epochs",
                            str(hyperparam.sup_epochs),
                            "--gan_epochs",
                            str(hyperparam.gan_epochs),
                            "--batch_size",
                            str(batch_size_element),
                            "--hidden_dim",
                            str(hidden_dim_element),
                            "--num_layers",
                            str(num_layers_element),
                            "--heads",
                            str(head),
                            "--dis_thresh",
                            str(hyperparam.dis_thresh),
                            "--optimizer",
                            hyperparam.optimizer,
                            "--learning_rate",
                            str(hyperparam.learning_rate),
                            "--lr_step",
                            str(hyperparam.lr_step),
                            "--lr_decay",
                            str(hyperparam.lr_decay),
                            "--gradient_type",
                            hyperparam.gradient_type,
                            "--out_subdir",
                            ts_now,
                        ]
                    )
                    print("Train complete!")
                    print("===============")
    print("===============")
    print("Grid search complete!")

def sine_grid_search():
    app.run(grid_search)

if __name__ == "__main__":
    app.run(grid_search)
