from ml_collections import config_dict
from collections import namedtuple

def get_config():

    Hyperparameter = namedtuple(
        "Hyperparameter",
        [
            "name",
            "seed",
            "feat_pred_no",
            "max_seq_len",
            "train_rate",
            "sup_epochs",
            "gan_epochs",
            "embed_epochs",
            "batch_size",
            "hidden_dim",
            "num_layers",
            "dis_thresh",
            "optimizer",
            "learning_rate",
            "lr_step",
            "lr_decay",
            "gradient_type",
            "heads",
        ],
    )


    # TimeGAN Arguments
    # wgan_hyperparam = Hyperparameter(
    #     "wgan_hyperparam",
    #     seed=42,
    #     feat_pred_no=1,
    #     max_seq_len=370,
    #     train_rate=0.5,
    #     sup_epochs=1000,
    #     gan_epochs=1000,
    #     embed_epochs=200,
    #     batch_size=[32, 64],
    #     hidden_dim=[32, 64],
    #     num_layers=[2, 3],
    #     dis_thresh=0.15,
    #     optimizer="adam",
    #     learning_rate=5e-4,
    #     gradient_type="wasserstein",
    # )


    # wgangp_hyperparam = Hyperparameter(
    #     "wgangp_hyperparam",
    #     seed=42,
    #     feat_pred_no=1,
    #     max_seq_len=370,
    #     train_rate=0.5,
    #     sup_epochs=10000,
    #     gan_epochs=10000,
    #     embed_epochs=1000,
    #     batch_size=[32, 64, 128],
    #     hidden_dim=[8, 16, 32, 64],
    #     num_layers=[2, 3],
    #     dis_thresh=0.15,
    #     optimizer="adam",
    #     learning_rate=5e-4,
    #     gradient_type="wasserstein_gp",
    # )
    # TimeGAN Arguments

    vanilla_hyperparam = Hyperparameter(
        "vanilla_hyperparam",
        seed=42,
        feat_pred_no=1,
        max_seq_len=50,
        train_rate=0.8,
        embed_epochs=1000, # 600
        sup_epochs=1000,
        gan_epochs=1000,
        batch_size=[32,64,128],
        hidden_dim=[8,16,32],
        num_layers=[2,3],
        heads=[2,4],
        dis_thresh=0.15,
        optimizer="adam",
        learning_rate=1e-3,
        lr_step=300,
        lr_decay=0.25,
        gradient_type="vanilla",
    )
    # wgan_config = config_dict.FrozenConfigDict(wgan_hyperparam._asdict())
    # wgangp_config = config_dict.FrozenConfigDict(wgangp_hyperparam._asdict())
    vanilla_config = config_dict.FrozenConfigDict(vanilla_hyperparam._asdict())

    # thaw frozen config dict
    config = config_dict.ConfigDict(vanilla_config)
    return config