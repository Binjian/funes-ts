# Local modules
import os
from typing import Dict, Union

# 3rd party modules
import numpy as np
from tqdm import trange

import torch
from torch.utils.tensorboard import SummaryWriter

# Self-written modules
from models.dataset import TimeGANDataset

from metrics import GeneralRNN
from metrics import FeaturePredictionDataset, OneStepPredictionDataset
from metrics import rmse_error
import traceback
from collections import OrderedDict
import datetime


# torch.autograd.set_detect_anomaly(True)


def embedding_trainer(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        e_opt: torch.optim.Optimizer,
        r_opt: torch.optim.Optimizer,
        sch_e_opt,
        sch_r_opt,
        args: Dict,
        writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)] = None,
):
    """The training loop for the embedding and recovery functions"""
    logger = trange(args.emb_epochs, desc=f"Epoch: 0, Loss: 0")

    for epoch in logger:
        if epoch% args.lr_step==0:
            print(f'embedding_trainer lr at epoch {epoch}: {sch_e_opt.get_last_lr()}')
        for X_mb, T_mb in dataloader:
            # Reset gradients
            model.zero_grad()

            # Forward Pass
            # time = [args.max_seq_len for _ in range(len(T_mb))]
            _, E_loss0, E_loss_T0 = model(X=X_mb, T=T_mb, Z=None, obj="autoencoder")

            loss = np.sqrt(E_loss_T0.item())

            # Backward Pass
            E_loss0.backward()

            # Update model parameters
            e_opt.step()
            r_opt.step()

            # if epoch%10000==0:
            #     print(e_opt.param_groups[0]['lr'])
            # break
        sch_e_opt.step()
        sch_r_opt.step()
        # Log loss for final batch of each epoch (29 iters)
        logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
        if writer:
            writer.add_scalar("Embedding/Loss:", loss, epoch)
            writer.flush()

    return loss


def supervisor_trainer(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        s_opt: torch.optim.Optimizer,
        g_opt: torch.optim.Optimizer,
        sch_s_opt,
        sch_g_opt,
        args: Dict,
        writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)] = None,
):
    """The training loop for the supervisor function"""
    logger = trange(args.sup_epochs, desc=f"Epoch: 0, Loss: 0")
    # sch_s_opt = torch.optim.lr_scheduler.StepLR(s_opt, step_size=10000, gamma=0.95)
    # sch_s_opt = torch.optim.lr_scheduler.ReduceLROnPlateau(s_opt, factor=args.lr_factor,patience=args.lr_patience,min_lr=args.lr_min, verbose=True)
    for epoch in logger:
        if epoch % args.lr_step == 0:
            print(f'supervisor_trainer lr at epoch {epoch}: {sch_s_opt.get_last_lr()}')
        for X_mb, T_mb in dataloader:
            # Reset gradients
            model.zero_grad()

            # Forward Pass
            S_loss = model(X=X_mb, T=T_mb, Z=None, obj="supervisor")

            # Backward Pass
            S_loss.backward()
            loss = np.sqrt(S_loss.item())

            # Update model parameters
            s_opt.step()

        sch_s_opt.step()
        # Log loss for final batch of each epoch (29 iters)
        logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
        if writer:
            writer.add_scalar("Supervisor/Loss:", loss, epoch)
            writer.flush()

    return loss


def joint_trainer_vanilla(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        e_opt: torch.optim.Optimizer,
        r_opt: torch.optim.Optimizer,
        s_opt: torch.optim.Optimizer,
        g_opt: torch.optim.Optimizer,
        d_opt: torch.optim.Optimizer,
        sch_e_opt,
        sch_r_opt,
        sch_s_opt,
        sch_g_opt,
        sch_d_opt,
        args: Dict,
        writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)] = None,
):
    """The training loop for training the model altogether"""
    logger = trange(args.gan_epochs, desc=f"Epoch: 0, E_loss: 0, G_loss: 0, D_loss: 0")

    is_error = False
    for epoch in logger:
        if epoch % args.lr_step == 0:
            print(f'joint trainer lr at epoch {epoch}: {sch_e_opt.get_last_lr()},{sch_r_opt.get_last_lr()},{sch_s_opt.get_last_lr()},{sch_g_opt.get_last_lr()},{sch_d_opt.get_last_lr()}')
        for X_mb, T_mb in dataloader:
            try:
                ## Generator Training
                for _ in range(2):
                    # Random Generator
                    Z_mb = torch.rand((args.batch_size, args.max_seq_len, args.Z_dim))

                    # Forward Pass (Generator)
                    model.zero_grad()
                    G_loss = model(X=X_mb, T=T_mb, Z=Z_mb, obj="generator")

                    G_loss.backward()
                    G_loss = np.sqrt(G_loss.item())

                    # Update model parameters
                    g_opt.step()
                    s_opt.step()

                    # Forward Pass (Embedding)
                    model.zero_grad()
                    E_loss, _, E_loss_T0 = model(X=X_mb, T=T_mb, Z=Z_mb, obj="autoencoder")

                    E_loss.backward()
                    E_loss = np.sqrt(E_loss.item())

                    # Update model parameters
                    e_opt.step()
                    r_opt.step()

                # Random Generator
                Z_mb = torch.rand((args.batch_size, args.max_seq_len, args.Z_dim))

                ## Discriminator Training
                model.zero_grad()
                # Forward Pass
                D_loss = model(X=X_mb, T=T_mb, Z=Z_mb, obj="discriminator")

                # Check Discriminator loss
                if D_loss > args.dis_thresh:
                    # Backward Pass
                    D_loss.backward()

                    # Update model parameters
                    d_opt.step()

                D_loss = D_loss.item()
                # break

            except:
                is_error = True
                print(traceback.format_exc())
                # gradients.close()
                # print(e1)
                break

        # ADD 100/1000 output
        # if epoch % 500 == 0:
        #     torch.save(model.state_dict(), f"{args.model_path}/model.pt")
        #     model.load_state_dict(torch.load(f"{args.model_path}/model.pt"))
        #     model.to(args.device)
        #     model.eval()
        #     with torch.no_grad():
        #         # Generate fake data
        #         T = T_mb
        #         Z = torch.rand((len(T), args.max_seq_len, args.Z_dim))
        #         generated_data = model(X=None, T=T, Z=Z, obj="inference")
        #         generated_data_np = generated_data.numpy()
        #     with open(f"{args.model_path}/fake_data_"+str(epoch)+".pickle", "wb") as fb:
        #         pickle.dump(generated_data_np, fb)

        # sch_e_opt.step()
        # sch_r_opt.step()
        # sch_s_opt.step()
        sch_g_opt.step()
        sch_d_opt.step()

        if is_error:
            print('##################################################')
            print(f'Error! Stop training at epoch {epoch}'.center(50))
            print('##################################################')
            break
        if E_loss < 0 or G_loss < 0 or D_loss < 0:
            print(
                "E_loss: "
                + str(E_loss)
                + " G_loss: "
                + str(G_loss)
                + " D_loss: "
                + str(D_loss)
            )
        logger.set_description(
            f"Epoch: {epoch}, E: {E_loss:.4f}, G: {G_loss:.4f}, D: {D_loss:.4f}"
        )
        if writer:
            writer.add_scalar("Joint/Embedding_Loss:", E_loss, epoch)
            writer.add_scalar("Joint/Generator_Loss:", G_loss, epoch)
            writer.add_scalar("Joint/Discriminator_Loss:", D_loss, epoch)
            writer.flush()
    return E_loss, G_loss, D_loss


def joint_trainer_wgan(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        e_opt: torch.optim.Optimizer,
        r_opt: torch.optim.Optimizer,
        s_opt: torch.optim.Optimizer,
        g_opt: torch.optim.Optimizer,
        d_opt: torch.optim.Optimizer,
        args: Dict,
        writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)] = None,
):
    """The training loop for training the model altogether"""
    logger = trange(args.gan_epochs, desc=f"Epoch: 0, E_loss: 0, G_loss: 0, D_loss: 0")

    for epoch in logger:
        for X_mb, T_mb in dataloader:
            ## Generator Training
            for _ in range(1):
                # Random Generator
                Z_mb = torch.rand((args.batch_size, args.max_seq_len, args.Z_dim))

                # Forward Pass (Generator)
                model.zero_grad()
                G_loss = model(X=X_mb, T=T_mb, Z=Z_mb, obj="generator")
                G_loss.backward()
                G_loss = np.sqrt(G_loss.item())

                # Update model parameters
                g_opt.step()
                s_opt.step()

                # Forward Pass (Embedding)
                model.zero_grad()
                E_loss, _, E_loss_T0 = model(X=X_mb, T=T_mb, Z=Z_mb, obj="autoencoder")
                E_loss.backward()
                E_loss = np.sqrt(E_loss.item())

                # Update model parameters
                e_opt.step()
                r_opt.step()
            # TODO add train more
            for _ in range(5):
                # Random Generator
                Z_mb = torch.rand((args.batch_size, args.max_seq_len, args.Z_dim))

                ## Discriminator Training
                model.zero_grad()
                # Forward Pass
                D_loss = model(X=X_mb, T=T_mb, Z=Z_mb, obj="discriminator")

                # Check Discriminator loss
                # if D_loss > args.dis_thresh:
                # Backward Pass
                D_loss.backward()

                # Update model parameters
                d_opt.step()
                # TODO add clip
                for p in model.discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

                D_loss = D_loss.item()
        if E_loss < 0 or G_loss < 0 or D_loss < 0:
            print(
                "E_loss: "
                + str(E_loss)
                + " G_loss: "
                + str(G_loss)
                + " D_loss: "
                + str(D_loss)
            )
        logger.set_description(
            f"Epoch: {epoch}, E: {E_loss:.4f}, G: {G_loss:.4f}, D: {D_loss:.4f}"
        )
        if writer:
            writer.add_scalar("Joint/Embedding_Loss:", E_loss, epoch)
            writer.add_scalar("Joint/Generator_Loss:", G_loss, epoch)
            writer.add_scalar("Joint/Discriminator_Loss:", D_loss, epoch)
            writer.flush()
    return E_loss, G_loss, D_loss


def timegan_trainer_vanilla(model, data, time, args):
    """The training procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model model that generates synthetic data
        - data (numpy.ndarray): The data for training the model
        - time (numpy.ndarray): The time for the model to be conditioned on
        - args (dict): The model/training configurations
    Returns:
        - generated_data (np.ndarray): The synthetic data generated by the model
    """

    # Initialize TimeGAN dataset and dataloader
    dataset = TimeGANDataset(data, time)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    model.to(args.device)

    # Initialize Optimizers
    # e_opt = torch.optim.Adam(model.embedder.parameters()) # ,lr=args.learning_rate
    # r_opt = torch.optim.Adam(model.recovery.parameters())
    # s_opt = torch.optim.Adam(model.supervisor.parameters())
    # g_opt = torch.optim.Adam(model.generator.parameters())
    # d_opt = torch.optim.Adam(model.discriminator.parameters())

    e_opt = torch.optim.Adam(model.embedder.parameters(), lr=args.learning_rate)
    r_opt = torch.optim.Adam(model.recovery.parameters(), lr=args.learning_rate)
    s_opt = torch.optim.Adam(model.supervisor.parameters(), lr=args.learning_rate)
    g_opt = torch.optim.Adam(model.generator.parameters(), lr=args.learning_rate)
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=args.learning_rate)

    sch_e_opt = torch.optim.lr_scheduler.StepLR(e_opt, step_size=args.lr_step, gamma=args.lr_decay)
    sch_r_opt = torch.optim.lr_scheduler.StepLR(r_opt, step_size=args.lr_step, gamma=args.lr_decay)
    sch_s_opt = torch.optim.lr_scheduler.StepLR(s_opt, step_size=args.lr_step, gamma=args.lr_decay)
    sch_g_opt = torch.optim.lr_scheduler.StepLR(g_opt, step_size=args.lr_step, gamma=args.lr_decay)
    sch_d_opt = torch.optim.lr_scheduler.StepLR(d_opt, step_size=args.lr_step, gamma=args.lr_decay)


    # restore checkpoint
    ckp_path = args.mdl_dir + "/ckp.pt"
    print('ckp:', ckp_path)
    if os.path.exists(ckp_path):
        checkpoint = torch.load(ckp_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        e_opt.load_state_dict(checkpoint["e_opt_state_dict"])
        r_opt.load_state_dict(checkpoint["r_opt_state_dict"])
        s_opt.load_state_dict(checkpoint["s_opt_state_dict"])
        g_opt.load_state_dict(checkpoint["g_opt_state_dict"])
        d_opt.load_state_dict(checkpoint["d_opt_state_dict"])
        args.last_emb_epochs += checkpoint["e-epoch"]
        args.last_sup_epochs += checkpoint["s-epoch"]
        args.last_gan_epochs += checkpoint["g-epoch"]

        # loss = checkpoint["loss"]
        print("Restored checkpoint from:", ckp_path)
    else:
        print("New model created")
    # TensorBoard writer
    writer = SummaryWriter(args.tbd_dir)

    print("\nStart Embedding Network Training")
    e_loss = embedding_trainer(
        model=model,
        dataloader=dataloader,
        e_opt=e_opt,
        r_opt=r_opt,
        sch_e_opt=sch_e_opt,
        sch_r_opt=sch_r_opt,
        args=args,
        writer=writer,
    )

    print("\nStart Training with Supervised Loss Only")
    s_loss = supervisor_trainer(
        model=model,
        dataloader=dataloader,
        s_opt=s_opt,
        g_opt=g_opt,
        sch_s_opt=sch_s_opt,
        sch_g_opt=sch_g_opt,
        args=args,
        writer=writer,
    )

    print("\nStart Joint Training")
    je_loss, jg_loss, jd_loss = joint_trainer_vanilla(
        model=model,
        dataloader=dataloader,
        e_opt=e_opt,
        r_opt=r_opt,
        s_opt=s_opt,
        g_opt=g_opt,
        d_opt=d_opt,
        sch_e_opt=sch_e_opt,
        sch_r_opt=sch_r_opt,
        sch_s_opt=sch_s_opt,
        sch_g_opt=sch_g_opt,
        sch_d_opt=sch_d_opt,
        args=args,
        writer=writer,
    )

    # Save checkpoint
    torch.save(
        {
            "e-epoch": args.emb_epochs,
            "s-epoch": args.sup_epochs,
            "g-epoch": args.gan_epochs,
            "model_state_dict": model.state_dict(),
            "e_opt_state_dict": e_opt.state_dict(),
            "r_opt_state_dict": r_opt.state_dict(),
            "s_opt_state_dict": s_opt.state_dict(),
            "g_opt_state_dict": g_opt.state_dict(),
            "d_opt_state_dict": d_opt.state_dict(),
            "e_loss": e_loss,
            "s_loss": s_loss,
            "je_loss": je_loss,
            "jg_loss": jg_loss,
            "jd_loss": jd_loss,
        },
        ckp_path,
    )

    # Save model, args, and hyperparameters
    torch.save(args, f"{args.mdl_dir}/args.pickle")
    torch.save(model.state_dict(), f"{args.mdl_dir}/model.pt")
    print(f"\nSaved at path: {args.mdl_dir}")


def timegan_trainer_wgan(model, data, time, args):
    """The training procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model model that generates synthetic data
        - data (numpy.ndarray): The data for training the model
        - time (numpy.ndarray): The time for the model to be conditioned on
        - args (dict): The model/training configurations
    Returns:
        - generated_data (np.ndarray): The synthetic data generated by the model
    """

    # Initialize TimeGAN dataset and dataloader
    dataset = TimeGANDataset(data, time)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=False
    )

    print(f"args.device: {args.device}")
    model.to(args.device)

    # Initialize Optimizers
    e_opt = torch.optim.Adam(model.embedder.parameters(), lr=args.learning_rate)
    r_opt = torch.optim.Adam(model.recovery.parameters(), lr=args.learning_rate)
    s_opt = torch.optim.Adam(model.supervisor.parameters(), lr=args.learning_rate)
    # TODO optimizer to rmsprop lr of 5e-5
    g_opt = torch.optim.RMSprop(model.generator.parameters(), lr=5e-5)
    d_opt = torch.optim.RMSprop(model.discriminator.parameters(), lr=5e-5)

    # restore checkpoint
    ckp_path = args.mdl_dir + "/ckp.pt"
    if os.path.exists(ckp_path):
        checkpoint = torch.load(ckp_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        e_opt.load_state_dict(checkpoint["e_opt_state_dict"])
        r_opt.load_state_dict(checkpoint["r_opt_state_dict"])
        s_opt.load_state_dict(checkpoint["s_opt_state_dict"])
        g_opt.load_state_dict(checkpoint["g_opt_state_dict"])
        d_opt.load_state_dict(checkpoint["d_opt_state_dict"])
        args.last_emb_epochs += checkpoint["e-epoch"]
        args.last_sup_epochs += checkpoint["s-epoch"]
        args.last_gan_epochs += checkpoint["g-epoch"]

        loss = checkpoint["loss"]
        print("Restored checkpoint from:", ckp_path)

    # TensorBoard writer
    writer = SummaryWriter(args.tbd_dir)

    print("\nStart Embedding Network Training")
    e_loss = embedding_trainer(
        model=model,
        dataloader=dataloader,
        e_opt=e_opt,
        r_opt=r_opt,
        args=args,
        writer=writer,
    )

    print("\nStart Training with Supervised Loss Only")
    s_loss = supervisor_trainer(
        model=model,
        dataloader=dataloader,
        s_opt=s_opt,
        g_opt=g_opt,
        args=args,
        writer=writer,
    )

    print("\nStart Joint Training")
    je_loss, jg_loss, jd_loss = joint_trainer_wgan(
        model=model,
        dataloader=dataloader,
        e_opt=e_opt,
        r_opt=r_opt,
        s_opt=s_opt,
        g_opt=g_opt,
        d_opt=d_opt,
        args=args,
        writer=writer,
    )

    # Save checkpoint
    torch.save(
        {
            "e-epoch": args.emb_epochs,
            "s-epoch": args.sup_epochs,
            "g-epoch": args.gan_epochs,
            "model_state_dict": model.state_dict(),
            "e_opt_state_dict": e_opt.state_dict(),
            "r_opt_state_dict": r_opt.state_dict(),
            "s_opt_state_dict": s_opt.state_dict(),
            "g_opt_state_dict": g_opt.state_dict(),
            "d_opt_state_dict": d_opt.state_dict(),
            "e_loss": e_loss,
            "s_loss": s_loss,
            "je_loss": je_loss,
            "jg_loss": jg_loss,
            "jd_loss": jd_loss,
        },
        ckp_path,
    )

    # Save model, args, and hyperparameters
    torch.save(args, f"{args.mdl_dir}/args.pickle")
    torch.save(model.state_dict(), f"{args.mdl_dir}/model.pt")
    print(f"\nSaved at path: {args.mdl_dir}")


def timegan_generator(model, T, args):
    """The inference procedure for TimeGAN
    Args:
        - model (torch.nn.module): The model model that generates synthetic data
        - T (List[int]): The time to be generated on
        - args (dict): The model/training configurations
    Returns:
        - generated_data (np.ndarray): The synthetic data generated by the model
    """
    # Load model for inference
    if not os.path.exists(args.mdl_dir):
        raise ValueError(f"Model directory not found...")

    # Load arguments and model
    with open(f"{args.mdl_dir}/args.pickle", "rb") as fb:
        args = torch.load(fb)

    model.load_state_dict(torch.load(f"{args.mdl_dir}/model.pt"))

    # print("\nGenerating Data...")
    # Initialize model to evaluation mode and run without gradients
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        # Generate fake data
        Z = torch.rand((len(T), args.max_seq_len, args.Z_dim))

        generated_data = model(X=None, T=T, Z=Z, obj="inference")

    return generated_data.numpy()


def feature_prediction(train_data, test_data, index):
    """Use the other features to predict a certain feature.

    Args:
    - train_data (train_data, train_time): training time-series
    - test_data (test_data, test_data): testing time-series
    - index: feature index to be predicted

    Returns:
    - perf: average performance of feature predictions (in terms of AUC or MSE)
    """
    train_data, train_time = train_data
    test_data, test_time = test_data
    # print('XXX',train_data.shape)

    # Parameters
    no, seq_len, dim = train_data.shape

    # Set model parameters

    args = {}
    args["device"] = "cuda"
    args["task"] = "regression"
    args["model_type"] = "gru"
    args["bidirectional"] = False
    args["epochs"] = 100
    args["batch_size"] = 128
    args["in_dim"] = dim - 1
    args["h_dim"] = dim - 1
    # args["in_dim"] = dim
    # args["h_dim"] = dim
    args["out_dim"] = 1
    args["n_layers"] = 3
    args["dropout"] = 0.5
    args["padding_value"] = -1.0
    args["max_seq_len"] = train_data.shape[1]
    args["learning_rate"] = 1e-3
    args["grad_clip_norm"] = 5.0

    # Output initialization
    perf = list()

    # For each index
    for idx in index:
        # Set training features and labels
        train_dataset = FeaturePredictionDataset(train_data, train_time, idx)
        # print(idx)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args["batch_size"], shuffle=True
        )

        # Set testing features and labels
        test_dataset = FeaturePredictionDataset(test_data, test_time, idx)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=no, shuffle=False
        )
        # print('XXX',train_data)

        # Initialize model
        model = GeneralRNN(args)
        model.to(args["device"])
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])

        logger = trange(args["epochs"], desc=f"Epoch: 0, Loss: 0")
        for epoch in logger:
            running_loss = 0.0

            for train_x, train_t, train_y in train_dataloader:
                train_x = train_x.to(args["device"])
                # print(train_x.shape)
                train_y = train_y.to(args["device"])
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                train_p = model(train_x, train_t)

                loss = criterion(train_p, train_y)
                # backward
                loss.backward()
                # optimize
                optimizer.step()

                running_loss += loss.item()

            logger.set_description(f"Epoch: {epoch}, Loss: {running_loss:.4f}")

        # Evaluate the trained model
        with torch.no_grad():
            temp_perf = 0
            for test_x, test_t, test_y in test_dataloader:
                test_x = test_x.to(args["device"])
                test_p = model(test_x, test_t).cpu().numpy()

                test_p = np.reshape(test_p, [-1])
                test_y = np.reshape(test_y.numpy(), [-1])

                temp_perf = rmse_error(test_y, test_p)

        perf.append(temp_perf)

    return perf


def one_step_ahead_prediction(train_data, test_data):
    """Use the previous time-series to predict one-step ahead feature values.

    Args:
    - train_data: training time-series
    - test_data: testing time-series

    Returns:
    - perf: average performance of one-step ahead predictions (in terms of AUC or MSE)
    """
    train_data, train_time = train_data
    test_data, test_time = test_data
    no, seq_len, dim = train_data.shape
    # Set model parameters
    args = {}
    args["device"] = "cuda"
    args["task"] = "regression"
    args["model_type"] = "gru"
    args["bidirectional"] = False
    args["epochs"] = 100
    args["batch_size"] = 128
    args["in_dim"] = dim
    args["h_dim"] = dim
    args["out_dim"] = dim
    args["n_layers"] = 3
    args["dropout"] = 0.5
    args["padding_value"] = -1.0
    args["max_seq_len"] = seq_len - 1  # only 99 is used for prediction
    args["learning_rate"] = 1e-3
    args["grad_clip_norm"] = 5.0

    # Set training features and labels
    train_dataset = OneStepPredictionDataset(train_data, train_time)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args["batch_size"], shuffle=True
    )

    # Set testing features and labels
    test_dataset = OneStepPredictionDataset(test_data, test_time)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=no, shuffle=True
    )
    # Initialize model
    model = GeneralRNN(args)
    model.to(args["device"])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])

    # Train the predictive model
    logger = trange(args["epochs"], desc=f"Epoch: 0, Loss: 0")
    for epoch in logger:
        running_loss = 0.0

        for train_x, train_t, train_y in train_dataloader:
            train_x = train_x.to(args["device"])
            train_y = train_y.to(args["device"])
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            train_p = model(train_x, train_t)
            loss = criterion(train_p, train_y)
            # backward
            loss.backward()
            # optimize
            optimizer.step()

            running_loss += loss.item()

        logger.set_description(f"Epoch: {epoch}, Loss: {running_loss:.4f}")

    # Evaluate the trained model
    with torch.no_grad():
        perf = 0
        for test_x, test_t, test_y in test_dataloader:
            test_x = test_x.to(args["device"])
            test_p = model(test_x, test_t).cpu()

            test_p = np.reshape(test_p.numpy(), [-1])
            test_y = np.reshape(test_y.numpy(), [-1])

            perf += rmse_error(test_y, test_p)

    return perf


if __name__ == "__main__":
    pass
