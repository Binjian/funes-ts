#!/usr/bin/env bash

import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=str, default="0")
    parser.add_argument('--node', type=str, default="0015")
    opt = parser.parse_args()

    return opt
args = parse_args()

batch_size = [64,128]
latent_dim = [8,16,32]
patch_size = [5,10]
loss = 'lsgan'
lr = 0.001
depth = [2,3]
heads = [2,4]

for bs in batch_size:
    for dim in latent_dim:
        for ps in patch_size:
            for dp in depth:
                for h in heads:
                    os.system(f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python sine_gan.py \
                    -gen_bs {bs} \
                    -dis_bs {bs} \
                    --seq_len 50 \
                    --patch_size {ps} \
                    --dist-url 'tcp://localhost:4321' \
                    --dist-backend 'nccl' \
                    --world-size 1 \
                    --rank {args.rank} \
                    --dataset battery \
                    --bottom_width 8 \
                    --max_iter 80000 \
                    --img_size 50 \
                    --gen_model my_gen \
                    --dis_model my_dis \
                    --df_dim 384 \
                    --heads {h} \
                    --d_depth {dp} \
                    --g_depth {dp} \
                    --dropout 0 \
                    --latent_dim {dim} \
                    --embed_dim {dim} \
                    --gf_dim 1024 \
                    --num_workers 16 \
                    --g_lr {lr} \
                    --d_lr {lr} \
                    --optimizer adam \
                    --loss {loss} \
                    --wd 1e-3 \
                    --beta1 0.9 \
                    --beta2 0.999 \
                    --phi 1 \
                    --batch_size {bs} \
                    --num_eval_imgs 50000 \
                    --init_type xavier_uniform \
                    --n_critic 1 \
                    --val_freq 20 \
                    --print_freq 50 \
                    --grow_steps 0 0 \
                    --fade_in 0 \
                    --ema_kimg 500 \
                    --ema_warmup 0.1 \
                    --ema 0.9999 \
                    --diff_aug translation,cutout,color \
                    --class_name sine \
                    --channels 1 \
                    --exp_name sine")

# --max_epoch 100\