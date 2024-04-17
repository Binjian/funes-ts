#!/bin/bash
train=true
export TZ="GMT-8"

# Experiment variables
exp="test"

# Iteration variables
emb_epochs=2
sup_epochs=1
gan_epochs=1

python tgan_app.py \
--device            cuda \
--exp               $exp \
--is_train          $train \
--seed              42 \
--feat_pred_no      1 \
--max_seq_len       413 \
--train_rate        0.5 \
--emb_epochs        $emb_epochs \
--sup_epochs        $sup_epochs \
--gan_epochs        $gan_epochs \
--batch_size        32 \
--hidden_dim        1 \
--num_layers        2 \
--dis_thresh        0.15 \
--optimizer         adam \
--learning_rate     5e-3 \