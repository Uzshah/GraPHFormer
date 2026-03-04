#!/bin/bash
# Example training script for GraPHFormer

python train.py \
    --exp_name graphformer_resnet18 \
    --dataset all_wo_others \
    --tree_model double \
    --image_encoder dinov2_vits14 \
    --image_size 252 \
    --embed_dim 128 \
    --h_size 256 \
    --batch_size 128 \
    --epochs 100 \
    --lr 3e-4 \
    --wd 0.1 \
    --temperature 0.07 \
    --loss_type clip \
    --warmup_epochs 5 \
    --save_freq 10 \
    --val_freq 5 \
    --aug_rotate \
    --aug_jitter_coords \
    --use_persistence_aug \
    --use_knn_eval \
    --knn_k 20 \
    --eval_jm \
    --eval_act
