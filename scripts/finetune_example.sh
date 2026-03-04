#!/bin/bash
# Example fine-tuning script for GraPHFormer

python finetune.py \
    --exp_name finetune_bil \
    --dataset bil_6_classes \
    --pretrained_checkpoint work_dir/graphformer_resnet18/model_best.pth \
    --mode multimodal \
    --fusion_mode concat \
    --tree_model double \
    --image_encoder dinov2_vits14 \
    --image_size 252 \
    --embed_dim 128 \
    --h_size 256 \
    --batch_size 64 \
    --epochs 50 \
    --lr 1e-4 \
    --wd 0.01 \
    --dropout 0.5 \
    --label_smoothing 0.1 \
    --linear_probe_epochs 10 \
    --early_stopping_patience 10 \
    --val_freq 1 \
    --eval_mode accuracy
