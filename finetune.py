"""
GraPHFormer Fine-tuning Script

Fine-tune pretrained CLIP-style models for classification.
Supports: image_only, tree_only, multimodal modes.

Usage:
    python finetune.py --exp_name my_finetune --pretrained_checkpoint path/to/checkpoint.pth
"""

import argparse
import datetime
import time
import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier

from graphformer.models import CLIPModel, FineTuneModel
from graphformer.augmentations import (
    Compose,
    RandomScaleCoords, RandomRotate, RandomJitter, RandomShift,
    RandomFlip, RandomMaskFeats, RandomJitterLength, RandomElasticate,
    RandomDropSubTrees, RandomSkipParentNode, RandomSwapSiblingSubTrees,
    CombinedPersistenceAugmentation,
)
from graphformer.utils import save_checkpoint, get_root_logger, set_seed
from graphformer.data import NeuronTreeDataset, get_collate_fn, LABEL_DICT


def evaluate_accuracy(model, data_loader, device):
    """Evaluate classification accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            _, logits = model(batch)

            pred = logits.argmax(dim=1)
            labels = batch.label.cuda() if not batch.label.is_cuda else batch.label

            correct += (pred == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    model.train()
    return accuracy


def extract_features(model, data_loader, device):
    """Extract features for KNN evaluation"""
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            _, _, features = model(batch, return_features=True)

            features_list.append(features)
            labels = batch.label.cuda() if not batch.label.is_cuda else batch.label
            labels_list.append(labels)

    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    model.train()
    return features, labels


def evaluate_knn(model, train_loader, test_loader, device, knn_k=20):
    """KNN evaluation using sklearn"""
    x_train, y_train = extract_features(model, train_loader, device)
    x_test, y_test = extract_features(model, test_loader, device)

    neigh = KNeighborsClassifier(n_neighbors=knn_k)
    neigh.fit(x_train.cpu().numpy(), y_train.cpu().numpy())

    score = neigh.score(x_test.cpu().numpy(), y_test.cpu().numpy())

    return score * 100


def mixup_data(features, labels, alpha=1.0):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = features.size(0)
    index = torch.randperm(batch_size).to(features.device)

    mixed_features = lam * features + (1 - lam) * features[index, :]
    labels_a, labels_b = labels, labels[index]

    return mixed_features, labels_a, labels_b, lam


def mixup_criterion(criterion, pred, labels_a, labels_b, lam):
    """Mixup loss"""
    return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GraPHFormer Fine-tuning")

    # Basic
    parser.add_argument("--work_dir", type=str, default="./work_dir")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="bil_6_classes")
    parser.add_argument("--data_dir", type=str, default="data/raw/bil")
    parser.add_argument("--seed", type=int, default=42)

    # Pretrained checkpoint
    parser.add_argument("--pretrained_checkpoint", type=str, default=None)

    # Fine-tuning mode
    parser.add_argument("--mode", type=str, default="multimodal",
                        choices=["image_only", "tree_only", "multimodal"])
    parser.add_argument("--freeze_encoders", action="store_true", default=False)
    parser.add_argument("--freeze_image_only", action="store_true", default=False)
    parser.add_argument("--linear_probe_epochs", type=int, default=0)
    parser.add_argument("--use_projection", action="store_true", default=False)
    parser.add_argument("--fusion_mode", type=str, default="concat",
                        choices=["concat", "add", "cross_attention", "bi_attention", "gated", "cmf", "mhcma"])

    # Tree Model
    parser.add_argument("--tree_model", type=str, default="double",
                        choices=["ori", "v2", "double"])
    parser.add_argument("--child_mode", type=str, default="sum")
    parser.add_argument("--input_features", nargs="+", type=int,
                        default=[2, 3, 4, 12, 13])
    parser.add_argument("--h_size", type=int, default=256)
    parser.add_argument("--bn", action="store_true", default=False)

    # Image Model
    parser.add_argument("--image_encoder", type=str, default="resnet18")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--freeze_image_backbone", action="store_true", default=False)

    # CLIP settings
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--loss_type", type=str, default="clip")

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--wd", default=0.01, type=float)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--gpu", default=0, type=int)

    # Regularization
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)
    parser.add_argument("--early_stopping_patience", type=int, default=0)

    # ArcFace loss
    parser.add_argument("--use_arcface", action="store_true", default=False)
    parser.add_argument("--arcface_s", type=float, default=30.0)
    parser.add_argument("--arcface_m", type=float, default=0.50)

    # Augmentation
    parser.add_argument("--aug_scale_coords", action="store_true", default=False)
    parser.add_argument("--aug_rotate", action="store_true", default=False)
    parser.add_argument("--aug_jitter_coords", action="store_true", default=False)
    parser.add_argument("--aug_shift_coords", action="store_true", default=False)
    parser.add_argument("--aug_flip", action="store_true", default=False)
    parser.add_argument("--aug_mask_feats", action="store_true", default=False)
    parser.add_argument("--aug_jitter_length", action="store_true", default=False)
    parser.add_argument("--aug_elasticate", action="store_true", default=False)
    parser.add_argument("--aug_drop_tree", action="store_true", default=False)
    parser.add_argument("--aug_skip_parent_node", action="store_true", default=False)
    parser.add_argument("--aug_swap_sibling_subtrees", action="store_true", default=False)

    # Persistence augmentation
    parser.add_argument("--use_persistence_aug", action="store_true", default=False)
    parser.add_argument("--pers_translation_scale", type=float, default=0.05)
    parser.add_argument("--pers_noise_scale", type=float, default=0.02)
    parser.add_argument("--pers_sigma_min", type=float, default=12.0)
    parser.add_argument("--pers_sigma_max", type=float, default=20.0)
    parser.add_argument("--sigma_px", type=float, default=16.0)

    # Evaluation
    parser.add_argument("--eval_mode", type=str, default="accuracy",
                        choices=["accuracy", "knn"])
    parser.add_argument("--knn_k", type=int, default=20)

    parser.add_argument("--cache_images", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()
    set_seed(args.seed)

    if args.linear_probe_epochs > 0:
        args.freeze_encoders = True
        
    # Setup work directory
    args.work_dir = f"{args.work_dir}/{args.exp_name}"
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    # Logger
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if args.debug:
        log_file = None
        args.save_freq = 10000
        args.val_freq = 1
    else:
        log_file = f"{args.work_dir}/finetune_{timestamp}.log"
    logger = get_root_logger(log_file=log_file, log_level="INFO")

    logger.info("=" * 60)
    logger.info("GraPHFormer FINE-TUNING")
    logger.info(f"Mode: {args.mode}")
    if args.mode == "multimodal":
        logger.info(f"Fusion Mode: {args.fusion_mode}")
    logger.info(f"Freeze Encoders: {args.freeze_encoders}")
    logger.info(f"Pretrained Checkpoint: {args.pretrained_checkpoint}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info("=" * 60)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    device = torch.device("cuda")

    # Load pretrained model or create from scratch
    if args.pretrained_checkpoint is not None:
        logger.info("=> Loading pretrained model...")
        if not os.path.isfile(args.pretrained_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.pretrained_checkpoint}")

        checkpoint = torch.load(args.pretrained_checkpoint, map_location=f"cuda:{args.gpu}")
        state_dict = checkpoint["state_dict"]

        # Auto-detect image encoder
        if args.mode in ['image_only', 'multimodal']:
            if "image_encoder.encoder.backbone.cls_token" in state_dict:
                args.image_encoder = "dinov2_vits14"
                logger.info(f"=> Detected DINOv2 image encoder")

        pretrained_model = CLIPModel(args).to(device)
        missing_keys, unexpected_keys = pretrained_model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            logger.warning(f"=> Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            logger.warning(f"=> Unexpected keys: {len(unexpected_keys)}")

        logger.info(f"=> Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        logger.info("=> Training from scratch")
        pretrained_model = CLIPModel(args).to(device)

    # Setup augmentations
    aug_switchs = [
        False,
        args.aug_scale_coords,
        args.aug_rotate,
        args.aug_jitter_coords,
        args.aug_shift_coords,
        args.aug_flip,
        args.aug_mask_feats,
        args.aug_jitter_length,
        args.aug_elasticate,
    ]
    aug_fns = [
        None,
        RandomScaleCoords(p=0.2),
        RandomRotate(p=0.5),
        RandomJitter(p=0.2),
        RandomShift(p=0.2),
        RandomFlip(p=1),
        RandomMaskFeats(p=0.2),
        RandomJitterLength(p=0.2),
        RandomElasticate(p=0.2),
    ]
    feat_augs = [aug_fns[i] for i in range(len(aug_switchs)) if aug_switchs[i] and aug_fns[i] is not None]
    feat_augs = Compose(feat_augs) if feat_augs else None

    topo_aug_switchs = [
        args.aug_drop_tree,
        args.aug_skip_parent_node,
        args.aug_swap_sibling_subtrees,
    ]
    topo_aug_fns = [
        RandomDropSubTrees(probs=[0.05], max_cnt=5),
        RandomSkipParentNode(probs=[0.05], max_cnt=10),
        RandomSwapSiblingSubTrees(probs=[0.05], max_cnt=10),
    ]
    topo_augs = [topo_aug_fns[i] for i in range(len(topo_aug_switchs)) if topo_aug_switchs[i]]
    topo_augs = Compose(topo_augs) if topo_augs else None

    # Persistence augmentation
    persistence_aug = None
    if args.use_persistence_aug:
        persistence_aug = CombinedPersistenceAugmentation(
            translation_scale=args.pers_translation_scale,
            noise_scale=args.pers_noise_scale,
            sigma_min=args.pers_sigma_min,
            sigma_max=args.pers_sigma_max,
        )

    # Create datasets
    collate_fn = get_collate_fn(device, use_images=True)

    trainset = NeuronTreeDataset(
        phase="train",
        dataset=args.dataset,
        label_dict=LABEL_DICT[args.dataset],
        input_features=args.input_features,
        topology_transformations=topo_augs,
        attribute_transformations=feat_augs,
        use_images=True,
        image_size=args.image_size,
        cache_images=args.cache_images,
        persistence_augmentation=persistence_aug,
        sigma_px=args.sigma_px,
    )

    testset = NeuronTreeDataset(
        phase="test",
        dataset=args.dataset,
        label_dict=LABEL_DICT[args.dataset],
        input_features=args.input_features,
        use_images=True,
        image_size=args.image_size,
        cache_images=args.cache_images,
    )

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset=testset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Train samples: {len(trainset)}, Test samples: {len(testset)}")
    logger.info(f"Number of classes: {len(trainset.classes)}")

    # Create fine-tuning model
    logger.info("=> Creating fine-tuning model...")
    model = FineTuneModel(
        pretrained_model=pretrained_model,
        num_classes=len(trainset.classes),
        mode=args.mode,
        freeze_encoders=args.freeze_encoders,
        fusion_mode=args.fusion_mode if args.mode == "multimodal" else None,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        use_projection=args.use_projection,
        use_arcface=args.use_arcface,
        arcface_s=args.arcface_s,
        arcface_m=args.arcface_m,
        freeze_image_only=args.freeze_image_only
    ).to(device)

    del pretrained_model
    logger.info(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.wd,
    )

    cudnn.benchmark = True

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=25,
        T_mult=1,
        eta_min=args.lr * 0.01
    )

    # Training loop
    best_metric = 0.0
    best_epoch = 0
    patience_counter = 0
    total_iters = len(train_loader) * args.epochs
    current_iter = 0
    start_time = time.time()

    logger.info("=> Starting fine-tuning...")
    logger.info(f"=> Regularization: dropout={args.dropout}, label_smoothing={args.label_smoothing}")

    for epoch in range(args.start_epoch + 1, args.epochs + 1):

        if epoch == args.linear_probe_epochs + 1 and args.linear_probe_epochs > 0:
            logger.info("="*30)
            logger.info(f"Linear probe phase completed. Unfreezing encoders for full fine-tuning.")
            model.unfreeze_encoders()
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Trainable parameters after unfreezing: {trainable_params:,}")

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr / 10,
                weight_decay=args.wd,
            )
            logger.info(f"Optimizer reset to include unfrozen parameters with a new learning rate of {args.lr / 10}.")
            logger.info("="*30)

        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for step, batch in enumerate(train_loader):
            try:
                batch = batch.to(device)

                if args.mixup_alpha > 0:
                    _, _, features = model(batch, return_features=True)
                    labels = batch.label.cuda() if not batch.label.is_cuda else batch.label

                    features, labels_a, labels_b, lam = mixup_data(features, labels, args.mixup_alpha)

                    if model.use_arcface:
                        extracted_features = model.feature_extractor(features)
                        logits = model.arcface(extracted_features, labels)
                        loss = model.criterion(logits, labels)
                    else:
                        logits = model.classifier(features)
                        loss = mixup_criterion(model.criterion, logits, labels_a, labels_b, lam)
                else:
                    loss, logits = model(batch)

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                pred = logits.argmax(dim=1)
                labels = batch.label.cuda() if not batch.label.is_cuda else batch.label
                correct += (pred == labels).sum().item()
                total += labels.size(0)

                epoch_loss += loss.item()
                current_iter += 1

                if step % 10 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time

                    log_str = (
                        f"Epoch {epoch:03d} | Step {step:03d}/{len(train_loader)} | "
                        f"Loss {loss.item():.4f} | "
                        f"LR {optimizer.param_groups[0]['lr']:.6f} | "
                        f"Elapsed {str(datetime.timedelta(seconds=int(elapsed)))}"
                    )
                    logger.info(log_str)
            except Exception as e:
                logger.info(f"Error in step {step}: {e}")
                continue

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        train_acc = correct / total * 100
        logger.info(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Evaluation
        if epoch % args.val_freq == 0:
            logger.info("=> Evaluating...")

            if args.eval_mode == "accuracy":
                test_acc = evaluate_accuracy(model, test_loader, device)
                logger.info(f"  Test Accuracy: {test_acc:.2f}%")
                metric = test_acc
            else:
                knn_acc = evaluate_knn(model, train_loader, test_loader, device, args.knn_k)
                logger.info(f"  KNN Accuracy (k={args.knn_k}): {knn_acc:.2f}%")
                metric = knn_acc

            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch
                patience_counter = 0

                checkpoint_path = f"{args.work_dir}/best_model.pth"
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "metric": metric,
                        "mode": args.mode,
                    },
                    is_best=True,
                    filename=checkpoint_path,
                )
                logger.info(f"  Saved new best checkpoint: {checkpoint_path}")
            else:
                patience_counter += 1

            logger.info(f"  Best: {best_metric:.2f}% at epoch {best_epoch}")

            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                logger.info(f"  Early stopping triggered")
                break

        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = f"{args.work_dir}/epoch_{epoch}.pth"
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=False,
                filename=checkpoint_path,
            )
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    logger.info("Fine-tuning complete!")
    logger.info(f"Best {args.eval_mode}: {best_metric:.2f}% at epoch {best_epoch}")
