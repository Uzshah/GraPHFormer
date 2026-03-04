"""
GraPHFormer Training Script

CLIP-style contrastive learning for neuron morphology representation.
Aligns tree structure representations with persistence images.

Usage:
    python train.py --exp_name my_experiment --dataset all_wo_others
"""

import argparse
import datetime
import time
import os
import json
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier

from graphformer.models import CLIPModel
from graphformer.augmentations import (
    Compose,
    RandomScaleCoords, RandomRotate, RandomJitter, RandomShift,
    RandomFlip, RandomMaskFeats, RandomJitterLength, RandomElasticate,
    RandomDropSubTrees, RandomSkipParentNode, RandomSwapSiblingSubTrees,
    CombinedPersistenceAugmentation,
)
from graphformer.utils import save_checkpoint, get_root_logger, set_seed
from graphformer.data import NeuronTreeDataset, get_collate_fn, LABEL_DICT


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    """KNN classification"""
    feature = F.normalize(feature, dim=-1)
    feature_bank = F.normalize(feature_bank, dim=-1)

    sim_matrix = torch.mm(feature, feature_bank.t())
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)

    sim_labels = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
    )

    sim_weight = (sim_weight / knn_t).exp()

    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, classes, device=sim_labels.device
    )
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )

    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    pred_labels = pred_scores.argmax(dim=-1)

    return pred_labels


def get_features_from_encoder(model, data_loader, device, fusion='concat'):
    """Extract features from encoder"""
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            tree_embed = model.encode_tree(batch)
            images = batch.images.cuda() if not batch.images.is_cuda else batch.images
            image_embed = model.encode_image(images)

            tree_embed = F.normalize(tree_embed, dim=-1)
            image_embed = F.normalize(image_embed, dim=-1)

            if fusion == 'concat':
                combined = torch.cat([tree_embed, image_embed], dim=1)
            elif fusion == 'add':
                combined = tree_embed + image_embed
            elif fusion == 'tree_only':
                combined = tree_embed
            elif fusion == 'image_only':
                combined = image_embed
            else:
                combined = torch.cat([tree_embed, image_embed], dim=1)

            combined = F.normalize(combined, dim=-1)

            features.append(combined)
            labels.append(batch.label.to(device))

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    return features, labels


def evaluate_sklearn_knn(model, train_loader, test_loader, device, knn_k=20, fusion='concat'):
    """Sklearn KNN evaluation"""
    tmp_model = copy.deepcopy(model).eval()

    x_train, y_train = get_features_from_encoder(tmp_model, train_loader, device, fusion)
    x_test, y_test = get_features_from_encoder(tmp_model, test_loader, device, fusion)

    neigh = KNeighborsClassifier(n_neighbors=knn_k)
    neigh.fit(x_train.cpu().numpy(), y_train.cpu().numpy())

    score = neigh.score(x_test.cpu().numpy(), y_test.cpu().numpy())

    del tmp_model
    return score


def evaluate_knn(model, memory_loader, test_loader, device, num_classes, knn_k=20, knn_t=0.5, fusion='concat'):
    """KNN evaluation during training"""
    model.eval()

    # Build memory bank from training set
    feature_bank = []
    with torch.no_grad():
        for batch in memory_loader:
            batch = batch.to(device)
            tree_embed = model.encode_tree(batch)
            images = batch.images.cuda() if not batch.images.is_cuda else batch.images
            image_embed = model.encode_image(images)

            tree_embed = F.normalize(tree_embed, dim=-1)
            image_embed = F.normalize(image_embed, dim=-1)

            if fusion == 'concat':
                combined = torch.cat([tree_embed, image_embed], dim=1)
            elif fusion == 'add':
                combined = tree_embed + image_embed
            elif fusion == 'tree_only':
                combined = tree_embed
            elif fusion == 'image_only':
                combined = image_embed
            else:
                combined = torch.cat([tree_embed, image_embed], dim=1)

            combined = F.normalize(combined, dim=-1)
            feature_bank.append(combined)

    feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
    feature_labels = torch.tensor(memory_loader.dataset.targets, device=device)

    # Test
    total_top1, total_num = 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            tree_embed = model.encode_tree(batch)
            images = batch.images.cuda() if not batch.images.is_cuda else batch.images
            image_embed = model.encode_image(images)

            tree_embed = F.normalize(tree_embed, dim=-1)
            image_embed = F.normalize(image_embed, dim=-1)

            if fusion == 'concat':
                combined = torch.cat([tree_embed, image_embed], dim=1)
            elif fusion == 'add':
                combined = tree_embed + image_embed
            elif fusion == 'tree_only':
                combined = tree_embed
            elif fusion == 'image_only':
                combined = image_embed
            else:
                combined = torch.cat([tree_embed, image_embed], dim=1)

            combined = F.normalize(combined, dim=-1)

            pred_labels = knn_predict(
                combined, feature_bank.t(), feature_labels,
                num_classes, knn_k, knn_t
            )

            total_num += combined.size(0)
            total_top1 += (pred_labels == batch.label.to(device)).float().sum().item()

    accuracy = total_top1 / total_num * 100
    model.train()
    return accuracy


def create_eval_dataset(phase, dataset_name, args):
    """Create evaluation dataset"""
    return NeuronTreeDataset(
        phase=phase,
        dataset=dataset_name,
        label_dict=LABEL_DICT[dataset_name],
        input_features=args.input_features,
        use_images=True,
        image_size=args.image_size,
        cache_images=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GraPHFormer Training")

    # Basic
    parser.add_argument("--work_dir", type=str, default="./work_dir")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="all_wo_others")
    parser.add_argument("--data_dir", type=str, default="data/raw/bil")
    parser.add_argument("--seed", type=int, default=42)

    # Tree Model
    parser.add_argument("--tree_model", type=str, default="double",
                        choices=["ori", "v2", "double"])
    parser.add_argument("--child_mode", type=str, default="sum")
    parser.add_argument("--input_features", nargs="+", type=int,
                        default=[2, 3, 4, 12, 13])
    parser.add_argument("--h_size", type=int, default=256)
    parser.add_argument("--bn", action="store_true", default=False)

    # Image Model
    parser.add_argument("--image_encoder", type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "resnet101", "simplecnn",
                                "smallvit", "persistencevit", "dinov2_vits14"])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--freeze_image_backbone", action="store_true", default=False)

    # CLIP settings
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--single_linear_proj", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--loss_type", type=str, default="clip",
                        choices=["clip", "infonce", "ntxent", "triplet"])

    # Triplet loss parameters
    parser.add_argument("--triplet_margin", type=float, default=1.0)
    parser.add_argument("--triplet_mining", type=str, default="batch_hard")
    parser.add_argument("--triplet_distance", type=str, default="euclidean")

    # Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--wd", default=0.1, type=float)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=5)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--gpu", default=0, type=int)

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

    # Evaluation datasets
    parser.add_argument("--eval_jm", action="store_true", default=False)
    parser.add_argument("--eval_act", action="store_true", default=False)
    parser.add_argument("--eval_neuron7", action="store_true", default=False)
    parser.add_argument("--eval_m1_cell", action="store_true", default=False)
    parser.add_argument("--eval_m1_region", action="store_true", default=False)
    parser.add_argument("--eval_swc_glia", action="store_true", default=False)

    # KNN evaluation
    parser.add_argument("--use_knn_eval", action="store_true", default=False)
    parser.add_argument("--knn_k", type=int, default=20)
    parser.add_argument("--knn_t", type=float, default=0.5)
    parser.add_argument("--knn_fusion", type=str, default="concat",
                        choices=["concat", "add", "tree_only", "image_only"])
    parser.add_argument("--use_sklearn_knn", action="store_true", default=False)

    parser.add_argument("--cache_images", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()
    set_seed(args.seed)

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
        log_file = f"{args.work_dir}/train_{timestamp}.log"
    logger = get_root_logger(log_file=log_file, log_level="INFO")

    logger.info("=" * 60)
    logger.info("GraPHFormer TRAINING")
    logger.info(f"Tree Encoder: {args.tree_model}")
    logger.info(f"Image Encoder: {args.image_encoder}")
    logger.info(f"Embedding Dimension: {args.embed_dim}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info("=" * 60)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    device = torch.device("cuda")

    # Create model
    logger.info("=> Creating model...")
    model = CLIPModel(args).to(device)
    logger.info(model)

    # Optimizer
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
            momentum=args.momentum
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

    # Resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=f"cuda:{args.gpu}")
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(f"=> Loaded checkpoint (epoch {checkpoint['epoch']})")

    cudnn.benchmark = True

    # Build augmentations
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

    # Create training dataset
    use_full_phase = args.dataset in ["all_wo_others", "all_with_neuron7", "neuron7", "ACT"]

    trainset = NeuronTreeDataset(
        phase="full" if use_full_phase else "train",
        dataset=args.dataset,
        label_dict=LABEL_DICT[args.dataset],
        data_dir=args.data_dir,
        topology_transformations=topo_augs,
        attribute_transformations=feat_augs,
        input_features=args.input_features,
        use_images=True,
        image_size=args.image_size,
        cache_images=args.cache_images,
        persistence_augmentation=persistence_aug,
        sigma_px=args.sigma_px,
    )

    collate_fn = get_collate_fn(device, use_images=True)

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
    )

    # Evaluation datasets
    eval_datasets = []
    eval_loaders = []
    eval_memory_loaders = []

    # BIL (always evaluated)
    bil_testset = create_eval_dataset("test", "bil_6_classes", args)
    bil_test_loader = DataLoader(
        dataset=bil_testset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    eval_datasets.append("BIL")
    eval_loaders.append(bil_test_loader)

    if args.use_knn_eval:
        bil_memory = create_eval_dataset("train", "bil_6_classes", args)
        bil_memory_loader = DataLoader(
            dataset=bil_memory,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        eval_memory_loaders.append(bil_memory_loader)

    # JM
    if args.eval_jm:
        jm_testset = create_eval_dataset("test", "JM", args)
        jm_test_loader = DataLoader(
            dataset=jm_testset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        eval_datasets.append("JM")
        eval_loaders.append(jm_test_loader)

        if args.use_knn_eval:
            jm_memory = create_eval_dataset("train", "JM", args)
            jm_memory_loader = DataLoader(
                dataset=jm_memory,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            eval_memory_loaders.append(jm_memory_loader)

    # ACT
    if args.eval_act:
        act_testset = create_eval_dataset("test", "ACT", args)
        act_test_loader = DataLoader(
            dataset=act_testset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        eval_datasets.append("ACT")
        eval_loaders.append(act_test_loader)

        if args.use_knn_eval:
            act_memory = create_eval_dataset("train", "ACT", args)
            act_memory_loader = DataLoader(
                dataset=act_memory,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            eval_memory_loaders.append(act_memory_loader)

    # Neuron7
    if args.eval_neuron7:
        neuron7_testset = create_eval_dataset("test", "neuron7", args)
        neuron7_test_loader = DataLoader(
            dataset=neuron7_testset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        eval_datasets.append("neuron7")
        eval_loaders.append(neuron7_test_loader)

        if args.use_knn_eval:
            neuron7_memory = create_eval_dataset("train", "neuron7", args)
            neuron7_memory_loader = DataLoader(
                dataset=neuron7_memory,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            eval_memory_loaders.append(neuron7_memory_loader)

    # M1_EXC_cell
    if args.eval_m1_cell:
        m1_cell_testset = create_eval_dataset("test", "m1_exc_cell", args)
        m1_cell_test_loader = DataLoader(
            dataset=m1_cell_testset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        eval_datasets.append("m1_exc_cell")
        eval_loaders.append(m1_cell_test_loader)

        if args.use_knn_eval:
            m1_cell_memory = create_eval_dataset("train", "m1_exc_cell", args)
            m1_cell_memory_loader = DataLoader(
                dataset=m1_cell_memory,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            eval_memory_loaders.append(m1_cell_memory_loader)

    # M1_EXC_region
    if args.eval_m1_region:
        m1_region_testset = create_eval_dataset("test", "m1_exc_region", args)
        m1_region_test_loader = DataLoader(
            dataset=m1_region_testset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        eval_datasets.append("m1_exc_region")
        eval_loaders.append(m1_region_test_loader)

        if args.use_knn_eval:
            m1_region_memory = create_eval_dataset("train", "m1_exc_region", args)
            m1_region_memory_loader = DataLoader(
                dataset=m1_region_memory,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            eval_memory_loaders.append(m1_region_memory_loader)

    # swc_glia
    if args.eval_swc_glia:
        swc_glia_testset = create_eval_dataset("test", "swc_glia_filtered_1000", args)
        swc_glia_test_loader = DataLoader(
            dataset=swc_glia_testset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        eval_datasets.append("swc_glia")
        eval_loaders.append(swc_glia_test_loader)

        if args.use_knn_eval:
            swc_glia_memory = create_eval_dataset("train", "swc_glia_filtered_1000", args)
            swc_glia_memory_loader = DataLoader(
                dataset=swc_glia_memory,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            eval_memory_loaders.append(swc_glia_memory_loader)

    # Learning rate scheduler
    def lr_schedule(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            min_lr_factor = 1e-6 / args.lr
            return min_lr_factor + (1 - min_lr_factor) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training loop
    best_metrics = {dataset: {"recall@5": 0.0, "epoch": 0} for dataset in eval_datasets}
    total_iters = len(train_loader) * args.epochs
    current_iter = 0
    start_time = time.time()

    logger.info("=> Starting training...")
    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            try:
                batch = batch.to(device)
                loss = model(batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
        logger.info(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f}")

        # Evaluation
        if epoch % args.val_freq == 0 and args.use_knn_eval:
            logger.info("=> Evaluating...")

            fusion_modes = ['concat', 'add', 'tree_only', 'image_only']

            for idx, (dataset_name, test_loader) in enumerate(zip(eval_datasets, eval_loaders)):
                memory_loader = eval_memory_loaders[idx]
                num_classes = len(test_loader.dataset.classes)

                # Use k=5 for JM, otherwise use args.knn_k
                k = 5 if dataset_name == "JM" else args.knn_k

                logger.info(f"\n  === {dataset_name} Dataset ===")

                fusion_results = {}
                best_fusion_acc = 0
                best_fusion_mode = None

                for fusion_mode in fusion_modes:
                    if args.use_sklearn_knn:
                        knn_acc = evaluate_sklearn_knn(
                            model, memory_loader, test_loader, device,
                            knn_k=k, fusion=fusion_mode
                        )
                        knn_acc = knn_acc * 100
                    else:
                        knn_acc = evaluate_knn(
                            model, memory_loader, test_loader, device,
                            num_classes, k, args.knn_t, fusion=fusion_mode
                        )

                    fusion_results[fusion_mode] = knn_acc

                    if knn_acc > best_fusion_acc:
                        best_fusion_acc = knn_acc
                        best_fusion_mode = fusion_mode

                    logger.info(f"    {fusion_mode:12s}: {knn_acc:.2f}%")

                logger.info(f"    {'BEST':12s}: {best_fusion_mode} ({best_fusion_acc:.2f}%)")

                # Save best checkpoint based on higher of concat or add
                concat_acc = fusion_results['concat']
                add_acc = fusion_results['add']
                primary_acc = max(concat_acc, add_acc)
                primary_fusion = 'add' if add_acc > concat_acc else 'concat'

                logger.info(f"    Best concat : {concat_acc:.2f}%")
                logger.info(f"    Best add    : {add_acc:.2f}%")
                logger.info(f"    Selected    : {primary_fusion} ({primary_acc:.2f}%)")

                if primary_acc > best_metrics[dataset_name]["recall@5"]:
                    best_metrics[dataset_name]["recall@5"] = primary_acc
                    best_metrics[dataset_name]["epoch"] = epoch
                    best_metrics[dataset_name]["fusion_mode"] = primary_fusion
                    best_metrics[dataset_name]["best_concat"] = concat_acc
                    best_metrics[dataset_name]["best_add"] = add_acc

                    checkpoint_path = f"{args.work_dir}/best_{dataset_name}_epoch_{epoch}.pth"
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "knn_accuracy": primary_acc,
                            "primary_fusion_mode": primary_fusion,
                            "concat_accuracy": concat_acc,
                            "add_accuracy": add_acc,
                            "fusion_results": fusion_results,
                            "dataset": dataset_name,
                        },
                        is_best=True,
                        filename=checkpoint_path,
                    )
                    logger.info(f"  Saved new best for {dataset_name}: {checkpoint_path}")

                logger.info(
                    f"  Best {dataset_name} ({best_metrics[dataset_name].get('fusion_mode', primary_fusion)}): "
                    f"{best_metrics[dataset_name]['recall@5']:.2f}% at epoch {best_metrics[dataset_name]['epoch']} "
                    f"[concat: {best_metrics[dataset_name].get('best_concat', concat_acc):.2f}%, "
                    f"add: {best_metrics[dataset_name].get('best_add', add_acc):.2f}%]"
                )

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

    logger.info("Training complete!")
    if args.use_knn_eval:
        logger.info("Best results:")
        for dataset_name in eval_datasets:
            logger.info(
                f"  {dataset_name}: {best_metrics[dataset_name]['recall@5']:.2f}% "
                f"at epoch {best_metrics[dataset_name]['epoch']}"
            )
