import math
import os
import time
from dataclasses import dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union
import warnings

import matplotlib
import numpy as np
import torch
import tyro
import yaml
import pprint
from tqdm import tqdm
import json

from gsplat import rasterization
import pycolmap_scene_manager as pycolmap

matplotlib.use("TkAgg")  # To avoid conflict with cv2

from utils import (
    load_checkpoint,
    prune_by_gradients,
    test_proper_pruning,
    get_frames,
    to_builtin
)

from feature_extractors import (
    get_feature_extractor,
    BACKPROJECTION_FEATURE_EXTRACTORS,
)



def create_feature_field(
    splats, feature_type="lseg", use_cpu=False, percentage_frames=100, n_views=None, normalize=True
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not use_cpu else "cpu"
    )
    if feature_type in ["one-hot","one-hot-3dovs", "clip-text-3dovs"]:
        data_dir = splats["data_dir"]
        feature_extractor = get_feature_extractor(feature_type, device, data_dir=data_dir)
    elif feature_type == "feature-map":
        feature_dir = splats["feature_dir"]
        feature_extractor = get_feature_extractor(feature_type, device, data_dir=None,feature_dir=feature_dir)
    elif feature_type == "lang-splat":
        feature_dir = splats["feature_dir"]
        data_dir = splats["data_dir"]
        feature_extractor = get_feature_extractor(feature_type, device, data_dir=data_dir, feature_dir=feature_dir, level=splats["level"])
    else:
        feature_extractor = get_feature_extractor(feature_type, device)
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)

    colmap_project = splats["colmap_project"]

    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    K = splats["camera_matrix"]

    n_gaussians = means.shape[0]

    gaussian_features = torch.zeros(
        n_gaussians, feature_extractor.dim, device=colors.device
    )
    gaussian_denoms = torch.ones(n_gaussians, device=colors.device) * 1e-12

    t1 = time.time()

    dummy_feats = torch.zeros(
        n_gaussians, feature_extractor.dim, device=colors.device
    )
    dummy_feats.requires_grad = True
    dummy_feats_denom = torch.zeros(n_gaussians, 3, device=colors.device)
    dummy_feats_denom.requires_grad = True

    print("Distilling features...")
    for frame in tqdm(
        get_frames(colmap_project, percentage_frames=percentage_frames, n_views=n_views)
    ):
        
        # Temp fix
        if frame["image_name"].startswith("test"):
            continue

        width = int(K[0, 2] * 2)
        height = int(K[1, 2] * 2)
        viewmat = frame["viewmat"].to(device)
        metadata = frame
        with torch.no_grad():
            output, _, meta = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmat[None],
                K[None],
                width=width,
                height=height,
                sh_degree=3,
            )
            feats = feature_extractor.extract_features(output[0], metadata)
            if feats is None:
                continue

        output_for_grad, _, meta = rasterization(
            means,
            quats,
            scales,
            opacities,
            dummy_feats,
            viewmat[None],
            K[None],
            width=width,
            height=height,
        )

        loss_num = (output_for_grad[0] * feats).mean()

        loss_num.backward()

        dummy_feats_copy = dummy_feats.grad.clone()

        dummy_feats.grad.zero_()

        output_for_grad, _, meta = rasterization(
            means,
            quats,
            scales,
            opacities,
            dummy_feats_denom,
            viewmat[None],
            K[None],
            width=width,
            height=height,
        )

        loss_denom = (output_for_grad[0]).mean()

        loss_denom.backward()

        gaussian_features += dummy_feats_copy  # / (dummy_feats_denom.grad[:,0:1]+1e-12)
        gaussian_denoms += dummy_feats_denom.grad[:, 0]
        dummy_feats_denom.grad.zero_()
    gaussian_features = gaussian_features / gaussian_denoms[..., None]
    if normalize:
        gaussian_features = gaussian_features / gaussian_features.norm(dim=-1, keepdim=True)
    # Replace nan values with 0
    print(
        "Number of NaN features",
        torch.isnan(gaussian_features).sum() // gaussian_features.shape[1],
    )
    gaussian_features[torch.isnan(gaussian_features)] = 0
    t2 = time.time()
    print("Time taken for feature distillation", t2 - t1)
    return gaussian_features

@dataclass
class FeatureParams:
    level: int = 0  # LangSplat Level
    feature_dir: str | None = None  # Directory for precomputed feature maps

@dataclass
class Args:
    data_dir: str = "./data/garden"  # colmap path
    checkpoint: str = "./data/garden/ckpts/ckpt_29999_rank0.pt"  # checkpoint path, can generate from original 3DGS repo
    results_dir: str = "./results/garden"  # output path
    rasterizer: Literal[
        "inria", "gsplat", "ply"
    ] | None = "gsplat"  # Original or GSplat for checkpoints
    format: Literal[
        "inria", "gsplat", "ply"
    ] = "gsplat"  # Original or GSplat for checkpoints
    data_factor: int = 4
    feature_field_batch_count: int = 1  # Number of batches to process for feature field
    run_feature_field_on_cpu: bool = False  # Run feature field on CPU
    feature: str = "lseg"  # Feature field type
    percentage_frames: int = 100  # Percentage of frames to process
    # feature_dir: str | None = None  # Directory for precomputed feature maps
    n_views: Union[
        int, None
    ] = None  # Number of views to process, None for to use percentage_frames,
    tag: str = "garden"
    prune: bool = True  # Whether to prune the splats before backprojection
    feature_params: FeatureParams = FeatureParams()  # Additional parameters for feature extractor

def main(
    args: Args
):
    data_dir = args.data_dir
    checkpoint = args.checkpoint
    results_dir = args.results_dir
    format = args.format or args.rasterizer
    if args.rasterizer:
        warnings.warn(
            "`rasterizer` is deprecated. Use `format` instead.", DeprecationWarning
        )
    if not format:
        raise ValueError("Must specify --format or the deprecated --rasterizer")
    data_factor = args.data_factor
    feature_field_batch_count = args.feature_field_batch_count
    run_feature_field_on_cpu = args.run_feature_field_on_cpu
    feature = args.feature
    percentage_frames = args.percentage_frames
    feature_dir = args.feature_params.feature_dir
    n_views = args.n_views
    tag = args.tag
    prune = args.prune
    level = args.feature_params.level

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")

    torch.set_default_device("cuda")

    # Check if the feature extractor is valid
    if feature not in BACKPROJECTION_FEATURE_EXTRACTORS:
        raise ValueError(
            f"Invalid feature extractor: {feature}. Available options: {', '.join(BACKPROJECTION_FEATURE_EXTRACTORS.keys())}"
        )

    os.makedirs(results_dir, exist_ok=True)
    splats = load_checkpoint(
        checkpoint, data_dir, format=format, data_factor=data_factor
    )

    if prune:
        splats_optimized = prune_by_gradients(splats)
        test_proper_pruning(splats, splats_optimized)
        splats = splats_optimized

    splats["data_dir"] = data_dir

    if feature in ["feature-map", "lang-splat"]:
        splats["feature_dir"] = feature_dir
        splats["level"] = level
    
    t1 = time.time()
    if n_views is not None:
        features = create_feature_field(splats, feature_type=feature, n_views=n_views)
        # torch.save(features, f"{results_dir}/features_{tag}_{feature}_{n_views}_views.pt")
    else:
        features = create_feature_field(
            splats, feature_type=feature, percentage_frames=percentage_frames
        )
    t2 = time.time()
    torch.save(
        features,
        f"{results_dir}/features_{feature.replace('-','_')}_{tag}.pt",
    )

    with open(f"{results_dir}/backproject_{feature.replace('-','_')}_{tag}.json", "w") as f:
        json.dump({
            "time_taken": t2 - t1
        }, f)


if __name__ == "__main__":
    cfg = tyro.cli(Args)

    print("Configuration:")
    pprint.pprint(to_builtin(cfg))
    print("\n")

    # Save config
    os.makedirs(cfg.results_dir, exist_ok=True)
    cfg_out_path = os.path.join(cfg.results_dir, f"backproject_{cfg.tag}.yaml")
    with open(cfg_out_path, "w") as f:
        yaml.dump(to_builtin(cfg), f)
    
    # Run Evaluation
    main(cfg)
