import math
import os
import time
from typing import Literal, Union
import torch
import tyro
from gsplat import rasterization
import pycolmap_scene_manager as pycolmap
import numpy as np
import matplotlib
from feature_extractors import (
    get_feature_extractor,
    BACKPROJECTION_FEATURE_EXTRACTORS,
)

matplotlib.use("TkAgg")  # To avoid conflict with cv2
from tqdm import tqdm
from lseg import LSegNet


from utils import (
    load_checkpoint,
    prune_by_gradients,
    test_proper_pruning,
    get_frames,
)


def create_feature_field(
    splats, feature_type="lseg", use_cpu=False, percentage_frames=100, n_views=None
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not use_cpu else "cpu"
    )
    feature_extractor = get_feature_extractor(feature_type, device)
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors_all = torch.cat([colors_dc, colors_rest], dim=1)

    colors = colors_dc[:, 0, :]  # * 0
    colors_0 = colors_dc[:, 0, :] * 0
    colmap_project = splats["colmap_project"]

    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    K = splats["camera_matrix"]
    colors.requires_grad = True
    colors_0.requires_grad = True

    gaussian_features = torch.zeros(
        colors.shape[0], feature_extractor.dim, device=colors.device
    )
    gaussian_denoms = torch.ones(colors.shape[0], device=colors.device) * 1e-12

    t1 = time.time()

    colors_feats = torch.zeros(
        colors.shape[0], feature_extractor.dim, device=colors.device
    )
    colors_feats.requires_grad = True
    colors_feats_0 = torch.zeros(colors.shape[0], 3, device=colors.device)
    colors_feats_0.requires_grad = True

    print("Distilling features...")
    for frame in tqdm(
        get_frames(colmap_project, percentage_frames=percentage_frames, n_views=n_views)
    ):

        width = int(K[0, 2] * 2)
        height = int(K[1, 2] * 2)
        viewmat = frame["viewmat"].to(device)
        with torch.no_grad():
            output, _, meta = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors_all,
                viewmat[None],
                K[None],
                width=width,
                height=height,
                sh_degree=3,
            )
            feats = feature_extractor.extract_features(output[0])

        output_for_grad, _, meta = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors_feats,
            viewmat[None],
            K[None],
            width=width,
            height=height,
        )

        target = (output_for_grad[0] * feats).mean()

        target.backward()

        colors_feats_copy = colors_feats.grad.clone()

        colors_feats.grad.zero_()

        output_for_grad, _, meta = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors_feats_0,
            viewmat[None],
            K[None],
            width=width,
            height=height,
        )

        target_0 = (output_for_grad[0]).mean()

        target_0.backward()

        gaussian_features += colors_feats_copy  # / (colors_feats_0.grad[:,0:1]+1e-12)
        gaussian_denoms += colors_feats_0.grad[:, 0]
        colors_feats_0.grad.zero_()
    gaussian_features = gaussian_features / gaussian_denoms[..., None]
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


def main(
    data_dir: str = "./data/garden",  # colmap path
    checkpoint: str = "./data/garden/ckpts/ckpt_29999_rank0.pt",  # checkpoint path, can generate from original 3DGS repo
    results_dir: str = "./results/garden",  # output path
    rasterizer: Literal[
        "inria", "gsplat"
    ] = "gsplat",  # Original or GSplat for checkpoints
    data_factor: int = 4,
    feature_field_batch_count: int = 1,  # Number of batches to process for feature field
    run_feature_field_on_cpu: bool = False,  # Run feature field on CPU
    feature: str = "lseg",  # Feature field type
    percentage_frames: int = 100,  # Percentage of frames to process
    n_views: Union[
        int, None
    ] = None,  # Number of views to process, None for to use percentage_frames
):

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
        checkpoint, data_dir, rasterizer=rasterizer, data_factor=data_factor
    )
    
    splats_optimized = prune_by_gradients(splats)
    test_proper_pruning(splats, splats_optimized)
    splats = splats_optimized
    
    if n_views is not None:
        features = create_feature_field(splats, feature_type=feature, n_views=n_views)
        torch.save(features, f"{results_dir}/features_{feature}_{n_views}_views.pt")
    else:
        features = create_feature_field(
            splats, feature_type=feature, percentage_frames=percentage_frames
        )
        torch.save(
            features,
            f"{results_dir}/features_{feature}_{percentage_frames}_percentage.pt",
        )


if __name__ == "__main__":
    tyro.cli(main)
