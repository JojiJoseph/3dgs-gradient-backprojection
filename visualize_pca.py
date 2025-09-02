from copy import deepcopy
from typing import Literal
import tyro
import os
import torch
import cv2
import imageio  # To generate gifs
import pycolmap_scene_manager as pycolmap
from gsplat import rasterization
import numpy as np
import clip
import matplotlib
from sklearn.decomposition import PCA

matplotlib.use("TkAgg")

from lseg import LSegNet
from utils import (
    prune_by_gradients,
    test_proper_pruning,
    get_viewmat_from_colmap_image,
    load_checkpoint,
    torch_to_cv,
)


def render_pca(
    splats,
    features,
    output_path,
    pca_on_gaussians=True,
    scale=1.0,
    feedback=True,
):
    if feedback:
        cv2.destroyAllWindows()
        cv2.namedWindow("PCA", cv2.WINDOW_NORMAL)
    frames = []
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    K = splats["camera_matrix"]
    aux_dir = output_path + ".images"
    os.makedirs(aux_dir, exist_ok=True)

    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features.detach().cpu().numpy())
    feats_min = np.min(features_pca, axis=(0, 1))
    feats_max = np.max(features_pca, axis=(0, 1))
    features_pca = (features_pca - feats_min) / (feats_max - feats_min)
    features_pca = torch.tensor(features_pca).float().cuda()
    if pca_on_gaussians:
        for image in sorted(
            splats["colmap_project"].images.values(), key=lambda x: x.name
        ):
            viewmat = get_viewmat_from_colmap_image(image)
            features_rendered, alphas, meta = rasterization(
                means,
                quats,
                scales * scale,
                opacities,
                features_pca,
                viewmats=viewmat[None],
                Ks=K[None],
                width=K[0, 2] * 2,
                height=K[1, 2] * 2,
                # sh_degree=3,
            )
            features_rendered = features_rendered[0]
            frame = torch_to_cv(features_rendered)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            frames.append(frame)
            if feedback:
                cv2.imshow("PCA", frame[..., ::-1])
                cv2.imwrite(f"{aux_dir}/{image.name}", frame[..., ::-1])
                cv2.waitKey(1)
    else:
        for image in sorted(
            splats["colmap_project"].images.values(), key=lambda x: x.name
        ):
            viewmat = get_viewmat_from_colmap_image(image)
            features_rendered, alphas, meta = rasterization(
                means,
                quats,
                scales * scale,
                opacities,
                features,
                viewmats=viewmat[None],
                Ks=K[None],
                width=K[0, 2] * 2,
                height=K[1, 2] * 2,
                # sh_degree=3,
            )
            features_rendered = features_rendered[0]
            h, w, c = features_rendered.shape
            features_rendered = (
                features_rendered.reshape(h * w, c).detach().cpu().numpy()
            )
            features_rendered = pca.transform(features_rendered)
            features_rendered = features_rendered.reshape(h, w, 3)
            features_rendered = (features_rendered - feats_min) / (
                feats_max - feats_min
            )
            frame = (features_rendered * 255).astype(np.uint8)
            frames.append(frame[..., ::-1])
            if feedback:
                cv2.imshow("PCA", frame)
                cv2.imwrite(f"{aux_dir}/{image.name}", frame)
                cv2.waitKey(1)
    imageio.mimsave(output_path, frames, fps=10, loop=0)
    if feedback:
        cv2.destroyAllWindows()


def main(
    data_dir: str = "./data/garden",  # colmap path
    checkpoint: str = "./data/garden/ckpts/ckpt_29999_rank0.pt",  # checkpoint path, can generate from original 3DGS repo
    feature_checkpoint: str = "./data/garden/features_dino.pt",  # path to features, can generate from original 3DGS repo
    results_dir: str = "./results/garden",  # output path
    format: Literal[
        "inria", "gsplat"
    ] = "gsplat",  # Original or gsplat for checkpoints
    data_factor: int = 4,
    show_visual_feedback: bool = True,
    tag: str = None,
    prune: bool = True
):

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")

    torch.set_default_device("cuda")

    os.makedirs(results_dir, exist_ok=True)
    splats = load_checkpoint(
        checkpoint, data_dir, format=format, data_factor=data_factor
    )
    if prune:
        splats_optimized = prune_by_gradients(splats)
        test_proper_pruning(splats, splats_optimized)
        splats = splats_optimized
    # if feature == "lseg":
    #     features = torch.load(f"{results_dir}/features_lseg.pt")
    # elif feature == "dino":
    #     features = torch.load(f"{results_dir}/features_dino.pt")
    features = torch.load(feature_checkpoint)

    if tag is None:
        tag = os.path.basename(feature_checkpoint).split(".")[0]


    render_pca(
        splats,
        features,
        f"{results_dir}/pca_gaussians_{tag}.gif",
        pca_on_gaussians=True,
        scale=0.20,
        feedback=show_visual_feedback,
    )

    render_pca(
        splats,
        features,
        f"{results_dir}/pca_renderings_{tag}.gif",
        pca_on_gaussians=False,
        feedback=show_visual_feedback,
    )


if __name__ == "__main__":
    tyro.cli(main)
