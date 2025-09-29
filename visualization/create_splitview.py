from typing import Literal
import tyro
import os
import sys
import torch
import cv2
import imageio  # To generate gifs
from gsplat import rasterization
import numpy as np
import matplotlib
from sklearn.decomposition import PCA

matplotlib.use("TkAgg")

# Add parent directory to sys.path so utils.py can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import (
    prune_by_gradients,
    test_proper_pruning,
    get_viewmat_from_colmap_image,
    load_checkpoint,
    torch_to_cv,
)




def render_pcd(
    splats,
    output_path,
):
    cv2.namedWindow("PCD", cv2.WINDOW_NORMAL)
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

    for image in sorted(
        splats["colmap_project"].images.values(), key=lambda x: x.name
    ):
        viewmat = get_viewmat_from_colmap_image(image)
        colors_rendered, alphas, meta = rasterization(
            means,
            quats,
            scales * 0.0,
            opacities,
            colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=K[0, 2] * 2,
            height=K[1, 2] * 2,
            sh_degree=3,
        )
        colors_rendered = colors_rendered[0]
        
        frame = torch_to_cv(colors_rendered)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(frame[..., ::-1])
        cv2.imshow("PCD", frame)
        cv2.waitKey(10)
    cv2.destroyAllWindows()
    imageio.mimsave(output_path, frames, fps=20, loop=0)


def render_rgb(
    splats,
    output_path,
):
    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
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

    for image in sorted(
        splats["colmap_project"].images.values(), key=lambda x: x.name
    ):
        viewmat = get_viewmat_from_colmap_image(image)
        colors_rendered, alphas, meta = rasterization(
            means,
            quats,
            scales * 1.0,
            opacities,
            colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=K[0, 2] * 2,
            height=K[1, 2] * 2,
            sh_degree=3,
        )
        colors_rendered = colors_rendered[0]
        
        frame = torch_to_cv(colors_rendered)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(frame[..., ::-1])
        cv2.imshow("RGB", frame)
        cv2.waitKey(10)
    cv2.destroyAllWindows()
    imageio.mimsave(output_path, frames, fps=20, loop=0)

def render_splitview(
    splats,
    output_path,
):
    cv2.namedWindow("SplitView", cv2.WINDOW_NORMAL)
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

    for image in sorted(
        splats["colmap_project"].images.values(), key=lambda x: x.name
    ):
        viewmat = get_viewmat_from_colmap_image(image)
        pcd_rendered, alphas, meta = rasterization(
            means,
            quats,
            scales * 0.0,
            opacities,
            colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=K[0, 2] * 2,
            height=K[1, 2] * 2,
            sh_degree=3,
        )
        pcd_rendered = pcd_rendered[0]

        colors_rendered, alphas, meta = rasterization(
            means,
            quats,
            scales * 1.0,
            opacities,
            colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=K[0, 2] * 2,
            height=K[1, 2] * 2,
            sh_degree=3,
        )
        colors_rendered = colors_rendered[0]

        frame1 = torch_to_cv(pcd_rendered)
        frame2 = torch_to_cv(colors_rendered)
        h, w, c = frame1.shape
        frame = np.zeros((h, w, c), dtype=np.uint8)
        frame[:, : w // 2, :] = frame1[:, : w // 2, :]
        frame[:, w // 2 :, :] = frame2[:, w // 2 :, :]
        
        # frame = torch_to_cv(colors_rendered)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(frame[..., ::-1])
        cv2.imshow("SplitView", frame)
        cv2.waitKey(10)
    cv2.destroyAllWindows()
    imageio.mimsave(output_path, frames, fps=20, loop=0)

def main(
    data_dir: str = "./data/garden",  # colmap path
    checkpoint: str = "./data/garden/ckpts/ckpt_29999_rank0.pt",  # checkpoint path
    results_dir: str = "./results/garden",  # output path
    format: Literal[
        "inria", "gsplat", "ply"
    ] = "gsplat",  # Original or gsplat for checkpoints
    data_factor: int = 4,
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


    if tag is None:
        tag = os.path.basename(data_dir)

    
    render_pcd(splats, f"{results_dir}/pcd_{tag}.gif")
    render_rgb(splats, f"{results_dir}/rgb_{tag}.gif")
    render_splitview(splats, f"{results_dir}/splitview_{tag}.gif")
    


if __name__ == "__main__":
    tyro.cli(main)
