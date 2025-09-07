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
import warnings

matplotlib.use("TkAgg")

from lseg import LSegNet
from utils import (
    create_checkerboard,
    prune_by_gradients,
    test_proper_pruning,
    get_viewmat_from_colmap_image,
    load_checkpoint,
)


def get_mask3d_lseg(splats, features, prompt, neg_prompt, threshold=None):

    net = LSegNet(
        backbone="clip_vitl16_384",
        features=256,
        crop_size=480,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    )
    # Load pre-trained weights
    net.load_state_dict(torch.load("./checkpoints/lseg_minimal_e200.ckpt"))
    net.eval()
    net.cuda()

    # Preprocess the text prompt
    clip_text_encoder = net.clip_pretrained.encode_text

    pos_prompt_length = len(prompt.split(";"))

    prompts = prompt.split(";") + neg_prompt.split(";")

    prompt = clip.tokenize(prompts)
    prompt = prompt.cuda()

    text_feat = clip_text_encoder(prompt)  # N, 512, N - number of prompts
    text_feat_norm = torch.nn.functional.normalize(text_feat, dim=1)

    features = torch.nn.functional.normalize(features, dim=1)
    score = features @ text_feat_norm.float().T
    mask_3d = score[:, :pos_prompt_length].max(dim=1)[0] > score[:, pos_prompt_length:].max(dim=1)[0]
    if threshold is not None:
        mask_3d = mask_3d & (score[:, 0] > threshold)
    mask_3d_inv = ~mask_3d

    return mask_3d, mask_3d_inv


def apply_mask3d(splats, mask3d, mask3d_inverted):
    if mask3d_inverted == None:
        mask3d_inverted = ~mask3d
    extracted = deepcopy(splats)
    deleted = deepcopy(splats)
    masked = deepcopy(splats)
    extracted["means"] = extracted["means"][mask3d]
    extracted["features_dc"] = extracted["features_dc"][mask3d]
    extracted["features_rest"] = extracted["features_rest"][mask3d]
    extracted["scaling"] = extracted["scaling"][mask3d]
    extracted["rotation"] = extracted["rotation"][mask3d]
    extracted["opacity"] = extracted["opacity"][mask3d]

    deleted["means"] = deleted["means"][mask3d_inverted]
    deleted["features_dc"] = deleted["features_dc"][mask3d_inverted]
    deleted["features_rest"] = deleted["features_rest"][mask3d_inverted]
    deleted["scaling"] = deleted["scaling"][mask3d_inverted]
    deleted["rotation"] = deleted["rotation"][mask3d_inverted]
    deleted["opacity"] = deleted["opacity"][mask3d_inverted]

    masked["features_dc"][mask3d] = 1  # (1 - 0.5) / 0.2820947917738781
    masked["features_dc"][~mask3d] = 0  # (0 - 0.5) / 0.2820947917738781
    masked["features_rest"][~mask3d] = 0

    return extracted, deleted, masked


def render_to_gif(
    output_path: str,
    splats,
    feedback: bool = False,
    use_checkerboard_background: bool = False,
    no_sh: bool = False,
):
    if feedback:
        cv2.destroyAllWindows()
        cv2.namedWindow("Rendering", cv2.WINDOW_NORMAL)
    frames = []
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    if no_sh == True:
        colors = colors_dc[:, 0, :]
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    K = splats["camera_matrix"]
    aux_dir = output_path + ".images"
    os.makedirs(aux_dir, exist_ok=True)
    for image in sorted(splats["colmap_project"].images.values(), key=lambda x: x.name):
        viewmat = get_viewmat_from_colmap_image(image)
        output, alphas, meta = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmat[None],
            K[None],
            width=K[0, 2] * 2,
            height=K[1, 2] * 2,
            sh_degree=3 if not no_sh else None,
        )
        frame = np.clip(output[0].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
        if use_checkerboard_background:
            checkerboard = create_checkerboard(frame.shape[1], frame.shape[0])
            alphas = alphas[0].detach().cpu().numpy()
            frame = frame * alphas + checkerboard * (1 - alphas)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(frame)
        if feedback:
            cv2.imshow("Rendering", frame[..., ::-1])
            cv2.imwrite(f"{aux_dir}/{image.name}", frame[..., ::-1])
            cv2.waitKey(1)
    if output_path is not None:
        imageio.mimsave(output_path, frames, fps=10, loop=0)
    if feedback:
        cv2.destroyAllWindows()


def render_mask_2d_to_gif(
    splats,
    features,
    prompt,
    neg_prompt,
    output_path: str,
    feedback: bool = False,
):
    if feedback:
        cv2.destroyAllWindows()
        cv2.namedWindow("Rendering", cv2.WINDOW_NORMAL)
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

    net = LSegNet(
        backbone="clip_vitl16_384",
        features=256,
        crop_size=480,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    )
    # Load pre-trained weights
    net.load_state_dict(torch.load("./checkpoints/lseg_minimal_e200.ckpt"))
    net.eval()
    net.cuda()

    # Preprocess the text prompt
    clip_text_encoder = net.clip_pretrained.encode_text
    pos_prompt_length = len(prompt.split(";"))

    prompts = prompt.split(";") + neg_prompt.split(";")

    prompt = clip.tokenize(prompts)
    prompt = prompt.cuda()

    text_feat = clip_text_encoder(prompt)  # N, 512, N - number of prompts
    text_feat_norm = torch.nn.functional.normalize(text_feat, dim=1)

    # features = torch.nn.functional.normalize(features, dim=1)

    for image in sorted(splats["colmap_project"].images.values(), key=lambda x: x.name):
        viewmat = get_viewmat_from_colmap_image(image)
        output, alphas, meta = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=K[0, 2] * 2,
            height=K[1, 2] * 2,
            sh_degree=3,
        )
        feats_rendered, _, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            features,
            viewmats=viewmat[None],
            Ks=K[None],
            width=K[0, 2] * 2,
            height=K[1, 2] * 2,
            # sh_degree=3,
        )
        feats_rendered = feats_rendered[0]
        feats_rendered = torch.nn.functional.normalize(feats_rendered, dim=-1)
        score = feats_rendered @ text_feat_norm.float().T
        mask2d = score[..., :pos_prompt_length].max(dim=2)[0] > score[..., pos_prompt_length:].max(dim=2)[0]
        # print(mask2d.shape)
        mask2d = mask2d[..., None].detach().cpu().numpy()
        frame = np.clip(output[0].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
        frame = frame * (
            0.75 + 0.25 * mask2d * np.array([255, 0, 0]) + (1 - mask2d) * 0.25
        )
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(frame)
        if feedback:
            cv2.imshow("Rendering", frame[..., ::-1])
            cv2.imwrite(f"{aux_dir}/{image.name}", frame[..., ::-1])
            cv2.waitKey(1)
    if output_path is not None:
        imageio.mimsave(output_path, frames, fps=10, loop=0)
    if feedback:
        cv2.destroyAllWindows()


def save_to_ckpt(
    output_path: str,
    splats,
):
    # Save Torch Checkpoint
    checkpoint_data = {
        "splats": {
            "means": splats["means"],
            "quats": splats["rotation"],
            "scales": splats["scaling"],
            "opacities": splats["opacity"],
            "sh0": splats["features_dc"],
            "shN": splats["features_rest"],
        }
    }
    torch.save(checkpoint_data, output_path)


def main(
    data_dir: str = "./data/garden",  # colmap path
    checkpoint: str = "./data/garden/ckpts/ckpt_29999_rank0.pt",  # checkpoint path, can generate from original 3DGS repo
    feature_path: str = "./results/garden/features_lseg_garden.pt",  # path to precomputed features
    results_dir: str = "./results/garden",  # output path
    rasterizer: Literal[
        "inria", "gsplat"
    ] | None = None,  # Original or gsplat for checkpoints,
    format: Literal[
        "inria", "gsplat", "ply"
    ] = "gsplat",  # Original or gsplat for checkpoints
    prompt: str = "Table",
    neg_prompt: str = "Vase;Other",
    data_factor: int = 4,
    show_visual_feedback: bool = True,
    export_checkpoint: bool = False,
    prune: bool = True,
):

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")

    torch.set_default_device("cuda")

    os.makedirs(results_dir, exist_ok=True)
    format = format or rasterizer
    if rasterizer:
        warnings.warn(
            "`rasterizer` is deprecated. Use `format` instead.", DeprecationWarning
        )
    if not format:
        raise ValueError("Must specify --format or the deprecated --rasterizer")
    splats = load_checkpoint(
        checkpoint, data_dir, format=format, data_factor=data_factor
    )
    if prune:
        splats_optimized = prune_by_gradients(splats)
        test_proper_pruning(splats, splats_optimized)
        splats = splats_optimized
    features = torch.load(feature_path)
    mask3d, mask3d_inv = get_mask3d_lseg(splats, features, prompt, neg_prompt)
    extracted, deleted, masked = apply_mask3d(splats, mask3d, mask3d_inv)

    render_mask_2d_to_gif(
        splats,
        features,
        prompt,
        neg_prompt,
        f"{results_dir}/mask2d.gif",
        show_visual_feedback,
    )

    render_to_gif(
        f"{results_dir}/extracted.gif",
        extracted,
        show_visual_feedback,
        use_checkerboard_background=True,
    )
    render_to_gif(f"{results_dir}/deleted.gif", deleted, show_visual_feedback)

    if export_checkpoint:
        save_to_ckpt(f"{results_dir}/extracted.pt", extracted)
        save_to_ckpt(f"{results_dir}/deleted.pt", deleted)


if __name__ == "__main__":
    tyro.cli(main)
