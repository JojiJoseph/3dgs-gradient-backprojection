import math
import os
import time
from typing import Literal
import torch
import tyro
from gsplat import rasterization
import pycolmap_scene_manager as pycolmap
import numpy as np
import matplotlib
matplotlib.use("TkAgg") # To avoid conflict with cv2
from tqdm import tqdm
from lseg import LSegNet
import cv2
from utils import load_checkpoint_blender, get_viewmat_from_blender_frame


def torch_to_cv(tensor):
    img_cv = tensor.detach().cpu().numpy()[..., ::-1]
    img_cv = np.clip(img_cv * 255, 0, 255).astype(np.uint8)
    return img_cv


def _detach_tensors_from_dict(d, inplace=True):
    if not inplace:
        d = d.copy()
    for key in d:
        if isinstance(d[key], torch.Tensor):
            d[key] = d[key].detach()
    return d


def get_viewmat_from_colmap_image(image):
    viewmat = torch.eye(4).float()  # .to(device)
    viewmat[:3, :3] = torch.tensor(image.R()).float()  # .to(device)
    viewmat[:3, 3] = torch.tensor(image.t).float()  # .to(device)
    return viewmat


def prune_by_gradients(splats):
    colmap_project = splats["colmap_project"]
    frame_idx = 0
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    K = splats["camera_matrix"]
    colors.requires_grad = True
    gaussian_grads = torch.zeros(colors.shape[0], device=colors.device)
    for image in sorted(colmap_project.images.values(), key=lambda x: x.name):
        viewmat = get_viewmat_from_colmap_image(image)
        output, _, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors[:, 0, :],
            viewmats=viewmat[None],
            Ks=K[None],
            # sh_degree=3,
            width=K[0, 2] * 2,
            height=K[1, 2] * 2,
        )
        frame_idx += 1
        pseudo_loss = ((output.detach() + 1 - output) ** 2).mean()
        pseudo_loss.backward()
        # print(colors.grad.shape)
        gaussian_grads += (colors.grad[:, 0]).norm(dim=[1])
        colors.grad.zero_()

    mask = gaussian_grads > 0
    print("Total splats", len(gaussian_grads))
    print("Pruned", (~mask).sum(), "splats")
    print("Remaining", mask.sum(), "splats")
    splats = splats.copy()
    splats["means"] = splats["means"][mask]
    splats["features_dc"] = splats["features_dc"][mask]
    splats["features_rest"] = splats["features_rest"][mask]
    splats["scaling"] = splats["scaling"][mask]
    splats["rotation"] = splats["rotation"][mask]
    splats["opacity"] = splats["opacity"][mask]
    return splats


def test_proper_pruning(splats, splats_after_pruning):
    colmap_project = splats["colmap_project"]
    frame_idx = 0
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]

    means_pruned = splats_after_pruning["means"]
    colors_dc_pruned = splats_after_pruning["features_dc"]
    colors_rest_pruned = splats_after_pruning["features_rest"]
    colors_pruned = torch.cat([colors_dc_pruned, colors_rest_pruned], dim=1)
    opacities_pruned = torch.sigmoid(splats_after_pruning["opacity"])
    scales_pruned = torch.exp(splats_after_pruning["scaling"])
    quats_pruned = splats_after_pruning["rotation"]

    K = splats["camera_matrix"]
    total_error = 0
    max_pixel_error = 0
    for image in sorted(colmap_project.images.values(), key=lambda x: x.name):
        viewmat = get_viewmat_from_colmap_image(image)
        output, _, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats=viewmat[None],
            Ks=K[None],
            sh_degree=3,
            width=K[0, 2] * 2,
            height=K[1, 2] * 2,
        )

        output_pruned, _, _ = rasterization(
            means_pruned,
            quats_pruned,
            scales_pruned,
            opacities_pruned,
            colors_pruned,
            viewmats=viewmat[None],
            Ks=K[None],
            sh_degree=3,
            width=K[0, 2] * 2,
            height=K[1, 2] * 2,
        )

        total_error += torch.abs((output - output_pruned)).sum()
        max_pixel_error = max(
            max_pixel_error, torch.abs((output - output_pruned)).max()
        )

    percentage_pruned = (
        (len(splats["means"]) - len(splats_after_pruning["means"]))
        / len(splats["means"])
        * 100
    )

    assert max_pixel_error < 1 / (
        255 * 2
    ), "Max pixel error should be less than 1/(255*2), safety margin"
    print(
        "Report {}% pruned, max pixel error = {}, total pixel error = {}".format(
            percentage_pruned, max_pixel_error, total_error
        )
    )


def create_feature_field_lseg_blender(splats, use_cpu=False):
    device = "cpu" if use_cpu else "cuda"

    net = LSegNet(
        backbone="clip_vitl16_384",
        features=256,
        crop_size=480,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    )
    # Load pre-trained weights
    net.load_state_dict(torch.load("./checkpoints/lseg_minimal_e200.ckpt", map_location=device))
    net.eval()
    net.to(device)

    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors_all = torch.cat([colors_dc, colors_rest], dim=1)

    colors = colors_dc[:, 0, :]  # * 0
    colors_0 = colors_dc[:, 0, :] * 0
    colors.to(device)
    colors_0.to(device)

    # colmap_project = splats["colmap_project"]

    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    K = splats["camera_matrix"]
    colors.requires_grad = True
    colors_0.requires_grad = True

    gaussian_features = torch.zeros(colors.shape[0], 512, device=colors.device)
    gaussian_denoms = torch.ones(colors.shape[0], device=colors.device) * 1e-12

    colors_feats = torch.zeros(colors.shape[0], 512, device=colors.device, requires_grad=True)
    colors_feats_0 = torch.zeros(colors.shape[0], 3, device=colors.device, requires_grad=True)

    for frame in tqdm(splats["transforms"]["frames"]):

        viewmat = torch.tensor(frame["transform_matrix"]).float().to(device)
        viewmat[:3, :3] = viewmat[:3, :3] @ torch.tensor(([1, 0, 0], [0, -1, 0], [0, 0, -1])).float().to(device)  # Flip Y axis
        viewmat = torch.linalg.inv(viewmat)  # Convert to camera-to-world matrix
        K = torch.tensor(splats["camera_matrix"]).float().to(device)
        width = int(K[0, 2] * 2)
        height = int(K[1, 2] * 2)

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

            output = torch.nn.functional.interpolate(
                output.permute(0, 3, 1, 2).to(device),
                size=(480, 480),
                mode="bilinear",
            )
            output.to(device)
            feats = net.forward(output)
            feats = torch.nn.functional.normalize(feats, dim=1)
            feats = torch.nn.functional.interpolate(
                feats, size=(height, width), mode="bilinear"
            )[0]
            feats = feats.permute(1, 2, 0)

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

        target = (output_for_grad[0].to(device) * feats).sum()
        target.to(device)
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

        target_0 = (output_for_grad[0]).sum()
        target_0.to(device)
        target_0.backward()

        gaussian_features += colors_feats_copy
        gaussian_denoms += colors_feats_0.grad[:, 0]
        colors_feats_0.grad.zero_()

        # Clean up unused variables and free GPU memory
        del viewmat, meta, _, output, feats, output_for_grad, colors_feats_copy, target, target_0
        torch.cuda.empty_cache()
    gaussian_features = gaussian_features / gaussian_denoms[..., None]
    gaussian_features = gaussian_features / gaussian_features.norm(dim=-1, keepdim=True)
    # Replace nan values with 0
    gaussian_features[torch.isnan(gaussian_features)] = 0
    return gaussian_features

def create_feature_field_identity(splats, use_cpu=False):
    print(splats.keys())
    print(splats["blender_img_dir"])
    mask_seg_dir = os.path.join(splats["blender_img_dir"], "..", "masks_seg")
    # exit()
    # print(mask_seg_dir)
    # exit()
    device = "cpu" if use_cpu else "cuda"

    net = LSegNet(
        backbone="clip_vitl16_384",
        features=256,
        crop_size=480,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    )
    # Load pre-trained weights
    net.load_state_dict(torch.load("./checkpoints/lseg_minimal_e200.ckpt", map_location=device))
    net.eval()
    net.to(device)

    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors_all = torch.cat([colors_dc, colors_rest], dim=1)

    colors = colors_dc[:, 0, :]  # * 0
    colors_0 = colors_dc[:, 0, :] * 0
    colors.to(device)
    colors_0.to(device)

    # colmap_project = splats["colmap_project"]

    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    K = splats["camera_matrix"]
    colors.requires_grad = True
    colors_0.requires_grad = True

    gaussian_features = torch.zeros(colors.shape[0], 4, device=colors.device)
    gaussian_denoms = torch.ones(colors.shape[0], device=colors.device) * 1e-12

    colors_feats = torch.zeros(colors.shape[0], 4, device=colors.device, requires_grad=True)
    colors_feats_0 = torch.zeros(colors.shape[0], 3, device=colors.device, requires_grad=True)

    codebook = np.array([
        [0.0, 0.0, 0.0],  # Black
        [255.0, 0, 0],  # Red
        [0, 255.0, 0],  # Green
        [0, 0, 255.0],  # Blue
    ])

    for frame in tqdm(splats["transforms"]["frames"]):

        viewmat = get_viewmat_from_blender_frame(frame)
        image_name = frame["file_path"].split("/")[-1]
        mask_seg_path = os.path.join(mask_seg_dir, image_name)
        mask_seg = cv2.imread(mask_seg_path, cv2.IMREAD_COLOR)
        # Find codebook index for each pixel
        mask_seg = cv2.cvtColor(mask_seg, cv2.COLOR_BGR2RGB)
        mask_seg_indices = np.argmin(
            np.linalg.norm(mask_seg[:, :, None] - codebook[None, None, :], axis=-1), axis=-1
        )

        mask_features = np.eye(len(codebook))[mask_seg_indices].astype(np.float32)
        # print("Mask features shape:", mask_features.shape)

        # print("Mask seg indices shape:", mask_seg_indices.shape)
        # exit()

        mask_features = torch.tensor(mask_features).float().to(device)

        K = torch.tensor(splats["camera_matrix"]).float().to(device)
        width = int(K[0, 2] * 2)
        height = int(K[1, 2] * 2)

        # with torch.no_grad():
        #     output, _, meta = rasterization(
        #         means,
        #         quats,
        #         scales,
        #         opacities,
        #         colors_all,
        #         viewmat[None],
        #         K[None],
        #         width=width,
        #         height=height,
        #         sh_degree=3,
        #     )

            # output = torch.nn.functional.interpolate(
            #     output.permute(0, 3, 1, 2).to(device),
            #     size=(480, 480),
            #     mode="bilinear",
            # )
            # output.to(device)
            # feats = net.forward(output)
            # feats = torch.nn.functional.normalize(feats, dim=1)
            # feats = torch.nn.functional.interpolate(
            #     feats, size=(height, width), mode="bilinear"
            # )[0]
            # feats = feats.permute(1, 2, 0)

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

        target = (output_for_grad[0].to(device) * mask_features).sum()
        target.to(device)
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

        target_0 = (output_for_grad[0]).sum()
        target_0.to(device)
        target_0.backward()

        gaussian_features += colors_feats_copy
        gaussian_denoms += colors_feats_0.grad[:, 0]
        colors_feats_0.grad.zero_()

        # Clean up unused variables and free GPU memory
        del viewmat, meta, _, output_for_grad, colors_feats_copy, target, target_0
        torch.cuda.empty_cache()
    gaussian_features = gaussian_features / gaussian_denoms[..., None]
    gaussian_features = gaussian_features / gaussian_features.norm(dim=-1, keepdim=True)
    # Replace nan values with 0
    gaussian_features[torch.isnan(gaussian_features)] = 0
    return gaussian_features

def main(
    data_dir: str = "./data/garden",  # blender path
    checkpoint: str = "./data/garden/ckpts/ckpt_29999_rank0.pt",  # checkpoint path, can generate from original 3DGS repo
    results_dir: str = "./results/garden",  # output path
    rasterizer: Literal[
        "inria", "gsplat"
    ] = "gsplat",  # Original or GSplat for checkpoints
    data_factor: int = 4,  # Data factor for downsampling
    feature_field_batch_count: int = 1,  # Number of batches to process for feature field
    run_feature_field_on_cpu: bool = False,  # Run feature field on CPU
):

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")

    torch.set_default_device("cuda")

    os.makedirs(results_dir, exist_ok=True)
    splats = load_checkpoint_blender(
        checkpoint, data_dir, rasterizer=rasterizer, data_factor=data_factor
    )

    features = create_feature_field_identity(splats)
    torch.save(features, f"{results_dir}/features_identity.pt")


if __name__ == "__main__":
    tyro.cli(main)
