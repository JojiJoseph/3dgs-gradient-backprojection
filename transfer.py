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


def load_checkpoint(
    checkpoint: str,
    data_dir: str,
    rasterizer: Literal["inria", "gsplat"] = "inria",
    data_factor: int = 1,
):

    colmap_dir = os.path.join(data_dir, "sparse/0/")
    if not os.path.exists(colmap_dir):
        colmap_dir = os.path.join(data_dir, "sparse")
    assert os.path.exists(colmap_dir), f"COLMAP directory {colmap_dir} does not exist."

    colmap_project = pycolmap.SceneManager(colmap_dir)
    colmap_project.load_cameras()
    colmap_project.load_images()
    colmap_project.load_points3D()
    model = torch.load(checkpoint)  # Make sure it is generated by 3DGS original repo
    if rasterizer == "inria":
        model_params, _ = model
        splats = {
            "active_sh_degree": model_params[0],
            "means": model_params[1],
            "features_dc": model_params[2],
            "features_rest": model_params[3],
            "scaling": model_params[4],
            "rotation": model_params[5],
            "opacity": model_params[6].squeeze(1),
        }
    elif rasterizer == "gsplat":
        print(model["splats"].keys())
        model_params = model["splats"]
        splats = {
            "active_sh_degree": 3,
            "means": model_params["means"],
            "features_dc": model_params["sh0"],
            "features_rest": model_params["shN"],
            "scaling": model_params["scales"],
            "rotation": model_params["quats"],
            "opacity": model_params["opacities"],
        }
    else:
        raise ValueError("Invalid rasterizer")

    _detach_tensors_from_dict(splats)

    # Assuming only one camera
    for camera in colmap_project.cameras.values():
        camera_matrix = torch.tensor(
            [
                [camera.fx, 0, camera.cx],
                [0, camera.fy, camera.cy],
                [0, 0, 1],
            ]
        )
        break

    camera_matrix[:2, :3] /= data_factor

    splats["camera_matrix"] = camera_matrix
    splats["colmap_project"] = colmap_project
    splats["colmap_dir"] = data_dir

    return splats


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


def create_feature_field_lseg(splats):

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

    gaussian_features = torch.zeros(colors.shape[0], 512, device=colors.device)
    gaussian_denoms = torch.ones(colors.shape[0], device=colors.device) * 1e-12

    t1 = time.time()

    colors_feats = torch.zeros(colors.shape[0], 512, device=colors.device)
    colors_feats.requires_grad = True
    colors_feats_0 = torch.zeros(colors.shape[0], 3, device=colors.device)
    colors_feats_0.requires_grad = True

    for image in tqdm(
        sorted(colmap_project.images.values(), key=lambda x: x.name),
        desc="Feature backprojection",
    ):

        image_name = image.name  # .split(".")[0] + ".jpg"

        viewmat = get_viewmat_from_colmap_image(image)

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
                output.permute(0, 3, 1, 2).cuda(),
                size=(480, 480),
                mode="bilinear",
            )
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

        target = (output_for_grad[0] * feats).sum()

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

        target_0.backward()

        gaussian_features += colors_feats_copy
        gaussian_denoms += colors_feats_0.grad[:, 0]
        colors_feats_0.grad.zero_()
    gaussian_features = gaussian_features / gaussian_denoms[..., None]
    gaussian_features = gaussian_features / gaussian_features.norm(dim=-1, keepdim=True)
    # Replace nan values with 0
    # print("NaN features", torch.isnan(gaussian_features).sum())
    gaussian_features[torch.isnan(gaussian_features)] = 0
    t2 = time.time()
    print("Time taken for feature backprojection", t2 - t1)
    return gaussian_features


def main(
    data_dir: str = "./data/garden",  # colmap path
    checkpoint: str = "./data/garden/ckpts/ckpt_29999_rank0.pt",  # checkpoint path, can generate from original 3DGS repo
    results_dir: str = "./results/garden",  # output path
    rasterizer: Literal[
        "inria", "gsplat"
    ] = "gsplat",  # Original or GSplat for checkpoints
    data_factor: int = 4,
):

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")

    torch.set_default_device("cuda")

    os.makedirs(results_dir, exist_ok=True)
    splats = load_checkpoint(
        checkpoint, data_dir, rasterizer=rasterizer, data_factor=data_factor
    )
    splats_optimized = prune_by_gradients(splats)
    test_proper_pruning(splats, splats_optimized)
    splats = splats_optimized
    features = create_feature_field_lseg(splats)
    torch.save(features, f"{results_dir}/features_lseg.pt")


if __name__ == "__main__":
    tyro.cli(main)
