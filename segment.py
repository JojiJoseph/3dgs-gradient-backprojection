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
matplotlib.use("TkAgg")

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
    if rasterizer == "original":
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


def create_checkerboard(width, height, size=64):
    checkerboard = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, size):
        for x in range(0, width, size):
            if (x // size + y // size) % 2 == 0:
                checkerboard[y : y + size, x : x + size] = 255
            else:
                checkerboard[y : y + size, x : x + size] = 128
    return checkerboard


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

    prompts = [prompt] + neg_prompt.split(";")

    prompt = clip.tokenize(prompts)
    prompt = prompt.cuda()

    text_feat = clip_text_encoder(prompt)  # N, 512, N - number of prompts
    text_feat_norm = torch.nn.functional.normalize(text_feat, dim=1)

    features = torch.nn.functional.normalize(features, dim=1)
    score = features @ text_feat_norm.float().T
    mask_3d = score[:, 0] > score[:, 1:].max(dim=1)[0]
    if threshold is not None:
        mask_3d = mask_3d & (score[:, 0] > threshold)
    mask_3d_inv = ~mask_3d

    return mask_3d, mask_3d_inv


def apply_mask3d(splats, mask3d, mask3d_inverted):
    if mask3d_inverted == None:
        mask3d_inverted = ~mask3d
    extracted = splats.copy()
    deleted = splats.copy()
    masked = splats.copy()
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
    imageio.mimsave(output_path, frames, fps=10, loop=0)
    if feedback:
        cv2.destroyAllWindows()


def main(
    data_dir: str = "./data/garden",  # colmap path
    checkpoint: str = "./data/garden/ckpts/ckpt_29999_rank0.pt",  # checkpoint path, can generate from original 3DGS repo
    results_dir: str = "./results/garden",  # output path
    rasterizer: Literal[
        "inria", "gsplat"
    ] = "gsplat",  # Original or gsplat for checkpoints
    prompt: str = "Table",
    neg_prompt: str = "Vase;Other",
    data_factor: int = 4,
    show_visual_feedback: bool = True,
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
    features = torch.load(f"{results_dir}/features_lseg.pt")
    mask3d, mask3d_inv = get_mask3d_lseg(splats, features, prompt, neg_prompt)
    extracted, deleted, masked = apply_mask3d(splats, mask3d, mask3d_inv)

    render_to_gif(
        f"{results_dir}/extracted.gif",
        extracted,
        show_visual_feedback,
        use_checkerboard_background=True,
    )
    render_to_gif(f"{results_dir}/deleted.gif", deleted, show_visual_feedback)


if __name__ == "__main__":
    tyro.cli(main)
