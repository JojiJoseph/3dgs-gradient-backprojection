# Basic OpenCV viewer with sliders for rotation and translation.
# Can be easily customizable to different use cases.
from typing import Literal
import torch
from gsplat import rasterization
import cv2
from scipy.spatial.transform import Rotation as scipyR
import pycolmap_scene_manager as pycolmap

import numpy as np
import json
import tyro

from utils import get_rpy_matrix

device = torch.device("cuda:0")


def _detach_tensors_from_dict(d, inplace=True):
    if not inplace:
        d = d.copy()
    for key in d:
        if isinstance(d[key], torch.Tensor):
            d[key] = d[key].detach()
    return d


# def load_gaussian_splats_from_input_file(input_path: str):
#     with open(input_path, "r") as f:
#         metadata = json.load(f)
#     checkpoint_path = metadata["checkpoint"]
#     model_params, _ = torch.load(checkpoint_path)

#     splats = {
#         "active_sh_degree": model_params[0],
#         "xyz": model_params[1],
#         "features_dc": model_params[2],
#         "features_rest": model_params[3],
#         "scaling": model_params[4],
#         "rotation": model_params[5],
#         "opacity": model_params[6].squeeze(1),
#     }

#     _detach_tensors_from_dict(splats)

#     return splats, metadata

def load_checkpoint(checkpoint: str, data_dir: str, rasterizer: Literal["inria", "gsplat"]="inria", data_factor: int = 1):

    colmap_project = pycolmap.SceneManager(f"{data_dir}/sparse/0")
    colmap_project.load_cameras()
    colmap_project.load_images()
    colmap_project.load_points3D()
    model = torch.load(checkpoint) # Make sure it is generated by 3DGS original repo
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

    camera_matrix[:2,:3] /= data_factor

    splats["camera_matrix"] = camera_matrix
    splats["colmap_project"] = colmap_project
    splats["colmap_dir"] = data_dir

    return splats



def main(checkpoint: str, data_dir: str, rasterizer: Literal["inria", "gsplat"]="gsplat", data_factor: int = 4):
    splats = load_checkpoint(checkpoint, data_dir, rasterizer, data_factor)
    K = splats["camera_matrix"].cuda()

    show_anaglyph = False
    width = int(K[0, 2] * 2)
    height = int(K[1, 2] * 2)

    means = splats["means"].float()
    opacities = splats["opacity"]
    quats = splats["rotation"]
    scales = splats["scaling"].float()

    opacities = torch.sigmoid(opacities)
    scales = torch.exp(scales)
    colors = torch.cat([splats["features_dc"], splats["features_rest"]], 1)

    cv2.namedWindow("GSplat Explorer", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Roll", "GSplat Explorer", 0, 180, lambda x: None)
    cv2.createTrackbar("Pitch", "GSplat Explorer", 0, 180, lambda x: None)
    cv2.createTrackbar("Yaw", "GSplat Explorer", 0, 180, lambda x: None)
    cv2.createTrackbar("X", "GSplat Explorer", 0, 1000, lambda x: None)
    cv2.createTrackbar("Y", "GSplat Explorer", 0, 1000, lambda x: None)
    cv2.createTrackbar("Z", "GSplat Explorer", 0, 1000, lambda x: None)
    cv2.createTrackbar("Scaling", "GSplat Explorer", 100, 100, lambda x: None)

    cv2.setTrackbarMin("Roll", "GSplat Explorer", -180)
    cv2.setTrackbarMax("Roll", "GSplat Explorer", 180)
    cv2.setTrackbarMin("Pitch", "GSplat Explorer", -180)
    cv2.setTrackbarMax("Pitch", "GSplat Explorer", 180)
    cv2.setTrackbarMin("Yaw", "GSplat Explorer", -180)
    cv2.setTrackbarMax("Yaw", "GSplat Explorer", 180)
    cv2.setTrackbarMin("X", "GSplat Explorer", -1000)
    cv2.setTrackbarMax("X", "GSplat Explorer", 1000)
    cv2.setTrackbarMin("Y", "GSplat Explorer", -1000)
    cv2.setTrackbarMax("Y", "GSplat Explorer", 1000)
    cv2.setTrackbarMin("Z", "GSplat Explorer", -1000)
    cv2.setTrackbarMax("Z", "GSplat Explorer", 1000)


    def update_trackbars_from_viewmat(world_to_camera):
        # if torch tensor is passed, convert to numpy
        if isinstance(world_to_camera, torch.Tensor):
            world_to_camera = world_to_camera.cpu().numpy()
        r = scipyR.from_matrix(world_to_camera[:3,:3])
        roll, pitch, yaw = r.as_euler('xyz')
        cv2.setTrackbarPos("Roll", "GSplat Explorer", np.rad2deg(roll).astype(int))
        cv2.setTrackbarPos("Pitch", "GSplat Explorer", np.rad2deg(pitch).astype(int))
        cv2.setTrackbarPos("Yaw", "GSplat Explorer", np.rad2deg(yaw).astype(int))
        cv2.setTrackbarPos("X", "GSplat Explorer", int(world_to_camera[0, 3]*100))
        cv2.setTrackbarPos("Y", "GSplat Explorer", int(world_to_camera[1, 3]*100))
        cv2.setTrackbarPos("Z", "GSplat Explorer", int(world_to_camera[2, 3]*100))

    while True:
        roll = cv2.getTrackbarPos("Roll", "GSplat Explorer")
        pitch = cv2.getTrackbarPos("Pitch", "GSplat Explorer")
        yaw = cv2.getTrackbarPos("Yaw", "GSplat Explorer")

        roll_rad = np.deg2rad(roll)
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)

        scaling = cv2.getTrackbarPos("Scaling", "GSplat Explorer") / 100.0

        viewmat = (
            torch.tensor(get_rpy_matrix(roll_rad, pitch_rad, yaw_rad))
            .float()
            .to(device)
        )
        
        viewmat[0, 3] = cv2.getTrackbarPos("X", "GSplat Explorer") / 100.0
        viewmat[1, 3] = cv2.getTrackbarPos("Y", "GSplat Explorer") / 100.0
        viewmat[2, 3] = cv2.getTrackbarPos("Z", "GSplat Explorer") / 100.0
        output, _, meta = rasterization(
            means,
            quats,
            scales * scaling,
            opacities,
            colors,
            viewmat[None],
            K[None],
            width=width,
            height=height,
            sh_degree=3,
        )

        output_cv = torch_to_cv(output[0])

        if show_anaglyph:
            viewmat[0, 3] = viewmat[0, 3]-0.05
            output_left = output_cv
            output, _, meta = rasterization(
                means,
                quats,
                scales * scaling,
                opacities,
                colors,
                viewmat[None],
                K[None],
                width=width,
                height=height,
                sh_degree=3,
            )
            output_right = torch_to_cv(output[0])
            output_left[...,:2] = 0
            output_right[...,-1] = 0
            output_cv = output_left + output_right

        cv2.imshow("GSplat Explorer", output_cv)
        full_key = cv2.waitKeyEx(1)
        key = full_key & 0xFF
        if key == ord("q"):
            break
        if key == ord("3"):
            show_anaglyph = not show_anaglyph
        if key in [ord("w"), ord("a"), ord("s"), ord("d")]:
            if key == ord("w"):
                viewmat[2, 3] -= 0.1
            if key == ord("s"):
                viewmat[2, 3] += 0.1
            if key == ord("a"):
                viewmat[0, 3] += 0.1
            if key == ord("d"):
                viewmat[0, 3] -= 0.1
            update_trackbars_from_viewmat(viewmat)
            
            


def torch_to_cv(tensor, permute=False):
    if permute:
        tensor = torch.clamp(tensor.permute(1, 2, 0), 0, 1).cpu().numpy()
    else:
        tensor = torch.clamp(tensor, 0, 1).cpu().numpy()
    return (tensor * 255).astype(np.uint8)[..., ::-1]


if __name__ == "__main__":
    tyro.cli(main)