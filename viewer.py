# Basic OpenCV viewer with sliders for rotation and translation.
# Can be easily customizable to different use cases.
from dataclasses import dataclass
from typing import Literal, Optional
import torch
from gsplat import rasterization
import cv2
from scipy.spatial.transform import Rotation as scipyR
import pycolmap_scene_manager as pycolmap
import warnings

import numpy as np
import json
import tyro

from utils import (
    get_rpy_matrix,
    prune_by_gradients,
    torch_to_cv,
    load_checkpoint,
)

# Check if CUDA is available. Else raise an error.
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. Please install the correct version of PyTorch with CUDA support."
    )

device = torch.device("cuda")
torch.set_default_device("cuda")


@dataclass
class Args:
    checkpoint: str  # Path to the 3DGS checkpoint file (.pth/.pt) to be visualized.
    data_dir: str  # Path to the COLMAP project directory containing sparse reconstruction.
    format: Optional[Literal["inria", "gsplat"]] = "gsplat"  # Format of the checkpoint: 'inria' (original 3DGS) or 'gsplat'.
    rasterizer: Optional[Literal["inria", "gsplat"]] = None  # [Deprecated] Use --format instead. Provided for backward compatibility.
    data_factor: int = 4  # Downscaling factor for the renderings.



class Viewer:
    def __init__(self, splats):
        self.splats = None
        self.camera_matrix = None
        self.width = None
        self.height = None
        self.viewmat = None
        self.window_name = "GSplat Explorer"
        self._init_sliders()
        self._load_splats(splats)

    def _load_splats(self, splats):
        K = splats["camera_matrix"].cuda()
        width = int(K[0, 2] * 2)
        height = int(K[1, 2] * 2)

        means = splats["means"].float()
        opacities = splats["opacity"]
        quats = splats["rotation"]
        scales = splats["scaling"].float()

        opacities = torch.sigmoid(opacities)
        scales = torch.exp(scales)
        colors = torch.cat([splats["features_dc"], splats["features_rest"]], 1)

        self.splats = splats
        self.camera_matrix = K
        self.width = width
        self.height = height
        self.means = means
        self.opacities = opacities
        self.quats = quats
        self.scales = scales
        self.colors = colors

    def _init_sliders(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        trackbars = {
            "Roll": (-180, 180),
            "Pitch": (-180, 180),
            "Yaw": (-180, 180),
            "X": (-1000, 1000),
            "Y": (-1000, 1000),
            "Z": (-1000, 1000),
            "Scaling": (0, 100),
        }

        for name, (min_val, max_val) in trackbars.items():
            cv2.createTrackbar(name, self.window_name, 0, max_val, lambda x: None)
            cv2.setTrackbarMin(name, self.window_name, min_val)
            cv2.setTrackbarMax(name, self.window_name, max_val)

        cv2.setTrackbarPos(
            "Scaling", self.window_name, 100
        )  # Default value for scaling is 100

    def update_trackbars_from_viewmat(self, world_to_camera):
        # if torch tensor is passed, convert to numpy
        if isinstance(world_to_camera, torch.Tensor):
            world_to_camera = world_to_camera.cpu().numpy()
        r = scipyR.from_matrix(world_to_camera[:3, :3])
        roll, pitch, yaw = r.as_euler("xyz")
        cv2.setTrackbarPos("Roll", self.window_name, np.rad2deg(roll).astype(int))
        cv2.setTrackbarPos("Pitch", self.window_name, np.rad2deg(pitch).astype(int))
        cv2.setTrackbarPos("Yaw", self.window_name, np.rad2deg(yaw).astype(int))
        cv2.setTrackbarPos("X", self.window_name, int(world_to_camera[0, 3] * 100))
        cv2.setTrackbarPos("Y", self.window_name, int(world_to_camera[1, 3] * 100))
        cv2.setTrackbarPos("Z", self.window_name, int(world_to_camera[2, 3] * 100))

    def _get_viewmat_from_trackbars(self):
        roll = cv2.getTrackbarPos("Roll", self.window_name)
        pitch = cv2.getTrackbarPos("Pitch", self.window_name)
        yaw = cv2.getTrackbarPos("Yaw", self.window_name)

        roll_rad = np.deg2rad(roll)
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)

        viewmat = (
            torch.tensor(get_rpy_matrix(roll_rad, pitch_rad, yaw_rad))
            .float()
            .to(device)
        )

        viewmat[0, 3] = cv2.getTrackbarPos("X", self.window_name) / 100.0
        viewmat[1, 3] = cv2.getTrackbarPos("Y", self.window_name) / 100.0
        viewmat[2, 3] = cv2.getTrackbarPos("Z", self.window_name) / 100.0

        return viewmat

    def render_gaussians(self, viewmat, scaling):
        output, _, _ = rasterization(
            self.means,
            self.quats,
            self.scales * scaling,
            self.opacities,
            self.colors,
            viewmat[None],
            self.camera_matrix[None],
            width=self.width,
            height=self.height,
            sh_degree=3,
        )
        return torch_to_cv(output[0])

    def run(self):
        """Run the interactive Gaussian Splat viewer loop once until exit."""
        show_anaglyph = False

        while True:
            scaling = cv2.getTrackbarPos("Scaling", self.window_name) / 100.0
            viewmat = self._get_viewmat_from_trackbars()

            output_cv = self.render_gaussians(viewmat, scaling)

            if show_anaglyph:
                offset_viewmat = viewmat.clone()
                offset_viewmat[0, 3] -= 0.05
                output_left = output_cv
                output_right = self.render_gaussians(offset_viewmat, scaling)
                output_left[..., :2] = 0
                output_right[..., -1] = 0
                output_cv = output_left + output_right

            cv2.imshow(self.window_name, output_cv)
            full_key = cv2.waitKeyEx(1)
            key = full_key & 0xFF

            if key == ord("q") or key == 27:
                break
            if key == ord("3"):
                show_anaglyph = not show_anaglyph
            if key in [ord("w"), ord("a"), ord("s"), ord("d")]:
                # Modify viewmat and sync UI
                delta = 0.1
                if key == ord("w"):
                    viewmat[2, 3] -= delta
                elif key == ord("s"):
                    viewmat[2, 3] += delta
                elif key == ord("a"):
                    viewmat[0, 3] += delta
                elif key == ord("d"):
                    viewmat[0, 3] -= delta
                self.update_trackbars_from_viewmat(viewmat)

        cv2.destroyAllWindows()


def main(args: Args):
    format = args.format or args.rasterizer
    if args.rasterizer:
        warnings.warn(
            "`rasterizer` is deprecated. Use `format` instead.", DeprecationWarning
        )
    if not format:
        raise ValueError("Must specify --format or the deprecated --rasterizer")

    splats = load_checkpoint(args.checkpoint, args.data_dir, format, args.data_factor)
    splats = prune_by_gradients(splats)

    viewer = Viewer(splats)
    viewer.run()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
