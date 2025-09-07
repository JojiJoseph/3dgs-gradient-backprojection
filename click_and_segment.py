# Basic OpenCV viewer with sliders for rotation and translation.
# Can be easily customizable to different use cases.
import torch
from gsplat import rasterization
import cv2
import tyro
import numpy as np
import json
from typing import Literal
import pycolmap_scene_manager as pycolmap
import clip
from lseg import LSegNet
from utils import (
    load_checkpoint,
    get_rpy_matrix,
    get_viewmat_from_colmap_image,
    prune_by_gradients,
    torch_to_cv,
)

device = torch.device("cuda:0")

from scipy.spatial.transform import Rotation as R

def view_matrix_to_rpy_t(view_matrix):
    """
    Convert a 4x4 view matrix to roll, pitch, yaw (RPY) and translation.
    """
    R_mat = view_matrix[:3, :3]  # Extract rotation
    t_vec = view_matrix[:3, 3]   # Extract translation
    rpy = R.from_matrix(R_mat).as_euler('xyz', degrees=True)  # Convert to roll, pitch, yaw
    return rpy, t_vec

def nearest_view_matrix(A, view_matrices, weight_translation=1.0):
    """
    Find the index of the nearest view matrix to A.
    
    Args:
        A (torch.Tensor): Target 4x4 view matrix.
        view_matrices (torch.Tensor): (N, 4, 4) Tensor of view matrices.
        weight_translation (float): Weighting factor for translation distance.
    
    Returns:
        int: Index of the nearest view matrix.
    """
    # Extract rotation and translation
    R_A = A[:3, :3]
    t_A = A[:3, 3]
    
    R_matrices = view_matrices[:, :3, :3]  # (N, 3, 3)
    t_matrices = view_matrices[:, :3, 3]   # (N, 3)

    # Compute rotation difference: d_R(A, V_i) = cos⁻¹((Tr(R_A R_i^T) - 1) / 2)
    R_diff = torch.matmul(R_A, R_matrices.transpose(1, 2))  # (N, 3, 3)
    trace_values = torch.einsum('nii->n', R_diff)  # Extract trace for each matrix
    rotation_distances = torch.acos(torch.clamp((trace_values - 1) / 2, -1, 1))  # Clamp for stability

    # Compute translation difference: ||t_A - t_i||
    translation_distances = torch.norm(t_A - t_matrices, dim=1)

    # Compute weighted sum of distances
    total_distances = rotation_distances + weight_translation * translation_distances

    # Return the index of the minimum distance
    return torch.argmin(total_distances).item()



def calculate_3d_to_2d(viewmat, fx, fy, cx, cy, position_homo):
    x, y, z, _ = position_homo
    x = x.item()
    y = y.item()
    z = z.item()
    x1 = viewmat[0, 0] * x + viewmat[0, 1] * y + viewmat[0, 2] * z + viewmat[0, 3]
    y1 = viewmat[1, 0] * x + viewmat[1, 1] * y + viewmat[1, 2] * z + viewmat[1, 3]
    z1 = viewmat[2, 0] * x + viewmat[2, 1] * y + viewmat[2, 2] * z + viewmat[2, 3]
    x = x1 * fx + cx * z1
    y = y1 * fy + cy * z1
    return int(x / z1), int(y / z1)


class UIManager:
    def __init__(self, window_name: str):
        """
        Manages OpenCV UI components like sliders and mouse callbacks.

        Args:
            window_name (str): Name of the OpenCV window.
        """
        self.window_name = window_name
        self.params = {
            "Roll": 0,
            "Pitch": 0,
            "Yaw": 0,
            "X": 0,
            "Y": 0,
            "Z": 0,
            "Scaling": 100,
        }
        # self.positive_prompt_locations = []
        # self.negative_prompt_locations = []
        self._trigger_callback = lambda: None
        self._setup_ui()

    def _setup_ui(self):
        """
        Sets up sliders and mouse callbacks for the OpenCV window.
        """
        sliders = [
            ("Roll", -180, 0, 180),
            ("Pitch", -180, 0, 180),
            ("Yaw", -180, 0, 180),
            ("X", -1000, 0, 1000),
            ("Y", -1000, 0, 1000),
            ("Z", -1000, 0, 1000),
            ("Scaling", 0, 100, 200),
        ]
        for slider_name, min_val, default_val, max_val in sliders:
            cv2.createTrackbar(
                slider_name,
                self.window_name,
                default_val,
                max_val,
                self._on_slider_change,
            )
            cv2.setTrackbarMin(slider_name, self.window_name, min_val)

        cv2.setMouseCallback(self.window_name, self._on_mouse_event)

    def _on_slider_change(self, value):
        """
        Callback for slider changes.
        """
        for param in self.params:
            self.params[param] = cv2.getTrackbarPos(param, self.window_name)

    def _on_mouse_event(self, event, x, y, flags, param):
        """
        Callback for mouse events.

        Args:
            event (int): OpenCV mouse event type.
            x (int): X-coordinate of the mouse event.
            y (int): Y-coordinate of the mouse event.
            flags (int): Event flags.
            param (Any): Additional parameters.
        """
        ctrl_pressed = flags & cv2.EVENT_FLAG_CTRLKEY
        trigger = False
        xy = None
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = x, y
            trigger = True
        elif event == cv2.EVENT_MBUTTONDOWN:
            xy = x, y
            trigger = True
        if trigger:
            self._trigger_callback(xy, event, ctrl_pressed)

    def _remove_prompt(self, locations, x, y):
        """
        Removes a prompt close to the specified location.

        Args:
            locations (list): List of existing prompt locations.
            x (int): X-coordinate.
            y (int): Y-coordinate.
        """
        del_idx = None
        for i, (x_i, y_i) in enumerate(locations):
            if abs(x_i - x) < 40 and abs(y_i - y) < 40:
                del_idx = i
                break
        if del_idx is not None:
            del locations[del_idx]

    def get_params(self):
        """
        Returns the current slider values.

        Returns:
            dict: Dictionary of slider names and their values.
        """
        return self.params

    def set_trigger_callback(self, callback):
        """
        Sets the trigger callback function.

        Args:
            callback (function): The callback function.
        """
        self._trigger_callback = callback


def main(
    data_dir: str = "./data/garden",  # colmap path
    checkpoint: str = "./data/garden/ckpts/ckpt_29999_rank0.pt",  # checkpoint path, can generate from original 3DGS repo
    rasterizer: Literal[
        "inria", "gsplat"
    ] = "gsplat",  # Original or GSplat for checkpoints
    results_dir: str = "./results/garden", # Not used as of now
    data_factor: int = 4,
    feature_path: str = "./results/garden/features_lseg_garden.pt",
    feature: str = "lseg"
):

    torch.set_default_device("cuda")

    splats = load_checkpoint(
        checkpoint, data_dir, rasterizer=rasterizer, data_factor=data_factor
    )
    splats = prune_by_gradients(splats)
    torch.set_grad_enabled(False)

    means = splats["means"].float()
    opacities = splats["opacity"]
    quats = splats["rotation"]
    scales = splats["scaling"].float()

    opacities = torch.sigmoid(opacities)
    scales = torch.exp(scales)
    colors = torch.cat([splats["features_dc"], splats["features_rest"]], 1)

    features = torch.load(feature_path)
    dim = features.shape[-1]

    K = splats["camera_matrix"].float()

    width = int(K[0, 2] * 2)
    height = int(K[1, 2] * 2)

    


    # Load viewmats from colmap

    viewmats = []
    for image in splats["colmap_project"].images.values():
        viewmats.append(get_viewmat_from_colmap_image(image))

    viewmats = torch.stack(viewmats).to(device)

    cv2.namedWindow("Click and Segment", cv2.WINDOW_NORMAL)
    ui_manager = UIManager("Click and Segment")

    if feature == "lseg":

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

        other_prompt = clip.tokenize(["other"])
        other_prompt = other_prompt.cuda()
        other_prompt = clip_text_encoder(other_prompt)  # N, 512, N - number of prompts
        other_prompt = torch.nn.functional.normalize(other_prompt, dim=1).float()
    else:
        other_prompt = None

    mask_3d = None

    positions_3d_positives = []
    positions_3d_negatives = []

    positive_prompts = torch.zeros(0, dim).to(device)
    if other_prompt is not None:
        negative_prompts = other_prompt.to(device)
    else:
        negative_prompts = torch.zeros(0, dim).to(device)

    def trigger_callback(xy, event, ctrl_pressed):
        if xy[0] >= width or xy[1] >= height:
            return
        params = ui_manager.get_params()

        nonlocal positive_prompts
        nonlocal negative_prompts

        roll = params["Roll"]
        pitch = params["Pitch"]
        yaw = params["Yaw"]

        roll_rad = np.deg2rad(roll)
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)

        viewmat = (
            torch.tensor(get_rpy_matrix(roll_rad, pitch_rad, yaw_rad))
            .float()
            .to(device)
        )
        viewmat[0, 3] = params["X"] / 100.0
        viewmat[1, 3] = params["Y"] / 100.0
        viewmat[2, 3] = params["Z"] / 100.0
        scaling = params["Scaling"] / 100.0
        output, alphas, meta = rasterization(
            means,
            quats,
            scales * scaling,
            opacities,
            features,
            viewmat[None],
            K[None],
            width=width,
            height=height,
            render_mode="RGB+ED",
        )

        dim = output.shape[-1]-1

        output, depth = output[0, ..., :dim], output[0, ..., dim]

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        if not ctrl_pressed:
            if xy is not None:
                Z = depth[xy[1], xy[0]]
                XY = (
                    torch.tensor([(xy[0] - cx) / fx * Z, (xy[1] - cy) / fy * Z, Z, 1.0])
                    .float()
                    .to(device)
                )
                XY = XY.reshape(4, 1)
                XY_world = torch.inverse(viewmat) @ XY
                if event == cv2.EVENT_LBUTTONDOWN:
                    positions_3d_positives.append(XY_world.cpu().numpy())
                elif event == cv2.EVENT_MBUTTONDOWN:
                    positions_3d_negatives.append(XY_world.cpu().numpy())
        output = torch.nn.functional.normalize(output, dim=-1)

        positive_2d_position = []
        negative_2d_position = []

        for pos in positions_3d_positives:
            x, y = calculate_3d_to_2d(viewmat, fx, fy, cx, cy, pos)
            positive_2d_position.append((x, y))

        for pos in positions_3d_negatives:
            x, y = calculate_3d_to_2d(viewmat, fx, fy, cx, cy, pos)
            negative_2d_position.append((x, y))

        if not ctrl_pressed and event == cv2.EVENT_LBUTTONDOWN:
            positive_prompts = torch.cat([positive_prompts, output[xy[1], xy[0]][None]])
        if not ctrl_pressed and event == cv2.EVENT_MBUTTONDOWN:
            negative_prompts = torch.cat([negative_prompts, output[xy[1], xy[0]][None]])
        if ctrl_pressed and event == cv2.EVENT_LBUTTONDOWN:
            del_idx = None
            for i, (x_i, y_i) in enumerate(positive_2d_position):
                if abs(x_i - xy[0]) < 40 and abs(y_i - xy[1]) < 40:
                    del_idx = i
                    break
            if del_idx is not None:
                positive_prompts = torch.cat(
                    [positive_prompts[:del_idx], positive_prompts[del_idx + 1 :]]
                )
                del positions_3d_positives[del_idx]
        if ctrl_pressed and event == cv2.EVENT_MBUTTONDOWN:
            del_idx = None
            for i, (x_i, y_i) in enumerate(negative_2d_position):
                if abs(x_i - xy[0]) < 40 and abs(y_i - xy[1]) < 40:
                    del_idx = i
                    break
            if del_idx is not None:
                negative_prompts = torch.cat(
                    [negative_prompts[: del_idx + 1], negative_prompts[del_idx + 2 :]]
                )
                del positions_3d_negatives[del_idx]
        nonlocal mask_3d
        if not positions_3d_positives:
            mask_3d = None
        elif len(negative_prompts) > 0:
            scores_pos = features @ positive_prompts.T  # [N, P]
            scores_pos = scores_pos.max(dim=1)  # [N]
            scores_neg = features @ negative_prompts.T  # [N, P]
            scores_neg = scores_neg.max(dim=1)  # [N]
            mask_3d = scores_pos.values > scores_neg.values

    ui_manager.set_trigger_callback(trigger_callback)

    while True:
        for image in splats["colmap_project"].images.values():
            viewmat_cmap = get_viewmat_from_colmap_image(image)
            roll = ui_manager.params["Roll"]
            pitch = ui_manager.params["Pitch"]
            yaw = ui_manager.params["Yaw"]

            roll_rad = np.deg2rad(roll)
            pitch_rad = np.deg2rad(pitch)
            yaw_rad = np.deg2rad(yaw)

            viewmat = (
                torch.tensor(get_rpy_matrix(roll_rad, pitch_rad, yaw_rad))
                .float()
                .to(device)
            )
            viewmat[0, 3] = ui_manager.params["X"] / 100.0
            viewmat[1, 3] = ui_manager.params["Y"] / 100.0
            viewmat[2, 3] = ui_manager.params["Z"] / 100.0
            scaling = ui_manager.params["Scaling"] / 100.0
            output, alphas, meta = rasterization(
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

            if mask_3d is not None:
                opacities_extracted = opacities.clone()
                opacities_deleted = opacities.clone()
                opacities_extracted[~mask_3d] = 0
                opacities_deleted[mask_3d] = 0
            else:
                opacities_extracted = opacities
                opacities_deleted = opacities * 0
            output, alphas, meta = rasterization(
                means,
                quats,
                scales * scaling,
                opacities_extracted,
                colors,
                viewmat_cmap[None],
                K[None],
                width=width,
                height=height,
                sh_degree=3,
            )
            output_cv2 = torch_to_cv(output[0])
            output, alphas, meta = rasterization(
                means,
                quats,
                scales * scaling,
                opacities_deleted,
                colors,
                viewmat_cmap[None],
                K[None],
                width=width,
                height=height,
                sh_degree=3,
            )
            output_cv3 = torch_to_cv(output[0])
            output_cv = cv2.hconcat([output_cv, output_cv2, output_cv3])

            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            viewmat_np = viewmat.cpu().numpy()
            for pos in positions_3d_positives:
                x, y = calculate_3d_to_2d(viewmat_np, fx, fy, cx, cy, pos)
                cv2.circle(output_cv, (x, y), 10, (0, 255, 0), -1)
            for pos in positions_3d_negatives:
                x, y = calculate_3d_to_2d(viewmat_np, fx, fy, cx, cy, pos)
                cv2.circle(output_cv, (x, y), 10, (0, 0, 255), -1)
            cv2.imshow("Click and Segment", output_cv)
            key = cv2.waitKey(10)
            if key == ord("q"):
                break
            if key == ord("n"):
                print("Next viewmat")
                # viewmat = get_nearest_viewmat(viewmats, ui_manager)
                nearest_idx = nearest_view_matrix(viewmat, viewmats, weight_translation=1.0)
                viewmat = viewmats[nearest_idx]
                roll, pitch, yaw = view_matrix_to_rpy_t(viewmat.cpu().numpy())[0]
                cv2.setTrackbarPos("Roll", "Click and Segment", int(roll))
                cv2.setTrackbarPos("Pitch", "Click and Segment", int(pitch))
                cv2.setTrackbarPos("Yaw", "Click and Segment", int(yaw))
                cv2.setTrackbarPos("X", "Click and Segment", int(viewmat[0, 3] * 100))
                cv2.setTrackbarPos("Y", "Click and Segment", int(viewmat[1, 3] * 100))
                cv2.setTrackbarPos("Z", "Click and Segment", int(viewmat[2, 3] * 100))
        if key == ord("q"):
            break
            



if __name__ == "__main__":
    tyro.cli(main)
