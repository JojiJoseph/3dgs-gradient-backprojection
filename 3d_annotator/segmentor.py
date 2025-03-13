import sys
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QAction,
    QMenuBar,
    QMessageBox,
    QActionGroup,
    QMenu,
    QScrollArea,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QDockWidget,
    QTextEdit,
    QWidget,
    QVBoxLayout,
    QToolBox,
    QSlider,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QGraphicsEllipseItem,

)
from PyQt5.QtGui import QColor, QBrush, QPen, QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QPointF

import PyQt5.QtCore as QtCore
from PyQt5.QtGui import QImage, QPixmap, QPainter
from scipy.spatial.transform import Rotation as R
from lseg import LSegNet
import clip

from slider import SliderWithInput

sys.path.append("..")
from utils import (
    load_checkpoint,
    _detach_tensors_from_dict,
    get_viewmat_from_colmap_image,
    torch_to_cv,
)
import pycolmap_scene_manager as pycolmap
import torch
from gsplat import rasterization
import numpy as np

torch.set_default_device("cuda")

class Segmentor:
    def __init__(self, splats):
        self.splats = splats
        self.colmap_project = splats["colmap_project"]

        self.means = splats["means"]
        self.colors_dc = splats["features_dc"]
        self.colors_rest = splats["features_rest"]
        self.colors = torch.cat([self.colors_dc, self.colors_rest], dim=1)
        self.opacities = torch.sigmoid(self.splats["opacity"])
        self.scales = torch.exp(self.splats["scaling"])
        self.quats = self.splats["rotation"]

        self.features = None

        net = LSegNet(
            backbone="clip_vitl16_384",
            features=256,
            crop_size=480,
            arch_option=0,
            block_depth=0,
            activation="lrelu",
        )
        # Load pre-trained weights
        net.load_state_dict(torch.load("../checkpoints/lseg_minimal_e200.ckpt"))
        net.eval()
        net.cuda()

        # Preprocess the text prompt
        clip_text_encoder = net.clip_pretrained.encode_text

        other_prompt = clip.tokenize(["other"])
        other_prompt = other_prompt.cuda()
        other_prompt = clip_text_encoder(other_prompt)  # N, 512, N - number of prompts
        other_prompt = torch.nn.functional.normalize(other_prompt, dim=1).float()

        self.other_prompt = other_prompt

    def load_features(self, features_path):
        self.features = torch.load(features_path)

    def prune_by_gradients(self):
        colors = self.colors_dc[:,0,:].clone()
        colors.requires_grad = True
        contributions = torch.zeros(self.colors_dc.shape[0], device="cuda")
        for image in self.colmap_project.images.values():
            viewmat = get_viewmat_from_colmap_image(image)
            output, _, _ = rasterization(
                self.means,
                self.quats,
                self.scales,
                self.opacities,
                colors[None],
                viewmat[None],
                self.splats["camera_matrix"][None],
                width=self.splats["camera_matrix"][0, 2] * 2,
                height=self.splats["camera_matrix"][1, 2] * 2,
                # sh_degree=3,
            )

            target = output.sum()
            target.backward()
            contributions += colors.grad.norm(dim=1)
            colors.grad.zero_()
        mask_3d = contributions > 0
        self.means = self.means[mask_3d]
        self.colors_dc = self.colors_dc[mask_3d]
        self.colors_rest = self.colors_rest[mask_3d]
        self.colors = self.colors[mask_3d]
        self.opacities = self.opacities[mask_3d]
        self.scales = self.scales[mask_3d]
        self.quats = self.quats[mask_3d]
        return mask_3d

    def render_2d_mask(viewmat):
        pass

    def render_3d_mask(self, viewmat, mask_3d):
        opacities = self.opacities.clone()
        opacities[~mask_3d] = 0
        output, _, _ = rasterization(
                self.means,
                self.quats,
                self.scales,
                opacities,
                self.colors,
                viewmat[None],
                self.splats["camera_matrix"][None],
                width=self.splats["camera_matrix"][0, 2] * 2,
                height=self.splats["camera_matrix"][1, 2] * 2,
                sh_degree=3,
            )
        output_cv = torch_to_cv(output[0])
        return np.ascontiguousarray(output_cv[..., ::-1])

    def render_extraction(viewmat):
        pass

    def render_deletion(viewmat):
        pass

    def render_features(self, viewmat):
        output, _, _ = rasterization(
            self.means,
            self.quats,
            self.scales,
            self.opacities,
            self.features,
            viewmat[None],
            self.splats["camera_matrix"][None],
            width=self.splats["camera_matrix"][0, 2] * 2,
            height=self.splats["camera_matrix"][1, 2] * 2,
            # sh_degree=3,
        )
        return output

    def render(self, viewmat, return_depth=False):
        if not return_depth:
            output, _, _ = rasterization(
                self.means,
                self.quats,
                self.scales,
                self.opacities,
                self.colors,
                viewmat[None],
                self.splats["camera_matrix"][None],
                width=self.splats["camera_matrix"][0, 2] * 2,
                height=self.splats["camera_matrix"][1, 2] * 2,
                sh_degree=3,
            )
            output_cv = torch_to_cv(output[0])
            return np.ascontiguousarray(output_cv[..., ::-1])
        else:
            output, _, depth = rasterization(
                self.means,
                self.quats,
                self.scales,
                self.opacities,
                self.colors,
                viewmat[None],
                self.splats["camera_matrix"][None],
                width=self.splats["camera_matrix"][0, 2] * 2,
                height=self.splats["camera_matrix"][1, 2] * 2,
                sh_degree=3,
                render_mode="RGB+ED",
            )
            output_cv = torch_to_cv(output[0,...,:3])
            output_depth = output[0,...,3].detach().cpu().numpy()
            output_cv = np.ascontiguousarray(output_cv[..., ::-1])
            output_depth = np.ascontiguousarray(output_depth)
            return output_cv, output_depth
        
    def render_with_mask_3d(self, viewmat, mask_3d=None,return_depth=False):
        colors = self.colors.clone()
        
        if mask_3d is not None:
            import math
            colors[mask_3d,0,0] = (1-0.5) / (1/math.sqrt(4*math.pi))
        if not return_depth:
            output, _, _ = rasterization(
                self.means,
                self.quats,
                self.scales,
                self.opacities,
                colors,
                viewmat[None],
                self.splats["camera_matrix"][None],
                width=self.splats["camera_matrix"][0, 2] * 2,
                height=self.splats["camera_matrix"][1, 2] * 2,
                sh_degree=3,
            )
            output_cv = torch_to_cv(output[0])
            return np.ascontiguousarray(output_cv[..., ::-1])
        else:
            output, _, depth = rasterization(
                self.means,
                self.quats,
                self.scales,
                self.opacities,
                self.colors,
                viewmat[None],
                self.splats["camera_matrix"][None],
                width=self.splats["camera_matrix"][0, 2] * 2,
                height=self.splats["camera_matrix"][1, 2] * 2,
                sh_degree=3,
                render_mode="RGB+ED",
            )
            output_cv = torch_to_cv(output[0,...,:3])
            output_depth = output[0,...,3].detach().cpu().numpy()
            output_cv = np.ascontiguousarray(output_cv[..., ::-1])
            output_depth = np.ascontiguousarray(output_depth)
            return output_cv, output_depth
        
    def get_point_prompt_mask(self, point_features, point_categories):
        point_features = torch.stack(point_features)
        point_categories = torch.tensor(point_categories)
        positive_features = point_features[point_categories == 1]
        negative_features = point_features[point_categories == 0]
        if positive_features.shape[0] == 0:
            return torch.zeros(self.features.shape[0], device="cuda", dtype=torch.bool)
        print(point_features.shape)
        # Normalize the point features
        positive_features = torch.nn.functional.normalize(positive_features, dim=1).float()
        negative_features = torch.nn.functional.normalize(negative_features, dim=1).float()
        negative_features = torch.cat([negative_features, self.other_prompt], dim=0)
        # Normalize self.features
        self.features = torch.nn.functional.normalize(self.features, dim=1).float()
        positve_scores = self.features @ positive_features.T
        negative_scores = self.features @ negative_features.T
        mask_3d = torch.max(positve_scores, dim=1).values > torch.max(negative_scores, dim=1).values
        return mask_3d