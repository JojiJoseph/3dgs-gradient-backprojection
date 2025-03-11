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

    def load_features(self, features_path):
        pass

    def prune_by_gradients(self):
        pass

    def render_2d_mask(viewmat):
        pass

    def render_3d_mask(viewmat):
        pass

    def render_extraction(viewmat):
        pass

    def render_deletion(viewmat):
        pass

    def render_features(viewmat):
        pass

    def render(self, viewmat):
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