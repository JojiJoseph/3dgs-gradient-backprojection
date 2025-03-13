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
    QToolBar,
    
)
from PyQt5.QtGui import QColor, QBrush, QPen, QPixmap, QImage, QPainter, QIcon
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

from image_viewer import ImageViewer
from tools import CircleTool, RectangleTool, LineTool, PolygonTool, PromptTool
from segmentor import Segmentor
# import cv2
# import matplotlib
# matplotlib.use("TkAgg")

torch.set_default_device("cuda")

splats = load_checkpoint(
    "../data/garden/ckpts/ckpt_29999_rank0.pt", "../data/garden", data_factor=4
)

means = splats["means"]
colors_dc = splats["features_dc"]
colors_rest = splats["features_rest"]
colors = torch.cat([colors_dc, colors_rest], dim=1)
opacities = torch.sigmoid(splats["opacity"])
scales = torch.exp(splats["scaling"])
quats = splats["rotation"]

colmap_project = splats["colmap_project"]
K = splats["camera_matrix"]
width = K[0, 2] * 2
height = K[1, 2] * 2
width = int(width)
height = int(height)


for image in colmap_project.images.values():
    viewmat = get_viewmat_from_colmap_image(image)
    output, _, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmat[None],
        K[None],
        width=width,
        height=height,
        sh_degree=3,
    )

    output_cv = output[0].cpu().numpy()
    output_cv = np.clip(output_cv, 0, 1)
    output_cv = (output_cv * 255).astype(np.uint8)  # [..., ::-1]
    break


segmentor = Segmentor(splats)
mask_3d = segmentor.prune_by_gradients()
print("No of gaussians after pruning: ", mask_3d.sum())
print("% of gaussians kept: ", mask_3d.sum() / mask_3d.shape[0] * 100)
segmentor.load_features("../results/garden/features_lseg.pt")

rendering = segmentor.render(viewmat)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.segmentor = segmentor

    def init_ui(self):
        """Initialize UI elements, including menus."""
        self.setWindowTitle("3D Annotator")
        self.setGeometry(100, 100, 800, 600)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.setCentralWidget(self.scroll_area)
        # self.render_label = QLabel("Rendering")
        # self.addDockWidget(Qt.RightDockWidgetArea, self.render_label)

        # Menubar
        self.create_menus()

        rendering_cv = np.ascontiguousarray(
            rendering
        )  # Make sure the array is contiguous in memory

        self.image_viewer = ImageViewer(rendering_cv)
        self.scroll_area.setWidget(self.image_viewer)

        self.points_3d = []
        self.points_3d_categories = []
        self.points_3d_items = []
        self.points_features = []
        self.mask_3d = None

        self.create_side_bar()
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.context_menu)

        self.create_toolbar()
        self.update_sliders_from_viewmat()

    def create_toolbar(self):
        self.toolbar = QToolBar("Toolbar")
        self.addToolBar(self.toolbar)
        undo_action = QAction(QIcon.fromTheme("edit-undo"), "Undo", self)
        self.toolbar.addAction(undo_action)
        undo_action.triggered.connect(self.image_viewer.undo_stack.undo)

        redo_action = QAction(QIcon.fromTheme("edit-redo"), "Redo", self)
        self.toolbar.addAction(redo_action)
        redo_action.triggered.connect(self.image_viewer.undo_stack.redo)

        zoom_in_action = QAction(QIcon.fromTheme("zoom-in"), "Zoom In", self)
        self.toolbar.addAction(zoom_in_action)
        zoom_in_action.triggered.connect(lambda: self.zoom_image(2))

        zoom_out_action = QAction(QIcon.fromTheme("zoom-out"), "Zoom Out", self)
        self.toolbar.addAction(zoom_out_action)
        zoom_out_action.triggered.connect(lambda: self.zoom_image(0.5))

        reset_zoom_action = QAction(QIcon.fromTheme("zoom-original"), "Reset Zoom", self)
        self.toolbar.addAction(reset_zoom_action)
        reset_zoom_action.triggered.connect(
            lambda: self.image_viewer.resetTransform()
        )

        fit_to_window_action = QAction(QIcon.fromTheme("zoom-fit-best"), "Fit to Window", self)
        self.toolbar.addAction(fit_to_window_action)
        fit_to_window_action.triggered.connect(lambda: self.fit_to_window())

    def update_sliders_from_viewmat(self):
        viewmat_np = viewmat.cpu().numpy()
        roll, pitch, yaw = R.from_matrix(viewmat_np[:3, :3]).as_euler("xyz")
        x, y, z = viewmat_np[:3, 3]
        self.roll_control.slider.setValue(np.rad2deg(roll))
        self.pitch_control.slider.setValue(np.rad2deg(pitch))
        self.yaw_control.slider.setValue(np.rad2deg(yaw))
        self.x_control.slider.setValue(x * 100)
        self.y_control.slider.setValue(y * 100)
        self.z_control.slider.setValue(z * 100)

    def create_side_bar(self):

        dock = QDockWidget("Sidebar", self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

        layout = QVBoxLayout()
        layout.setSizeConstraint(QVBoxLayout.SetFixedSize)

        self.roll_control = SliderWithInput("Roll", -180, 180, 0, self.redraw)
        self.pitch_control = SliderWithInput("Pitch", -180, 180, 0, self.redraw)
        self.yaw_control = SliderWithInput("Yaw", -180, 180, 0, self.redraw)
        self.x_control = SliderWithInput("X", -1000, 1000, 0, self.redraw)
        self.y_control = SliderWithInput("Y", -1000, 1000, 0, self.redraw)
        self.z_control = SliderWithInput("Z", -1000, 1000, 0, self.redraw)
        

        for control in [
            self.roll_control,
            self.pitch_control,
            self.yaw_control,
            self.x_control,
            self.y_control,
            self.z_control,
        ]:
            layout.addWidget(control)

        content = QWidget()
        content.setLayout(layout)
        dock.setWidget(content)


    def _viewmat_from_sliders(self):
        roll = self.roll_control.slider.value()
        roll_rad = np.deg2rad(roll)
        pitch = self.pitch_control.slider.value()
        pitch_rad = np.deg2rad(pitch)
        yaw = self.yaw_control.slider.value()
        yaw_rad = np.deg2rad(yaw)
        x = self.x_control.slider.value() / 100
        y = self.y_control.slider.value() / 100
        z = self.z_control.slider.value() / 100

        R_np = R.from_euler("xyz", [roll_rad, pitch_rad, yaw_rad]).as_matrix()
        viewmat = torch.eye(4)
        viewmat[:3, :3] = torch.tensor(R_np)
        viewmat[:3, 3] = torch.tensor([x, y, z])
        return viewmat
    
    def _pixmap_from_cv(self, np_array):
        height, width, channels = np_array.shape
        bytes_per_line = channels * width
        qimage = QImage(
            np_array.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        return QPixmap.fromImage(qimage)

    def redraw(self):
        viewmat = self._viewmat_from_sliders()
        self.viewmat = viewmat

        if len(self.points_features) > 0:
            mask_3d = segmentor.get_point_prompt_mask(self.points_features, self.points_3d_categories)
        else:
            mask_3d = torch.ones(segmentor.features.shape[0], device="cuda", dtype=torch.bool)
        self.mask_3d = mask_3d

        
        img = segmentor.render_with_mask_3d(viewmat, self.mask_3d)
        pixmap = self._pixmap_from_cv(img)
        self.image_viewer.image_item.setPixmap(pixmap)

        scene = self.image_viewer.scene
        for item in self.points_3d_items:
            scene.removeItem(item)
        self.points_3d_items = []

        for (x, y, z), cat in zip(self.points_3d, self.points_3d_categories):
            print(x, y, z)
            viewmat_np = viewmat.cpu().numpy()
            x, y, z, _ = viewmat_np @ np.array([[x], [y], [z], [1]])
            fx = segmentor.splats["camera_matrix"][0, 0]
            fy = segmentor.splats["camera_matrix"][1, 1]
            cx = segmentor.splats["camera_matrix"][0, 2].item()
            cy = segmentor.splats["camera_matrix"][1, 2].item()
            fx, fy = fx.item(), fy.item()
            width = segmentor.splats["camera_matrix"][0, 2] * 2
            height = segmentor.splats["camera_matrix"][1, 2] * 2
            x = (x / z) * fx + cx
            y = (y / z) * fy + cy
            print(x, y)
            item = QGraphicsEllipseItem(x, y, 25, 25)
            if cat == 1:
                item.setBrush(QBrush(QColor(0, 255, 0)))
            else:
                item.setBrush(QBrush(QColor(255, 0, 0)))
            scene.addItem(item)
            self.points_3d_items.append(item)

        
        output = segmentor.render_3d_mask(viewmat, mask_3d)
        import cv2
        cv2.imshow("output", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            exit()

    def zoom_image(self, factor=1):
        self.image_viewer.scale(factor, factor)

    def context_menu(self, pos):
        context_menu = self.create_context_menu()
        context_menu.exec_(self.mapToGlobal(pos))

    def create_context_menu(self):
        context_menu = QMenu(self)
        undo_action = QAction(QIcon.fromTheme("edit-undo"), "Undo", self)
        context_menu.addAction(undo_action)
        undo_action.triggered.connect(self.undo_action)
        redo_action = QAction(QIcon.fromTheme("edit-redo"), "Redo", self)
        redo_action.triggered.connect(self.redo_action)
        context_menu.addAction(redo_action)
        return context_menu

    def create_menus(self):
        """Create all necessary menus."""
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction(new_action := self.create_action("New", self.new_file))
        file_menu.addAction(self.create_action("Open", self.open_file))
        file_menu.addSeparator()
        file_menu.addAction(exit_action := self.create_action("Exit", self.close))
        exit_action.setIcon(QIcon.fromTheme("application-exit"))

        exit_action.setShortcut("Ctrl+Q")

        # Edit Menu
        edit_menu = menubar.addMenu("Edit")
        undo_action = QAction(QIcon.fromTheme("edit-undo"), "Undo", self)
        undo_action.triggered.connect(self.undo_action)
        # Add shortcut
        undo_action.setShortcut("Ctrl+Z")
        edit_menu.addAction(undo_action)
        redo_action = QAction(QIcon.fromTheme("edit-redo"), "Redo", self)
        redo_action.triggered.connect(self.redo_action)
        # Add shortcut
        redo_action.setShortcut("Ctrl+Shift+Z")
        edit_menu.addAction(redo_action)

        # Tools Menu
        tools_menu = menubar.addMenu("Tools")
        tools_action_group = QActionGroup(self)
        point_prompt_action = QAction("Point Prompt", self)
        point_prompt_action.setCheckable(True)
        point_prompt_action.triggered.connect(self.set_point_prompt_tool)
        tools_action_group.addAction(point_prompt_action)

        

        circle_action = QAction("Circle", self)
        circle_action.setCheckable(True)
        circle_action.triggered.connect(lambda: self.image_viewer.set_tool(CircleTool()))
        tools_action_group.addAction(circle_action)
        tools_menu.addAction(circle_action)

        rectangle_action = QAction("Rectangle", self)
        rectangle_action.setCheckable(True)
        rectangle_action.triggered.connect(lambda: self.image_viewer.set_tool(RectangleTool()))
        tools_action_group.addAction(rectangle_action)
        tools_menu.addAction(rectangle_action)

        line_action = QAction("Line", self)
        line_action.setCheckable(True)
        tools_action_group.addAction(line_action)
        tools_menu.addAction(line_action)
        line_action.triggered.connect(lambda: self.image_viewer.set_tool(LineTool()))

        polygon_action = QAction("Polygon", self)
        polygon_action.setCheckable(True)
        polygon_action.triggered.connect(lambda: self.image_viewer.set_tool(PolygonTool()))
        tools_action_group.addAction(polygon_action)
        tools_menu.addAction(polygon_action)


        rect_action = QAction("Rectangle", self)
        rect_action.setCheckable(True)
        tools_action_group.addAction(rect_action)
        tools_menu.addAction(rect_action)
        rect_action.triggered.connect(lambda: self.image_viewer.set_tool(RectangleTool()))


        brush_action = QAction("Brush", self)
        brush_action.setCheckable(True)
        tools_action_group.addAction(brush_action)
        tools_menu.addAction(brush_action)
        tools_menu.addAction(point_prompt_action)

        eraser_action = QAction("Eraser", self)
        eraser_action.setCheckable(True)
        tools_action_group.addAction(eraser_action)
        tools_menu.addAction(eraser_action)

        

        point_prompt_action.setChecked(True)
        point_prompt_action.triggered.connect(lambda: self.image_viewer.set_tool(PromptTool()))

        self.tools_action_group = tools_action_group

        # Help Menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction(self.create_action("About", self.show_about))

        zoom_menu = menubar.addMenu("Zoom")
        zoom_menu.addAction(zoom_in_action := self.create_action("Zoom In", lambda: self.zoom_image(2)))
        zoom_in_action.setShortcut("Ctrl+=")
        zoom_in_action.setIcon(QIcon.fromTheme("zoom-in"))
        zoom_menu.addAction(
            zoom_out_action := self.create_action("Zoom Out", lambda: self.zoom_image(0.5))
        )
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.setIcon(QIcon.fromTheme("zoom-out"))
        zoom_menu.addAction(
            reset_zoom_action := self.create_action("Reset Zoom", lambda: self.image_viewer.resetTransform())
        )
        reset_zoom_action.setShortcut("Ctrl+0")
        reset_zoom_action.setIcon(QIcon.fromTheme("zoom-original"))
        fit_to_window_action = zoom_menu.addAction("Fit to Window")
        fit_to_window_action.setShortcut("Ctrl+9")
        fit_to_window_action.setIcon(QIcon.fromTheme("zoom-fit-best"))
        fit_to_window_action.triggered.connect(lambda: self.fit_to_window())
        zoom_menu.addAction(fit_to_window_action)

    def fit_to_window(self):
        scroll_area = self.scroll_area
        scroll_area_width = scroll_area.width()
        scroll_area_height = scroll_area.height()
        factor = min(scroll_area_width / width, scroll_area_height / height)
        self.image_viewer.resetTransform()
        self.zoom_image(factor)

    def set_point_prompt_tool(self):
        pass

    def set_line_prompt_tool(self):
        pass

    def create_action(self, name, slot):
        """Helper method to create actions."""
        action = QAction(name, self)
        action.triggered.connect(slot)
        return action

    # Menu Action Handlers
    def new_file(self):
        print("New File action triggered")

    def open_file(self):
        print("Open File action triggered")

    def undo_action(self):
        self.image_viewer.undo_stack.undo()

    def redo_action(self):
        self.image_viewer.undo_stack.redo()

    def show_about(self):
        QMessageBox.about(
            self,
            "About",
            "Maintainer: <a href='https://github.com/jojijoseph'>Joji</a>",
        )

    def closeEvent(self, event):
        # Show confirmation dialog before closing the window
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure to quit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
