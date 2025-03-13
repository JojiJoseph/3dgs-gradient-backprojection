from PyQt5.QtGui import QImage, QPixmap, QPainter, QBrush, QPen, QColor
from PyQt5.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsEllipseItem,
    QUndoStack,
    QUndoCommand,
    QGraphicsRectItem,
    QGraphicsLineItem,
    QGraphicsPolygonItem
)
from PyQt5.QtCore import QPointF, Qt, QLineF, QRectF
from PyQt5.QtGui import QPolygonF
import random

from tools import CircleTool, PromptTool







class ImageViewer(QGraphicsView):
    def __init__(self, numpy_image):
        super().__init__()

        # Convert NumPy image to QPixmap
        height, width, channels = numpy_image.shape
        bytes_per_line = channels * width
        qimage = QImage(
            numpy_image.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimage)

        # Scene to hold image
        self.scene = QGraphicsScene(self)
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)

        # Set up the view
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(
            QGraphicsView.AnchorUnderMouse
        )  # Zooms relative to mouse
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)  # Enable panning

        self.zoom_factor = 1.0
        # Undo stack
        self.undo_stack = QUndoStack(self)

        self.active_tool = PromptTool()

    def wheelEvent(self, event):
        """Handle zoom on mouse wheel."""
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor  # Zoom in
        else:
            zoom_factor = zoom_out_factor  # Zoom out

        # Apply zoom with transformation
        self.scale(zoom_factor, zoom_factor)

    def set_tool(self, tool):
        """Switch active tool."""
        self.active_tool = tool

    def mousePressEvent(self, event):
        """Delegate mouse press to active tool."""
        self.active_tool.on_mouse_press(self, event)

    def mouseMoveEvent(self, event):
        """Delegate mouse move to active tool."""
        self.active_tool.on_mouse_move(self, event)

    def mouseReleaseEvent(self, event):
        """Delegate mouse release to active tool."""
        self.active_tool.on_mouse_release(self, event)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for Undo/Redo"""
        if event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self.undo_stack.undo()  # Ctrl+Z to Undo
        elif event.key() == Qt.Key_Y and event.modifiers() & Qt.ControlModifier:
            self.undo_stack.redo()  # Ctrl+Y to Redo
        else:
            self.active_tool.on_key_press(self, event)
