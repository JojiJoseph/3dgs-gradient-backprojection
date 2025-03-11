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


class Tool:
    def on_mouse_press(self, viewer, event):
        pass

    def on_mouse_move(self, viewer, event):
        pass

    def on_mouse_release(self, viewer, event):
        pass

    def on_key_press(self, viewer, event):
        pass


class CircleTool(Tool):
    def on_mouse_press(self, viewer, event):
        if event.button() == Qt.LeftButton:
            scene_pos = viewer.mapToScene(event.pos())
            command = AddCircleCommand(viewer.scene, scene_pos)
            viewer.undo_stack.push(command)


class RectangleTool(Tool):
    def on_mouse_press(self, viewer, event):
        if event.button() == Qt.LeftButton:
            scene_pos = viewer.mapToScene(event.pos())
            command = AddRectangleCommand(viewer.scene, scene_pos)
            viewer.undo_stack.push(command)


class AddRectangleCommand(QUndoCommand):
    """Command to add a rectangle, supporting undo and redo."""

    def __init__(self, scene, pos, width=20, height=20):
        super().__init__("Add Rectangle")
        self.scene = scene
        self.pos = pos
        self.width = width
        self.height = height
        self.rectangle = None  # Will be assigned when executed

    def redo(self):
        """Redo the action (add the rectangle)"""
        if self.rectangle is None:
            color = QColor(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            )  # Random color
            self.rectangle = QGraphicsRectItem(
                self.pos.x(), self.pos.y(), self.width, self.height
            )
            self.rectangle.setBrush(QBrush(color))  # Fill color
            self.rectangle.setPen(QPen(Qt.black, 2))  # Border
        self.scene.addItem(self.rectangle)  # Add to scene

    def undo(self):
        """Undo the action (remove the rectangle)"""
        if self.rectangle:
            self.scene.removeItem(self.rectangle)


class AddCircleCommand(QUndoCommand):
    """Command to add a circle, supporting undo and redo."""

    def __init__(self, scene, pos, radius=20):
        super().__init__("Add Circle")
        self.scene = scene
        self.pos = pos
        self.radius = radius
        self.circle = None  # Will be assigned when executed

    def redo(self):
        """Redo the action (add the circle)"""
        if self.circle is None:
            color = QColor(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            )  # Random color
            self.circle = QGraphicsEllipseItem(
                self.pos.x() - self.radius,
                self.pos.y() - self.radius,
                self.radius * 2,
                self.radius * 2,
            )
            self.circle.setBrush(QBrush(color))  # Fill color
            self.circle.setPen(QPen(Qt.black, 2))  # Border
        self.scene.addItem(self.circle)  # Add to scene

    def undo(self):
        """Undo the action (remove the circle)"""
        if self.circle:
            self.scene.removeItem(self.circle)


class AddLineCommand(QUndoCommand):
    def __init__(self, scene, start, end):
        super().__init__("Draw Line")
        self.scene = scene
        self.start = start
        self.end = end
        self.line_item = None

    def redo(self):
        """Add the line to the scene"""
        if self.line_item is None:
            self.line_item = QGraphicsLineItem(QLineF(self.start, self.end))
            self.line_item.setPen(QPen(Qt.black, 2))
        self.scene.addItem(self.line_item)

    def undo(self):
        """Remove the line from the scene"""
        if self.line_item:
            self.scene.removeItem(self.line_item)


class LineTool(Tool):
    def __init__(self):
        super().__init__()
        self.start_pos = None  # Where the line starts
        self.preview_line = None  # Temporary QGraphicsLineItem

    def on_mouse_press(self, viewer, event):
        """Start drawing a line on left-click"""
        if event.button() == Qt.LeftButton:
            self.start_pos = viewer.mapToScene(event.pos())  # Store start position
            # Create a temporary preview line
            self.preview_line = QGraphicsLineItem(
                QLineF(self.start_pos, self.start_pos)
            )
            self.preview_line.setPen(QPen(Qt.red, 2, Qt.DashLine))  # Red dashed preview
            viewer.scene.addItem(self.preview_line)

    def on_mouse_move(self, viewer, event):
        """Update preview line as mouse moves"""
        if self.preview_line:
            end_pos = viewer.mapToScene(event.pos())  # Get new endpoint
            self.preview_line.setLine(QLineF(self.start_pos, end_pos))  # Update line

    def on_mouse_release(self, viewer, event):
        """Finalize the line and add it to the undo stack"""
        if event.button() == Qt.LeftButton and self.preview_line:
            end_pos = viewer.mapToScene(event.pos())

            # Remove preview line
            viewer.scene.removeItem(self.preview_line)
            self.preview_line = None

            # Push final line to the undo stack
            command = AddLineCommand(viewer.scene, self.start_pos, end_pos)
            viewer.undo_stack.push(command)

            self.start_pos = None  # Reset for next line

class AddPolygonCommand(QUndoCommand):
    def __init__(self, scene, points):
        super().__init__("Draw Polygon")
        self.scene = scene
        self.points = points
        self.polygon_item = None

    def redo(self):
        """Adds the polygon to the scene."""
        if self.polygon_item is None:
            polygon = QPolygonF(self.points)
            self.polygon_item = QGraphicsPolygonItem(polygon)
            self.polygon_item.setPen(QPen(Qt.black, 2))
        self.scene.addItem(self.polygon_item)

    def undo(self):
        """Removes the polygon from the scene."""
        if self.polygon_item:
            self.scene.removeItem(self.polygon_item)


class PolygonTool(Tool):
    def __init__(self):
        super().__init__()
        self.points = []  # Stores polygon points
        self.preview_polygon = None  # Temporary polygon preview
        self.first_point = None  # Stores the first point for closure check

    def on_mouse_press(self, viewer, event):
        """Add points to the polygon preview."""
        if event.button() == Qt.LeftButton:
            scene_pos = viewer.mapToScene(event.pos())

            if not self.points:
                self.first_point = scene_pos  # Store the first point for closure check

            # Check if clicking near the first point to close polygon
            if self.first_point and len(self.points) >= 3:
                if (scene_pos - self.first_point).manhattanLength() < 10:  # Close polygon
                    self.finalize_polygon(viewer)
                    return

            self.points.append(scene_pos)  # Add point to polygon
            
            # Update the preview polygon
            self.update_preview(viewer)

    def on_mouse_move(self, viewer, event):
        """Update the preview polygon when moving the mouse."""
        if self.points:
            temp_points = self.points + [viewer.mapToScene(event.pos())]
            self.update_preview(viewer, temp_points)

    def on_key_press(self, viewer, event):
        """Complete polygon on Enter key."""
        if event.key() == Qt.Key_Return and len(self.points) >= 3:
            self.finalize_polygon(viewer)
        print(event.key())

    def update_preview(self, viewer, temp_points=None):
        """Updates the preview polygon dynamically."""
        if self.preview_polygon:
            viewer.scene.removeItem(self.preview_polygon)  # Remove old preview

        polygon = QPolygonF(temp_points or self.points)
        self.preview_polygon = QGraphicsPolygonItem(polygon)
        self.preview_polygon.setPen(QPen(Qt.red, 2, Qt.DashLine))  # Dashed red preview
        viewer.scene.addItem(self.preview_polygon)

    def finalize_polygon(self, viewer):
        """Commit the polygon to the undo stack and reset the tool."""
        if len(self.points) >= 3:
            viewer.scene.removeItem(self.preview_polygon)  # Remove preview
            command = AddPolygonCommand(viewer.scene, self.points)
            viewer.undo_stack.push(command)

        # Reset for next polygon
        self.points = []
        self.preview_polygon = None
        self.first_point = None


class RectangleTool(Tool):
    def __init__(self):
        super().__init__()
        self.start_pos = None  # First corner
        self.preview_rect = None  # Temporary preview rectangle

    def on_mouse_press(self, viewer, event):
        """Start rectangle on first click, finalize on second click."""
        if event.button() == Qt.LeftButton:
            scene_pos = viewer.mapToScene(event.pos())

            if self.start_pos is None:
                # First click: Store start position
                self.start_pos = scene_pos
                self.preview_rect = QGraphicsRectItem(QRectF(self.start_pos, self.start_pos))
                self.preview_rect.setPen(QPen(Qt.red, 2, Qt.DashLine))  # Dashed red preview
                viewer.scene.addItem(self.preview_rect)
            else:
                # Second click: Finalize the rectangle
                self.finalize_rectangle(viewer, scene_pos)

    def on_mouse_move(self, viewer, event):
        """Update preview rectangle on mouse move."""
        if self.start_pos and self.preview_rect:
            end_pos = viewer.mapToScene(event.pos())
            rect = QRectF(self.start_pos, end_pos).normalized()
            self.preview_rect.setRect(rect)

    def finalize_rectangle(self, viewer, end_pos):
        """Commit the rectangle to the undo stack."""
        if self.preview_rect:
            viewer.scene.removeItem(self.preview_rect)  # Remove preview

        # Push the final rectangle as a command for undo/redo
        command = AddRectangleCommand(viewer.scene, self.start_pos, end_pos)
        viewer.undo_stack.push(command)

        # Reset tool state
        self.start_pos = None
        self.preview_rect = None

class AddRectangleCommand(QUndoCommand):
    def __init__(self, scene, start, end):
        super().__init__("Draw Rectangle")
        self.scene = scene
        self.start = start
        self.end = end
        self.rect_item = None

    def redo(self):
        """Adds the rectangle to the scene."""
        if self.rect_item is None:
            rect = QRectF(self.start, self.end).normalized()
            self.rect_item = QGraphicsRectItem(rect)
            self.rect_item.setPen(QPen(Qt.black, 2))
        self.scene.addItem(self.rect_item)

    def undo(self):
        """Removes the rectangle from the scene."""
        if self.rect_item:
            self.scene.removeItem(self.rect_item)




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

        self.active_tool = CircleTool()

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

    # def mousePressEvent(self, event):
    #     """ Handle mouse clicks to place circles with undo support """
    #     if event.button() == Qt.LeftButton:  # Left-click to place a circle
    #         scene_pos = self.mapToScene(event.pos())  # Convert to scene coordinates
    #         command = AddCircleCommand(self.scene, scene_pos)
    #         self.undo_stack.push(command)

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
