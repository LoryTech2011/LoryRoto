import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class ViewerWidget(QWidget):
    point_clicked = Signal(float, float, int)
    brush_stroke = Signal(object, str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 320)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.frame: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.points: list[dict] = []
        self.overlays: list[dict] = []
        self.overlay_color = "#43e09b"
        self.point_label = 1
        self.brush_action: str | None = None
        self.brush_radius = 20
        self.display_mode = "overlay"
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self._dragging_pan = False
        self._brush_points: list[tuple[float, float]] = []
        self._brush_preview_action: str | None = None
        self._image_rect = QRectF()

    def set_frame(self, frame: np.ndarray | None) -> None:
        self.frame = frame
        self.update()

    def set_overlay(self, mask: np.ndarray | None, points: list[dict]) -> None:
        self.mask = mask
        self.points = points
        self.overlays = []
        self.update()

    def set_subject_overlays(self, overlays: list[dict]) -> None:
        self.overlays = overlays
        self.points = []
        masks = [overlay["mask"] for overlay in overlays]
        self.mask = np.logical_or.reduce(masks) if masks else None
        self.update()

    def set_display_mode(self, mode: str) -> None:
        self.display_mode = mode
        self.update()

    def _fit_rect(self) -> QRectF:
        if self.frame is None:
            return QRectF()
        height, width = self.frame.shape[:2]
        scale = min(self.width() / width, self.height() / height) * self.zoom
        draw_width, draw_height = width * scale, height * scale
        x = (self.width() - draw_width) / 2 + self.pan.x()
        y = (self.height() - draw_height) / 2 + self.pan.y()
        return QRectF(x, y, draw_width, draw_height)

    def _image_point(self, position: QPointF) -> tuple[float, float] | None:
        rect = self._fit_rect()
        if not rect.contains(position) or self.frame is None:
            return None
        height, width = self.frame.shape[:2]
        x = (position.x() - rect.x()) * width / rect.width()
        y = (position.y() - rect.y()) * height / rect.height()
        return min(max(x, 0), width - 1), min(max(y, 0), height - 1)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111820"))
        if self.frame is None:
            painter.setPen(QColor("#a9b4c0"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open or drag a video")
            painter.end()
            return

        rect = self._fit_rect()
        self._image_rect = rect
        height, width = self.frame.shape[:2]
        if self.display_mode == "mask":
            pixels = (
                np.zeros((height, width), dtype=np.uint8)
                if self.mask is None
                else self.mask.astype(np.uint8) * 255
            )
            image = QImage(
                pixels.data,
                width,
                height,
                pixels.strides[0],
                QImage.Format.Format_Grayscale8,
            )
        elif self.display_mode == "isolated":
            alpha = (
                np.zeros((height, width), dtype=np.uint8)
                if self.mask is None
                else self.mask.astype(np.uint8) * 255
            )
            pixels = np.dstack((self.frame, alpha))
            image = QImage(
                pixels.data,
                width,
                height,
                pixels.strides[0],
                QImage.Format.Format_RGBA8888,
            )
        else:
            image = QImage(
                self.frame.data,
                width,
                height,
                self.frame.strides[0],
                QImage.Format.Format_RGB888,
            )
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(rect, image)

        if self.display_mode == "overlay":
            layers = self.overlays or (
                [{"mask": self.mask, "points": self.points, "color": "#43e09b"}]
                if self.mask is not None
                else []
            )
            for layer in layers:
                layer_mask = layer["mask"]
                if layer_mask is None or layer_mask.shape != (height, width):
                    continue
                color = QColor(layer["color"])
                color.setAlpha(105)
                overlay = np.zeros((height, width, 4), dtype=np.uint8)
                overlay[layer_mask] = (
                    color.red(),
                    color.green(),
                    color.blue(),
                    color.alpha(),
                )
                mask_image = QImage(
                    overlay.data,
                    width,
                    height,
                    overlay.strides[0],
                    QImage.Format.Format_RGBA8888,
                )
                painter.drawImage(rect, mask_image)

        sx, sy = rect.width() / width, rect.height() / height
        point_layers = self.overlays or [
            {"points": self.points, "color": "#43e09b"}
        ]
        for layer in point_layers:
            for point in layer["points"]:
                color = QColor(
                    layer["color"] if point["label"] == 1 else "#ff647c"
                )
                painter.setPen(QPen(color, 2))
                x = rect.x() + float(point["x"]) * sx
                y = rect.y() + float(point["y"]) * sy
                painter.drawEllipse(QPointF(x, y), 6, 6)
                painter.drawLine(QPointF(x - 9, y), QPointF(x + 9, y))
                painter.drawLine(QPointF(x, y - 9), QPointF(x, y + 9))
        if self._brush_points:
            color = QColor(
                self.overlay_color
                if self._brush_preview_action == "add"
                else "#ff647c"
            )
            color.setAlpha(105)
            pen = QPen(
                color,
                max(1.0, 2 * self.brush_radius * sx),
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            painter.setPen(pen)
            painter.setBrush(color)
            painter.setClipRect(rect)
            brush_path = QPainterPath()
            first_x, first_y = self._brush_points[0]
            first_point = QPointF(rect.x() + first_x * sx, rect.y() + first_y * sy)
            brush_path.moveTo(first_point)
            if len(self._brush_points) == 1:
                radius = max(0.5, self.brush_radius * sx)
                painter.drawEllipse(first_point, radius, radius)
            else:
                for x, y in self._brush_points[1:]:
                    brush_path.lineTo(rect.x() + x * sx, rect.y() + y * sy)
                painter.drawPath(brush_path)
        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._dragging_pan = True
            self._last_position = event.position()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self._image_point(event.position())
        if point is None:
            return
        if self.brush_action:
            self._brush_points = [point]
            self._brush_preview_action = self.brush_action
            self.update()
        else:
            self.point_clicked.emit(point[0], point[1], self.point_label)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging_pan:
            delta = event.position() - self._last_position
            self.pan += delta
            self._last_position = event.position()
            self.update()
            return
        if self._brush_points:
            point = self._image_point(event.position())
            if point is not None:
                self._brush_points.append(point)
                self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._dragging_pan = False
            return
        if event.button() == Qt.MouseButton.LeftButton and self._brush_points:
            self.brush_stroke.emit(
                self._brush_points,
                self._brush_preview_action or "add",
                self.brush_radius,
            )
            self._brush_points = []
            self._brush_preview_action = None
            self.update()

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.zoom = min(max(self.zoom * factor, 0.25), 8.0)
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update()
