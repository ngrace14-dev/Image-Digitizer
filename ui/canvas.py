"""Interactive QGraphicsView viewport for rendering physical thread paths and editable needle penetrations."""

import math
from typing import Callable, List, Optional, Sequence, Tuple
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView
from PyQt6.QtGui import QPen, QBrush, QColor, QPainter, QWheelEvent, QMouseEvent
from PyQt6.QtCore import Qt

Point2D = Tuple[float, float]


class StitchViewport(QGraphicsView):
    """Interactive canvas that renders physical stitch coordinates and supports manual stitch marker editing."""

    def __init__(self, on_stitches_changed: Optional[Callable[[List[Point2D]], None]] = None) -> None:
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.on_stitches_changed = on_stitches_changed

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QColor("#f0f0f0"))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        self._stitches: List[Point2D] = []
        self.edit_mode: bool = False
        self.hide_jumps: bool = False
        self._dragging_idx: int = -1
        self._auto_fit_done: bool = False

    def set_edit_mode(self, enabled: bool) -> None:
        self.edit_mode = enabled
        self.setDragMode(
            QGraphicsView.DragMode.NoDrag if enabled else QGraphicsView.DragMode.ScrollHandDrag
        )
        self.render_stitches(self._stitches, preserve_view=True)

    def render_stitches(
        self,
        stitches: Sequence[Point2D],
        preserve_view: bool = False,
        hide_jumps: Optional[bool] = None,
    ) -> None:
        if hide_jumps is not None:
            self.hide_jumps = hide_jumps
        self._stitches = list(stitches)
        self.scene.clear()

        if len(self._stitches) < 2:
            return

        thread_pen = QPen(QColor(30, 144, 255))
        thread_pen.setWidthF(0.45)
        thread_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

        jump_pen = QPen(QColor(220, 53, 69, 140))
        jump_pen.setWidthF(0.25)
        jump_pen.setStyle(Qt.PenStyle.DashLine)

        needle_pen = QPen(QColor(17, 17, 17))
        needle_pen.setWidthF(0.08 if self.edit_mode else 0.0)
        needle_brush = QBrush(QColor("#ffd166") if self.edit_mode else QColor(17, 17, 17))
        r = 0.35 if self.edit_mode else 0.18

        for i in range(len(self._stitches) - 1):
            x1, y1 = self._stitches[i]
            x2, y2 = self._stitches[i + 1]
            if math.hypot(x2 - x1, y2 - y1) > 7.0:
                if not self.hide_jumps:
                    self.scene.addLine(x1, y1, x2, y2, jump_pen)
            else:
                self.scene.addLine(x1, y1, x2, y2, thread_pen)
            self.scene.addEllipse(x1 - r, y1 - r, r * 2, r * 2, needle_pen, needle_brush)

        last_x, last_y = self._stitches[-1]
        self.scene.addEllipse(last_x - r, last_y - r, r * 2, r * 2, needle_pen, needle_brush)

        if not preserve_view and not self._auto_fit_done:
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.scale(0.85, 0.85)
            self._auto_fit_done = True

    def _find_nearest_stitch(self, x_mm: float, y_mm: float, max_dist_mm: float = 1.2) -> int:
        best_idx = -1
        best_dist = max_dist_mm
        for idx, (sx, sy) in enumerate(self._stitches):
            d = math.hypot(sx - x_mm, sy - y_mm)
            if d <= best_dist:
                best_dist = d
                best_idx = idx
        return best_idx

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is None or not self.edit_mode:
            super().mousePressEvent(event)
            return

        scene_pt = self.mapToScene(event.pos())
        x_mm, y_mm = float(scene_pt.x()), float(scene_pt.y())

        if event.button() == Qt.MouseButton.RightButton:
            hit_idx = self._find_nearest_stitch(x_mm, y_mm)
            if hit_idx >= 0 and len(self._stitches) > 2:
                self._stitches.pop(hit_idx)
                self.render_stitches(self._stitches, preserve_view=True)
                if self.on_stitches_changed:
                    self.on_stitches_changed(self._stitches)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._stitches.append((x_mm, y_mm))
                self._dragging_idx = len(self._stitches) - 1
                self.render_stitches(self._stitches, preserve_view=True)
                if self.on_stitches_changed:
                    self.on_stitches_changed(self._stitches)
                return

            hit_idx = self._find_nearest_stitch(x_mm, y_mm)
            if hit_idx >= 0:
                self._dragging_idx = hit_idx
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is not None and self.edit_mode and self._dragging_idx >= 0:
            scene_pt = self.mapToScene(event.pos())
            self._stitches[self._dragging_idx] = (float(scene_pt.x()), float(scene_pt.y()))
            self.render_stitches(self._stitches, preserve_view=True)
            if self.on_stitches_changed:
                self.on_stitches_changed(self._stitches)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if self._dragging_idx >= 0:
            self._dragging_idx = -1
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is None:
            return
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(zoom_factor, zoom_factor)


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

    mock_stitches = [
        (0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (0.0, 0.0),
        (10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0), (10.0, 10.0),
        (20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0), (20.0, 20.0),
    ]

    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("Digitizer Prototype - Stitch Viewport")
    win.resize(800, 600)
    central = QWidget()
    win.setCentralWidget(central)
    layout = QVBoxLayout(central)
    vp = StitchViewport()
    layout.addWidget(vp)
    vp.render_stitches(mock_stitches)
    win.show()
    sys.exit(app.exec())

