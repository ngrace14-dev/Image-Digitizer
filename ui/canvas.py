"""Interactive QGraphicsView viewport for rendering physical thread paths and needle penetrations."""

from typing import Sequence, Tuple
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView
from PyQt6.QtGui import QPen, QColor, QPainter, QWheelEvent
from PyQt6.QtCore import Qt

Point2D = Tuple[float, float]


class StitchViewport(QGraphicsView):
    """Interactive canvas that renders physical stitch coordinates and needle drops."""

    def __init__(self) -> None:
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Enable high-quality rendering (crucial for "TrueView" aesthetics)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QColor("#f0f0f0"))

        # Enable dragging to pan the canvas
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def render_stitches(self, stitches: Sequence[Point2D]) -> None:
        """Draws the physical thread path and needle penetrations."""
        self.scene.clear()

        if not stitches or len(stitches) < 2:
            return

        # Pen for the thread (Madeira blue)
        thread_pen = QPen(QColor(30, 144, 255))
        thread_pen.setWidthF(0.5)
        thread_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

        # Pen for the needle penetrations
        needle_pen = QPen(QColor(0, 0, 0))
        needle_pen.setWidthF(0)

        for i in range(len(stitches) - 1):
            x1, y1 = stitches[i]
            x2, y2 = stitches[i + 1]

            self.scene.addLine(x1, y1, x2, y2, thread_pen)
            self.scene.addEllipse(x1 - 0.2, y1 - 0.2, 0.4, 0.4, needle_pen)

        # Draw final needle drop
        last_x, last_y = stitches[-1]
        self.scene.addEllipse(last_x - 0.2, last_y - 0.2, 0.4, 0.4, needle_pen)

        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.scale(0.8, 0.8)

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        """Allow smooth mouse-wheel zooming."""
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

