"""Main application window for the Image Digitizer prototype."""

from typing import Sequence, Tuple
from PyQt6.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
    QComboBox,
    QFileDialog,
    QMessageBox,
)
from ui.canvas import StitchViewport
from core.stitch_engine import generate_concentric_fill, generate_tatami_fill
from io.dst_writer import write_dst

Point2D = Tuple[float, float]


class MainWindow(QMainWindow):
    """Main window hosting the StitchViewport and interactive toolbar controls."""

    def __init__(self, stitches: Sequence[Point2D] | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Digitizer Prototype - Stitch Viewport")
        self.resize(900, 650)

        self._sample_boundary: list[Point2D] = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        ]
        self.current_stitches: list[Point2D] = list(stitches) if stitches else []

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Toolbar controls
        controls_layout = QHBoxLayout()
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Concentric Fill", "Tatami Fill (0°)", "Tatami Fill (45°)"])
        self.mode_selector.currentIndexChanged.connect(self._regenerate_from_mode)

        self.export_btn = QPushButton("Export .DST")
        self.export_btn.clicked.connect(self._export_dst)

        self.status_label = QLabel()

        controls_layout.addWidget(QLabel("Fill Mode:"))
        controls_layout.addWidget(self.mode_selector)
        controls_layout.addWidget(self.export_btn)
        controls_layout.addStretch()
        controls_layout.addWidget(self.status_label)

        main_layout.addLayout(controls_layout)

        # Add our custom viewport
        self.viewport = StitchViewport()
        main_layout.addWidget(self.viewport)

        if self.current_stitches:
            self._update_viewport(self.current_stitches)
        else:
            self._regenerate_from_mode()

    def _update_viewport(self, stitches: Sequence[Point2D]) -> None:
        self.current_stitches = list(stitches)
        self.viewport.render_stitches(self.current_stitches)
        self.status_label.setText(f"Stitches: {len(self.current_stitches)}")

    def _regenerate_from_mode(self) -> None:
        mode = self.mode_selector.currentText()
        if mode == "Concentric Fill":
            stitches = generate_concentric_fill(self._sample_boundary, stitch_length_mm=2.5, pixel_scale_mm=0.25)
        elif mode == "Tatami Fill (0°)":
            stitches = generate_tatami_fill(
                self._sample_boundary,
                stitch_length_mm=3.5,
                row_spacing_mm=1.0,
                angle_deg=0.0,
                pixel_scale_mm=0.25,
            )
        else:
            stitches = generate_tatami_fill(
                self._sample_boundary,
                stitch_length_mm=3.5,
                row_spacing_mm=1.0,
                angle_deg=45.0,
                pixel_scale_mm=0.25,
            )
        self._update_viewport(stitches)

    def _export_dst(self) -> None:
        if not self.current_stitches:
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Tajima DST",
            "design.dst",
            "Tajima Embroidery (*.dst)",
        )
        if filepath:
            write_dst(filepath, self.current_stitches)
            QMessageBox.information(self, "Export Complete", f"Saved {len(self.current_stitches)} stitches to:\n{filepath}")
