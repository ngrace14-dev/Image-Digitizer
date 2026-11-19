"""Main application window for the Image Digitizer prototype."""

from typing import Sequence, Tuple
from PyQt6.QtCore import Qt, QTimer
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
    QDialog,
    QSlider,
    QCheckBox,
)
from ui.canvas import StitchViewport
from core.geometry import calculate_distance
from core.stitch_engine import generate_concentric_fill, generate_tatami_fill
from io.dst_writer import write_dst

Point2D = Tuple[float, float]


class ExportPreviewDialog(QDialog):
    """Modal dialog showing a realistic stitch-out preview and telemetry before exporting to .DST."""

    def __init__(self, parent: QWidget | None, stitches: Sequence[Point2D]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stitch-Out Preview — Confirm Before Exporting .DST")
        self.resize(760, 580)
        self.stitches = list(stitches)
        self.visible_count = len(self.stitches)

        layout = QVBoxLayout(self)

        jumps = sum(
            1
            for i in range(1, len(self.stitches))
            if calculate_distance(self.stitches[i - 1], self.stitches[i]) > 7.0
        )
        header_label = QLabel(
            f"<b>Pre-Export Sew-Out Simulation</b> &nbsp;|&nbsp; "
            f"Total Stitches: <b>{len(self.stitches)}</b> &nbsp;|&nbsp; "
            f"Jump Stitches (&gt;7mm): <b>{jumps}</b>"
        )
        layout.addWidget(header_label)

        self.preview_viewport = StitchViewport()
        self.preview_viewport.render_stitches(self.stitches)
        layout.addWidget(self.preview_viewport, stretch=1)

        sim_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶ Simulate Sew-Out")
        self.play_btn.clicked.connect(self._toggle_play)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(2)
        self.slider.setMaximum(max(2, len(self.stitches)))
        self.slider.setValue(len(self.stitches))
        self.slider.valueChanged.connect(self._on_scrub)

        self.hide_jumps_chk = QCheckBox("Simulate Trimmed Jumps (Final Look)")
        self.hide_jumps_chk.setChecked(True)
        self.hide_jumps_chk.toggled.connect(self._refresh_preview)

        sim_layout.addWidget(self.play_btn)
        sim_layout.addWidget(self.slider, stretch=1)
        sim_layout.addWidget(self.hide_jumps_chk)
        layout.addLayout(sim_layout)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Back to Editor")
        cancel_btn.clicked.connect(self.reject)
        confirm_btn = QPushButton("Confirm & Save .DST")
        confirm_btn.setDefault(True)
        confirm_btn.clicked.connect(self.accept)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(confirm_btn)
        layout.addLayout(btn_layout)

        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._step_simulation)

    def _toggle_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("▶ Simulate Sew-Out")
        else:
            if self.slider.value() >= len(self.stitches):
                self.slider.setValue(2)
            self.timer.start()
            self.play_btn.setText("⏸ Pause")

    def _step_simulation(self) -> None:
        step = max(1, len(self.stitches) // 80)
        nxt = min(len(self.stitches), self.slider.value() + step)
        self.slider.setValue(nxt)
        if nxt >= len(self.stitches):
            self.timer.stop()
            self.play_btn.setText("▶ Simulate Sew-Out")

    def _on_scrub(self, val: int) -> None:
        self.visible_count = val
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        self.preview_viewport.render_stitches(
            self.stitches[: self.visible_count],
            preserve_view=True,
            hide_jumps=self.hide_jumps_chk.isChecked(),
        )


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

        self.edit_markers_chk = QCheckBox("Edit Stitch Markers (Drag / Shift+Click / Right-Click)")
        self.edit_markers_chk.toggled.connect(self._toggle_edit_markers)

        self.export_btn = QPushButton("Export .DST")
        self.export_btn.clicked.connect(self._export_dst)

        self.status_label = QLabel()

        controls_layout.addWidget(QLabel("Fill Mode:"))
        controls_layout.addWidget(self.mode_selector)
        controls_layout.addWidget(self.edit_markers_chk)
        controls_layout.addWidget(self.export_btn)
        controls_layout.addStretch()
        controls_layout.addWidget(self.status_label)

        main_layout.addLayout(controls_layout)

        # Add our custom viewport
        self.viewport = StitchViewport(on_stitches_changed=self._on_manual_stitch_edit)
        main_layout.addWidget(self.viewport)

        if self.current_stitches:
            self._update_viewport(self.current_stitches)
        else:
            self._regenerate_from_mode()

    def _toggle_edit_markers(self, enabled: bool) -> None:
        self.viewport.set_edit_mode(enabled)

    def _on_manual_stitch_edit(self, updated_stitches: list[Point2D]) -> None:
        self.current_stitches = list(updated_stitches)
        self.status_label.setText(f"Stitches: {len(self.current_stitches)}")

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

        preview_dialog = ExportPreviewDialog(self, self.current_stitches)
        if preview_dialog.exec() != QDialog.DialogCode.Accepted:
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
