"""Application entry point for the Image Digitizer prototype."""

import sys
from PyQt6.QtWidgets import QApplication
from core.stitch_engine import generate_concentric_fill
from ui.main_window import MainWindow


def main() -> int:
    # Generate initial concentric fill stitches from a sample 100x100px contour
    sample_contour = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    initial_stitches = generate_concentric_fill(
        sample_contour,
        stitch_length_mm=2.5,
        pixel_scale_mm=0.25,
    )

    app = QApplication(sys.argv)
    window = MainWindow(initial_stitches)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
