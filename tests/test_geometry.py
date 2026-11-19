"""Unit tests for core geometry, stitch engines, pathfinding, and DST binary export."""

import os
import tempfile
import numpy as np
from core.geometry import mm_to_units, units_to_mm, calculate_distance, ensure_valid_polygon, inset_polygon
from core.pathfinding import optimize_stitch_blocks
from core.stitch_engine import (
    sample_path,
    generate_concentric_fill,
    generate_tatami_fill,
    generate_multi_polygon_tatami,
)
from vision.vectorizer import preprocess_binary_mask, extract_hierarchical_polygons
from io.dst_writer import write_dst


def test_unit_conversions() -> None:
    assert mm_to_units(1.0) == 10
    assert mm_to_units(2.5) == 25
    assert abs(units_to_mm(25) - 2.5) < 1e-6


def test_polygon_insetting() -> None:
    square = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
    poly = ensure_valid_polygon(square)
    assert abs(poly.area - 400.0) < 1e-6

    inset_list = inset_polygon(poly, 2.0)
    assert len(inset_list) == 1
    # Shrinking a 20x20 square by 2mm on all sides yields a 16x16 square (area 256)
    assert abs(inset_list[0].area - 256.0) < 1.0


def test_concentric_fill_generation() -> None:
    mock_opencv_contour = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    stitches = generate_concentric_fill(mock_opencv_contour, stitch_length_mm=2.5, pixel_scale_mm=0.25)
    assert len(stitches) > 10
    for x, y in stitches:
        assert 0.0 <= x <= 25.0
        assert 0.0 <= y <= 25.0


def test_tatami_fill_generation() -> None:
    mock_opencv_contour = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    stitches = generate_tatami_fill(
        mock_opencv_contour,
        stitch_length_mm=3.5,
        row_spacing_mm=1.0,
        angle_deg=0.0,
        pixel_scale_mm=0.25,
    )
    assert len(stitches) > 20


def test_hierarchical_extraction_and_multi_polygon_tatami() -> None:
    # Create a synthetic image with two letters: an 'O' (outer box with interior hole) and an 'I' (solid bar)
    canvas = np.full((120, 200, 3), 255, dtype=np.uint8)
    # Letter 'O' in red (BGR: 0, 0, 230)
    canvas[20:100, 20:90] = (0, 0, 230)
    canvas[40:80, 40:70] = (255, 255, 255)  # Interior hole
    # Letter 'I' in black (BGR: 10, 10, 10)
    canvas[20:100, 130:160] = (10, 10, 10)

    binary = preprocess_binary_mask(canvas, threshold=165)
    polygons = extract_hierarchical_polygons(binary, min_area=20.0, pixel_scale_mm=0.25)
    assert len(polygons) == 2
    # Verify the 'O' polygon captured its interior hole
    holes_counts = sorted(len(p.interiors) for p in polygons)
    assert holes_counts == [0, 1]

    stitches = generate_multi_polygon_tatami(
        polygons,
        stitch_length_mm=3.0,
        row_spacing_mm=0.8,
        angle_deg=30.0,
        include_outline=True,
    )
    assert len(stitches) > 40
    # Verify only a tiny number of long jumps (> 7 mm) occur (no global scanline ping-ponging across letters)
    long_jumps = sum(
        1
        for i in range(1, len(stitches))
        if calculate_distance(stitches[i - 1], stitches[i]) > 7.0
    )
    assert long_jumps <= 4


def test_pathfinding_optimization() -> None:
    block_a = [(0.0, 0.0), (5.0, 0.0)]
    block_b = [(100.0, 100.0), (105.0, 100.0)]
    block_c = [(6.0, 0.0), (10.0, 0.0)]
    ordered = optimize_stitch_blocks([block_a, block_b, block_c])
    assert ordered[0] == block_a
    assert ordered[1] == block_c
    assert ordered[2] == block_b


def test_dst_writer() -> None:
    stitches = [(0.0, 0.0), (2.5, 0.0), (2.5, 2.5), (0.0, 2.5), (0.0, 0.0)]
    with tempfile.TemporaryDirectory() as tmpdir:
        dst_path = os.path.join(tmpdir, "sample.dst")
        write_dst(dst_path, stitches, label="TEST")
        assert os.path.exists(dst_path)
        with open(dst_path, "rb") as f:
            data = f.read()
        # 512-byte header + 5 stitch records (3 bytes each) + 1 EOF record (3 bytes) = 530 bytes
        assert len(data) == 512 + (len(stitches) + 1) * 3
        assert data[-3:] == bytes([0x00, 0x00, 0xF3])

