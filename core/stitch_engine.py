"""Core stitch generation algorithms for Running, Concentric Fill, and Tatami Fill stitches."""

import math
from typing import List, Sequence
import cv2
from shapely.affinity import rotate
from shapely.geometry import LineString, MultiLineString

from core.geometry import (
    Point2D,
    calculate_distance,
    ensure_valid_polygon,
)


def sample_path(coords: Sequence[Point2D], stitch_length: float) -> List[Point2D]:
    """Breaks a geometric line into individual needle penetrations."""
    if not coords:
        return []

    stitches: List[Point2D] = [coords[0]]
    current_pt = coords[0]

    for next_pt in coords[1:]:
        dist = math.hypot(next_pt[0] - current_pt[0], next_pt[1] - current_pt[1])
        if dist >= stitch_length:
            stitches.append(next_pt)
            current_pt = next_pt

    return stitches


def generate_running_stitch(image_path: str, stitch_length: float = 10.0) -> List[Point2D]:
    """
    Load a grayscale image, threshold to binary, find the largest outer contour,
    and sample points along that path to simulate a running stitch.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    largest_contour = max(contours, key=cv2.contourArea)
    points: List[Point2D] = [(float(pt[0][0]), float(pt[0][1])) for pt in largest_contour]

    stitches = [points[0]]
    current_point = points[0]

    for point in points[1:]:
        dist = calculate_distance(current_point, point)
        if dist >= stitch_length:
            stitches.append(point)
            current_point = point

    stitches.append(stitches[0])
    return stitches


def generate_concentric_fill(
    pixel_boundary: Sequence[Point2D],
    stitch_length_mm: float = 2.5,
    pixel_scale_mm: float = 0.25,
) -> List[Point2D]:
    """
    Takes a pixel boundary, scales it to physical dimensions, and generates
    a concentric fill pattern using negative buffering.
    """
    if len(pixel_boundary) < 3:
        return []

    physical_coords = [(x * pixel_scale_mm, y * pixel_scale_mm) for x, y in pixel_boundary]
    poly = ensure_valid_polygon(physical_coords)

    all_stitches: List[Point2D] = []
    current_poly = poly
    row_spacing = stitch_length_mm * 0.8

    while current_poly.area > 0.1:
        ring_coords = [(float(x), float(y)) for x, y in current_poly.exterior.coords]
        sampled_ring = sample_path(ring_coords, stitch_length_mm)
        all_stitches.extend(sampled_ring)

        current_poly = current_poly.buffer(-row_spacing)

        if current_poly.geom_type == "MultiPolygon":
            current_poly = max(current_poly.geoms, key=lambda p: p.area)
        elif current_poly.geom_type != "Polygon":
            break

    return all_stitches


def generate_tatami_fill(
    pixel_boundary: Sequence[Point2D],
    stitch_length_mm: float = 3.5,
    row_spacing_mm: float = 0.4,
    angle_deg: float = 0.0,
    pixel_scale_mm: float = 0.25,
) -> List[Point2D]:
    """
    Generates a directional parallel Tatami fill across a polygon boundary
    with staggered needle penetrations along alternating scanlines.
    """
    if len(pixel_boundary) < 3:
        return []

    physical_coords = [(x * pixel_scale_mm, y * pixel_scale_mm) for x, y in pixel_boundary]
    poly = ensure_valid_polygon(physical_coords)
    if poly.is_empty:
        return []

    origin = poly.centroid
    rotated_poly = rotate(poly, -angle_deg, origin=origin) if angle_deg != 0.0 else poly

    min_x, min_y, max_x, max_y = rotated_poly.bounds
    stitches_rotated: List[Point2D] = []

    y = min_y + row_spacing_mm * 0.5
    row_index = 0

    while y <= max_y:
        scanline = LineString([(min_x - 1.0, y), (max_x + 1.0, y)])
        intersection = rotated_poly.intersection(scanline)

        segments: List[LineString] = []
        if isinstance(intersection, LineString) and not intersection.is_empty:
            segments = [intersection]
        elif isinstance(intersection, MultiLineString):
            segments = [seg for seg in intersection.geoms if not seg.is_empty]

        for seg in segments:
            coords = list(seg.coords)
            if len(coords) < 2:
                continue
            x_start, y_pos = coords[0]
            x_end, _ = coords[-1]
            if x_start > x_end:
                x_start, x_end = x_end, x_start

            stagger = (row_index % 3) * (stitch_length_mm / 3.0)
            row_pts: List[Point2D] = [(x_start, y_pos)]
            x_curr = x_start + (stitch_length_mm - stagger)
            while x_curr < x_end:
                row_pts.append((x_curr, y_pos))
                x_curr += stitch_length_mm
            if row_pts[-1][0] != x_end:
                row_pts.append((x_end, y_pos))

            if row_index % 2 == 1:
                row_pts.reverse()

            stitches_rotated.extend(row_pts)

        y += row_spacing_mm
        row_index += 1

    if not stitches_rotated:
        return []

    if angle_deg != 0.0:
        path_line = LineString(stitches_rotated)
        restored_line = rotate(path_line, angle_deg, origin=origin)
        return [(float(x), float(y)) for x, y in restored_line.coords]

    return [(float(x), float(y)) for x, y in stitches_rotated]
