"""Core stitch generation algorithms for Running, Concentric Fill, and Tatami Fill stitches."""

import math
from typing import List, Sequence, Tuple, Union
import cv2
from shapely.affinity import rotate
from shapely.geometry import LineString, MultiLineString, Polygon

from core.geometry import (
    Point2D,
    calculate_distance,
    ensure_valid_polygon,
)
from core.pathfinding import optimize_stitch_blocks


def sample_path(coords: Sequence[Point2D], stitch_length: float) -> List[Point2D]:
    """Breaks a geometric line into individual needle penetrations, preserving corners and endpoints."""
    if not coords:
        return []

    stitches: List[Point2D] = [coords[0]]
    current_pt = coords[0]

    for next_pt in coords[1:]:
        dist = math.hypot(next_pt[0] - current_pt[0], next_pt[1] - current_pt[1])
        if dist >= stitch_length:
            steps = max(1, int(round(dist / stitch_length)))
            for s in range(1, steps + 1):
                t = s / steps
                interp = (
                    current_pt[0] + (next_pt[0] - current_pt[0]) * t,
                    current_pt[1] + (next_pt[1] - current_pt[1]) * t,
                )
                stitches.append(interp)
            current_pt = next_pt

    if stitches[-1] != coords[-1]:
        stitches.append(coords[-1])

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
    pixel_boundary: Union[Sequence[Point2D], Polygon],
    stitch_length_mm: float = 2.5,
    pixel_scale_mm: float = 0.25,
) -> List[Point2D]:
    """
    Takes a pixel boundary or Shapely Polygon, scales it to physical dimensions,
    and generates a concentric fill pattern using negative buffering across all sub-islands.
    """
    if isinstance(pixel_boundary, Polygon):
        poly = pixel_boundary
    else:
        if len(pixel_boundary) < 3:
            return []
        physical_coords = [(x * pixel_scale_mm, y * pixel_scale_mm) for x, y in pixel_boundary]
        poly = ensure_valid_polygon(physical_coords)

    all_stitches: List[Point2D] = []
    row_spacing = stitch_length_mm * 0.8

    queue: List[Polygon] = [poly]
    while queue:
        current_poly = queue.pop(0)
        if current_poly.is_empty or current_poly.area <= 0.1:
            continue

        ring_coords = [(float(x), float(y)) for x, y in current_poly.exterior.coords]
        all_stitches.extend(sample_path(ring_coords, stitch_length_mm))

        for interior in current_poly.interiors:
            hole_coords = [(float(x), float(y)) for x, y in interior.coords]
            all_stitches.extend(sample_path(hole_coords, stitch_length_mm))

        eroded = current_poly.buffer(-row_spacing)
        if eroded.is_empty:
            continue
        if eroded.geom_type == "Polygon":
            queue.insert(0, eroded)
        elif eroded.geom_type == "MultiPolygon":
            sub_polys = sorted(eroded.geoms, key=lambda p: p.area, reverse=True)
            for sp in reversed(sub_polys):
                queue.insert(0, sp)

    return all_stitches


def generate_tatami_fill(
    pixel_boundary: Union[Sequence[Point2D], Polygon],
    stitch_length_mm: float = 3.5,
    row_spacing_mm: float = 0.4,
    angle_deg: float = 0.0,
    pixel_scale_mm: float = 0.25,
) -> List[Point2D]:
    """
    Generates a directional parallel Tatami fill across a polygon boundary (including holes).
    Uses scanline run-chaining so shapes that split into branchesstitch each branch contiguously.
    """
    if isinstance(pixel_boundary, Polygon):
        poly = pixel_boundary
    else:
        if len(pixel_boundary) < 3:
            return []
        physical_coords = [(x * pixel_scale_mm, y * pixel_scale_mm) for x, y in pixel_boundary]
        poly = ensure_valid_polygon(physical_coords)

    if poly.is_empty:
        return []

    origin = poly.centroid
    rotated_poly = rotate(poly, -angle_deg, origin=origin) if angle_deg != 0.0 else poly

    min_x, min_y, max_x, max_y = rotated_poly.bounds
    chains: List[List[Tuple[int, float, float, float]]] = []
    active_chains: List[Tuple[int, float, float, float]] = []

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

        current_runs: List[Tuple[float, float]] = []
        for seg in segments:
            coords = list(seg.coords)
            if len(coords) < 2:
                continue
            x_start, _ = coords[0]
            x_end, _ = coords[-1]
            if x_start > x_end:
                x_start, x_end = x_end, x_start
            if x_end - x_start >= 0.15:
                current_runs.append((x_start, x_end))

        current_runs.sort(key=lambda r: r[0])
        used_active = set()
        next_active: List[Tuple[int, float, float, float]] = []

        for x_start, x_end in current_runs:
            best_chain_idx = -1
            best_overlap = -1.0
            for a_idx, (c_idx, ax0, ax1, _ay) in enumerate(active_chains):
                if a_idx in used_active:
                    continue
                overlap = min(x_end, ax1) - max(x_start, ax0)
                if overlap >= -row_spacing_mm * 1.5 and overlap > best_overlap:
                    best_overlap = overlap
                    best_chain_idx = a_idx

            if best_chain_idx != -1:
                used_active.add(best_chain_idx)
                c_idx = active_chains[best_chain_idx][0]
                chains[c_idx].append((row_index, x_start, x_end, y))
                next_active.append((c_idx, x_start, x_end, y))
            else:
                new_c_idx = len(chains)
                chains.append([(row_index, x_start, x_end, y)])
                next_active.append((new_c_idx, x_start, x_end, y))

        active_chains = next_active
        y += row_spacing_mm
        row_index += 1

    if not chains:
        return []

    blocks_rotated: List[List[Point2D]] = []
    for chain in chains:
        block: List[Point2D] = []
        for local_idx, (r_idx, x_start, x_end, y_pos) in enumerate(chain):
            stagger = (r_idx % 3) * (stitch_length_mm / 3.0)
            row_pts: List[Point2D] = [(x_start, y_pos)]
            x_curr = x_start + (stitch_length_mm - stagger)
            while x_curr < x_end:
                row_pts.append((x_curr, y_pos))
                x_curr += stitch_length_mm
            if row_pts[-1][0] != x_end:
                row_pts.append((x_end, y_pos))

            if local_idx % 2 == 1:
                row_pts.reverse()
            block.extend(row_pts)
        if block:
            blocks_rotated.append(block)

    ordered_blocks = optimize_stitch_blocks(blocks_rotated)
    stitches_rotated: List[Point2D] = [pt for blk in ordered_blocks for pt in blk]
    if not stitches_rotated:
        return []

    if angle_deg != 0.0:
        path_line = LineString(stitches_rotated)
        restored_line = rotate(path_line, angle_deg, origin=origin)
        return [(float(x), float(y)) for x, y in restored_line.coords]

    return [(float(x), float(y)) for x, y in stitches_rotated]


def generate_multi_polygon_tatami(
    polygons_mm: Sequence[Polygon],
    stitch_length_mm: float = 3.5,
    row_spacing_mm: float = 0.4,
    angle_deg: float = 0.0,
    include_outline: bool = True,
) -> List[Point2D]:
    """
    Fill each connected component Polygon (including interior holes) individually,
    optionally add a crisp edge-defining running stitch contour, and optimize the
    component sewing sequence via TSP pathfinding.
    """
    component_blocks: List[List[Point2D]] = []
    for poly in polygons_mm:
        if poly.is_empty:
            continue
        fill_pts = generate_tatami_fill(
            poly,
            stitch_length_mm=stitch_length_mm,
            row_spacing_mm=row_spacing_mm,
            angle_deg=angle_deg,
            pixel_scale_mm=1.0,
        )
        if include_outline:
            ext_coords = [(float(x), float(y)) for x, y in poly.exterior.coords]
            outline_pts = sample_path(ext_coords, min(2.0, stitch_length_mm * 0.7))
            for hole in poly.interiors:
                hole_coords = [(float(x), float(y)) for x, y in hole.coords]
                outline_pts.extend(sample_path(hole_coords, min(2.0, stitch_length_mm * 0.7)))
            fill_pts.extend(outline_pts)

        if fill_pts:
            component_blocks.append(fill_pts)

    ordered = optimize_stitch_blocks(component_blocks)
    return [pt for blk in ordered for pt in blk]
