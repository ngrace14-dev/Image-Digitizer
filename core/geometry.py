"""Geometric utilities and unit conversion helpers for embroidery digitizing."""

import math
from typing import List, Tuple, Sequence
from shapely.geometry import Polygon, MultiPolygon, LineString

Point2D = Tuple[float, float]

# 1 embroidery machine unit (especially in Tajima DST) equals 0.1 mm
MM_TO_MACHINE_UNIT = 10.0
MACHINE_UNIT_TO_MM = 0.1


def mm_to_units(val_mm: float) -> int:
    """Convert millimeters to integer machine units (0.1 mm increments)."""
    return int(round(val_mm * MM_TO_MACHINE_UNIT))


def units_to_mm(units: float) -> float:
    """Convert machine units (0.1 mm increments) to millimeters."""
    return float(units) * MACHINE_UNIT_TO_MM


def calculate_distance(p1: Point2D, p2: Point2D) -> float:
    """Calculate Euclidean distance between two 2D points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def scale_points(points: Sequence[Point2D], scale_factor: float) -> List[Point2D]:
    """Scale a sequence of (x, y) coordinates by a constant factor."""
    return [(x * scale_factor, y * scale_factor) for x, y in points]


def ensure_valid_polygon(coords: Sequence[Point2D]) -> Polygon:
    """Construct a valid Shapely Polygon from coordinates, repairing self-intersections."""
    poly = Polygon(coords)
    if not poly.is_valid:
        repaired = poly.buffer(0)
        if isinstance(repaired, MultiPolygon):
            poly = max(repaired.geoms, key=lambda p: p.area)
        elif isinstance(repaired, Polygon):
            poly = repaired
    return poly


def inset_polygon(poly: Polygon, distance: float) -> List[Polygon]:
    """
    Shrink (negative distance) or expand (positive distance) a polygon.
    Returns a list of valid resulting Polygons.
    """
    buffered = poly.buffer(-abs(distance))
    if buffered.is_empty:
        return []
    if isinstance(buffered, Polygon):
        return [buffered]
    if isinstance(buffered, MultiPolygon):
        return [g for g in buffered.geoms if isinstance(g, Polygon) and not g.is_empty]
    return []


def interpolate_line(p1: Point2D, p2: Point2D, step: float) -> List[Point2D]:
    """Interpolate intermediate needle drop points along a segment from p1 to p2."""
    dist = calculate_distance(p1, p2)
    if dist <= step or step <= 0:
        return [p1, p2]
    line = LineString([p1, p2])
    num_steps = int(math.ceil(dist / step))
    return [
        (
            line.interpolate(i / num_steps, normalized=True).x,
            line.interpolate(i / num_steps, normalized=True).y,
        )
        for i in range(num_steps + 1)
    ]
