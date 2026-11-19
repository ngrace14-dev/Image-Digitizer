"""OpenCV pipelines for color quantization and bitmap-to-polygon contour extraction."""

from typing import List, Tuple
import cv2
import numpy as np
from shapely.geometry import Polygon
from core.geometry import Point2D, ensure_valid_polygon


def preprocess_binary_mask(
    image_or_mask: np.ndarray,
    threshold: int = 128,
    invert: bool = True,
    morph_kernel_size: int = 3,
) -> np.ndarray:
    """
    Convert an image or mask into a denoised binary mask using chroma-aware
    foreground detection (if 3-channel BGR) and morphological opening/closing.
    """
    if image_or_mask.ndim == 3:
        # Chroma-aware darkness against white background so saturated reds/oranges aren't lost
        b, g, r = cv2.split(image_or_mask.astype(np.float32))
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        min_channel = np.minimum(np.minimum(r, g), b)
        effective_gray = np.clip(0.55 * luma + 0.45 * min_channel, 0, 255).astype(np.uint8)
    else:
        effective_gray = image_or_mask

    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(effective_gray, threshold, 255, thresh_type)

    if morph_kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary


def quantize_image_colors(image: np.ndarray, num_colors: int = 6) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """
    Reduce an image's palette to `num_colors` using K-Means clustering
    (similar to Hatch's Prepare Artwork for Embroidery tool).
    Returns the quantized BGR image and a list of RGB palette tuples.
    """
    data = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        data,
        num_colors,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    centers_uint8 = np.uint8(centers)
    quantized = centers_uint8[labels.flatten()].reshape(image.shape)
    palette_rgb = [(int(c[2]), int(c[1]), int(c[0])) for c in centers_uint8]
    return quantized, palette_rgb


def extract_contours_from_binary(
    binary_mask: np.ndarray,
    min_area: float = 15.0,
    epsilon_factor: float = 0.0015,
) -> List[List[Point2D]]:
    """
    Extract external contours from a binary mask and simplify them into polygon vertex lists.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons: List[List[Point2D]] = []

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_factor * peri, True)
        pts: List[Point2D] = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
        if len(pts) >= 3:
            polygons.append(pts)

    return polygons


def extract_hierarchical_polygons(
    binary_mask: np.ndarray,
    min_area: float = 15.0,
    epsilon_factor: float = 0.0015,
    pixel_scale_mm: float = 1.0,
) -> List[Polygon]:
    """
    Extract connected-component polygons INCLUDING interior holes (e.g. inside 'R', 'D', 'O')
    using two-level CCOMP contour hierarchy.
    Returns valid Shapely Polygons scaled by `pixel_scale_mm`.
    """
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        return []

    hier = hierarchy[0]
    polygons: List[Polygon] = []

    for idx, cnt in enumerate(contours):
        # Parent < 0 means outer boundary in RETR_CCOMP
        if hier[idx][3] != -1:
            continue
        if cv2.contourArea(cnt) < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_factor * peri, True)
        shell: List[Point2D] = [
            (float(pt[0][0]) * pixel_scale_mm, float(pt[0][1]) * pixel_scale_mm)
            for pt in approx
        ]
        if len(shell) < 3:
            continue

        holes: List[List[Point2D]] = []
        child_idx = hier[idx][2]
        while child_idx != -1:
            child_cnt = contours[child_idx]
            if cv2.contourArea(child_cnt) >= min_area * 0.4:
                c_peri = cv2.arcLength(child_cnt, True)
                c_approx = cv2.approxPolyDP(child_cnt, epsilon_factor * c_peri, True)
                hole_pts: List[Point2D] = [
                    (float(pt[0][0]) * pixel_scale_mm, float(pt[0][1]) * pixel_scale_mm)
                    for pt in c_approx
                ]
                if len(hole_pts) >= 3:
                    holes.append(hole_pts)
            child_idx = hier[child_idx][0]

        poly = Polygon(shell, holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty:
            if poly.geom_type == "Polygon":
                polygons.append(poly)
            elif poly.geom_type == "MultiPolygon":
                polygons.extend([g for g in poly.geoms if g.area >= min_area * (pixel_scale_mm ** 2)])

    return polygons

