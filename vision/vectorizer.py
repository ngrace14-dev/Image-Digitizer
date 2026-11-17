"""OpenCV pipelines for color quantization and bitmap-to-polygon contour extraction."""

from typing import List, Tuple
import cv2
import numpy as np
from core.geometry import Point2D


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
    min_area: float = 25.0,
    epsilon_factor: float = 0.002,
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
