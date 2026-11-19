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


def extract_color_layer_masks(
    image_bgr: np.ndarray,
    num_colors: int = 4,
    bg_threshold: int = 235,
    min_pixels: int = 20,
) -> List[Tuple[Tuple[int, int, int], np.ndarray, bool]]:
    """
    Separate an image into distinct quantized color layer binary masks (e.g., separating dark outlines
    from interior fill colors). Returns a list of (rgb_tuple, binary_mask, is_dark_outline) ordered so
    interior fill colors come FIRST and dark outlines come LAST.
    """
    if image_bgr.ndim == 2:
        binary = preprocess_binary_mask(image_bgr, threshold=165)
        return [((33, 37, 41), binary, True)]

    fg_mask = preprocess_binary_mask(image_bgr, threshold=bg_threshold, invert=True, morph_kernel_size=3)
    if cv2.countNonZero(fg_mask) < min_pixels:
        return []

    quantized_bgr, _ = quantize_image_colors(image_bgr, num_colors=max(2, num_colors))
    fg_pixels = quantized_bgr[fg_mask > 0]
    unique_bgr, counts = np.unique(fg_pixels, axis=0, return_counts=True)

    layers: List[Tuple[Tuple[int, int, int], np.ndarray, bool]] = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for bgr_val, count in zip(unique_bgr, counts):
        if int(count) < min_pixels:
            continue
        b_c, g_c, r_c = int(bgr_val[0]), int(bgr_val[1]), int(bgr_val[2])
        if r_c > bg_threshold and g_c > bg_threshold and b_c > bg_threshold:
            continue
        exact = np.all(quantized_bgr == bgr_val, axis=-1).astype(np.uint8) * 255
        layer_mask = cv2.bitwise_and(exact, fg_mask)
        layer_mask = cv2.morphologyEx(layer_mask, cv2.MORPH_CLOSE, kernel)
        if cv2.countNonZero(layer_mask) < min_pixels:
            continue
        is_dark = (r_c < 85 and g_c < 85 and b_c < 90)
        layers.append(((r_c, g_c, b_c), layer_mask, is_dark))

    layers.sort(key=lambda item: (1 if item[2] else 0, item[0]))
    return layers


def extract_medial_skeleton_paths(
    binary_mask: np.ndarray,
    pixel_scale_mm: float = 0.25,
    min_chain_pixels: int = 4,
) -> List[List[Tuple[float, float, float]]]:
    """
    Extract the medial-axis centerline skeleton and local stroke width profile (via distance transform)
    from a binary mask using morphological thinning.
    Returns a list of chains, where each point is (x_mm, y_mm, width_mm).
    """
    mask_u8 = (binary_mask > 0).astype(np.uint8)
    if np.count_nonzero(mask_u8) < min_chain_pixels:
        return []

    dist_px = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)

    skel = np.zeros_like(mask_u8)
    temp_img = mask_u8.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    guard = 0
    while guard < 64:
        guard += 1
        eroded = cv2.erode(temp_img, element)
        opened = cv2.dilate(eroded, element)
        sub = cv2.subtract(temp_img, opened)
        skel = cv2.bitwise_or(skel, sub)
        temp_img = eroded
        if cv2.countNonZero(temp_img) == 0:
            break

    ys, xs = np.where(skel > 0)
    if len(xs) < min_chain_pixels:
        return []

    h, w = skel.shape
    visited = np.zeros_like(skel, dtype=bool)
    chains: List[List[Tuple[float, float, float]]] = []

    coords = list(zip(xs.tolist(), ys.tolist()))

    def _degree(pt: Tuple[int, int]) -> int:
        px, py = pt
        y0, y1 = max(0, py - 1), min(h, py + 2)
        x0, x1 = max(0, px - 1), min(w, px + 2)
        return int(np.count_nonzero(skel[y0:y1, x0:x1])) - 1

    coords.sort(key=lambda pt: 0 if _degree(pt) == 1 else (1 if _degree(pt) >= 3 else 2))
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]

    for sx, sy in coords:
        if visited[sy, sx]:
            continue
        chain: List[Tuple[float, float, float]] = []
        cx, cy = sx, sy
        while True:
            visited[cy, cx] = True
            radius_mm = max(0.4, float(dist_px[cy, cx]) * pixel_scale_mm)
            width_mm = min(8.5, max(0.8, radius_mm * 2.1))
            chain.append(((cx + 0.5) * pixel_scale_mm, (cy + 0.5) * pixel_scale_mm, width_mm))

            found_next = False
            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and skel[ny, nx] > 0 and not visited[ny, nx]:
                    cx, cy = nx, ny
                    found_next = True
                    break
            if not found_next:
                break

        if len(chain) >= min_chain_pixels:
            chains.append(chain)

    return chains


