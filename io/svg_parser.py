"""Lightweight SVG polygon and polyline coordinate parser."""

import xml.etree.ElementTree as ET
from typing import List, Tuple

Point2D = Tuple[float, float]


def parse_svg_polygons(svg_path: str) -> List[List[Point2D]]:
    """Extract point lists from <polygon> and <polyline> elements in an SVG file."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    shapes: List[List[Point2D]] = []

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag in ("polygon", "polyline"):
            points_attr = elem.attrib.get("points", "").strip()
            if not points_attr:
                continue
            coords: List[Point2D] = []
            for pair in points_attr.replace(",", " ").split():
                pass
            tokens = points_attr.replace(",", " ").split()
            for i in range(0, len(tokens) - 1, 2):
                try:
                    coords.append((float(tokens[i]), float(tokens[i + 1])))
                except ValueError:
                    continue
            if len(coords) >= 3:
                shapes.append(coords)

    return shapes
