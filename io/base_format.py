"""Abstract base class for embroidery file formats."""

from abc import ABC, abstractmethod
from typing import List, Tuple

Point2D = Tuple[float, float]


class BaseEmbroideryFormat(ABC):
    """Interface for reading and writing embroidery stitch files."""

    @abstractmethod
    def write(self, filepath: str, stitches_mm: List[Point2D]) -> None:
        """Write a sequence of (x_mm, y_mm) stitch coordinates to disk."""
        raise NotImplementedError

    @abstractmethod
    def read(self, filepath: str) -> List[Point2D]:
        """Read stitch coordinates from disk and return them in millimeters."""
        raise NotImplementedError
