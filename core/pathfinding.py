"""Pathfinding and jump-stitch minimization algorithms (TSP heuristics)."""

from typing import List, Sequence
from core.geometry import Point2D, calculate_distance


def optimize_stitch_blocks(blocks: Sequence[List[Point2D]]) -> List[List[Point2D]]:
    """
    Order disconnected stitch blocks using a greedy nearest-neighbor TSP heuristic.
    Also reverses individual blocks if starting from their end point reduces travel distance.
    """
    valid_blocks = [list(b) for b in blocks if b]
    if len(valid_blocks) <= 1:
        return valid_blocks

    ordered: List[List[Point2D]] = [valid_blocks.pop(0)]

    while valid_blocks:
        current_end = ordered[-1][-1]
        best_idx = 0
        best_dist = float("inf")
        reverse_best = False

        for idx, candidate in enumerate(valid_blocks):
            dist_start = calculate_distance(current_end, candidate[0])
            dist_end = calculate_distance(current_end, candidate[-1])

            if dist_start < best_dist:
                best_dist = dist_start
                best_idx = idx
                reverse_best = False

            if dist_end < best_dist:
                best_dist = dist_end
                best_idx = idx
                reverse_best = True

        chosen = valid_blocks.pop(best_idx)
        if reverse_best:
            chosen = list(reversed(chosen))
        ordered.append(chosen)

    return ordered
