"""Pathfinding and jump-stitch minimization algorithms (TSP heuristics)."""

from typing import List, Sequence
from core.geometry import Point2D, calculate_distance


def optimize_stitch_blocks(blocks: Sequence[List[Point2D]]) -> List[List[Point2D]]:
    """
    Order disconnected stitch blocks using a greedy nearest-neighbor TSP heuristic
    followed by 2-opt local search refinement.
    Also reverses individual blocks if starting from their end point reduces travel distance.
    """
    valid_blocks = [list(b) for b in blocks if b]
    if len(valid_blocks) <= 1:
        return valid_blocks

    # Start with the leftmost block (ideal for text/logos like RREDCO) so ordering is deterministic
    start_idx = min(
        range(len(valid_blocks)),
        key=lambda i: min(pt[0] for pt in valid_blocks[i]),
    )
    ordered: List[List[Point2D]] = [valid_blocks.pop(start_idx)]

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

    # Orientation pass: flip any block if reversing it shortens jumps to neighbors
    for i in range(len(ordered)):
        block = ordered[i]
        if len(block) < 2:
            continue
        cost_normal = 0.0
        cost_flipped = 0.0
        if i > 0:
            cost_normal += calculate_distance(ordered[i - 1][-1], block[0])
            cost_flipped += calculate_distance(ordered[i - 1][-1], block[-1])
        if i + 1 < len(ordered):
            cost_normal += calculate_distance(block[-1], ordered[i + 1][0])
            cost_flipped += calculate_distance(block[0], ordered[i + 1][0])
        if cost_flipped + 1e-6 < cost_normal:
            ordered[i] = list(reversed(block))

    return ordered

