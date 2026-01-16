import numpy as np
from typing import List

# =========================
# Segment assignment (pixel-exact)
# =========================

def assign_segments_vertical(mask: np.ndarray, targets: List[int], direction: str = "left_to_right") -> np.ndarray:
    """
    Assigns segments by filling columns.
    Optimized to use vectorized operations where possible.
    """
    h, w = mask.shape
    seg_map = np.full((h, w), -1, dtype=np.int32)

    # Identify which pixels are in the mask
    # We want to fill columns.
    # Let's get indices of all mask pixels, sorted by column then row.
    # "left_to_right": sort by x (asc), then y.
    # "right_to_left": sort by x (desc), then y.

    ys, xs = np.where(mask)

    if direction == "left_to_right":
        # lexsort((primary, secondary)) -> we want x primary, y secondary
        # But lexsort takes (secondary, primary).
        # So lexsort((ys, xs)) sorts by xs (primary), then ys (secondary).
        order = np.lexsort((ys, xs))
    else: # right_to_right -> largest x first
        # We can sort by -xs
        order = np.lexsort((ys, -xs))

    # Now we have the order of pixels to fill
    sorted_ys = ys[order]
    sorted_xs = xs[order]

    _fill_segments_by_order(seg_map, sorted_ys, sorted_xs, targets)

    # Handle unassigned (should be none if targets sum to total)
    # But just in case:
    unassigned = mask & (seg_map < 0)
    if unassigned.any():
        seg_map[unassigned] = len(targets) - 1

    return seg_map

def assign_segments_horizontal(mask: np.ndarray, targets: List[int], direction: str = "top_to_bottom") -> np.ndarray:
    """
    Assigns segments by filling rows.
    """
    h, w = mask.shape
    seg_map = np.full((h, w), -1, dtype=np.int32)

    ys, xs = np.where(mask)

    if direction == "top_to_bottom":
        # Sort by y (primary), then x (secondary)
        # lexsort((x, y)) -> y primary, x secondary
        order = np.lexsort((xs, ys))
    else: # bottom_to_top
        order = np.lexsort((xs, -ys))

    sorted_ys = ys[order]
    sorted_xs = xs[order]

    _fill_segments_by_order(seg_map, sorted_ys, sorted_xs, targets)

    unassigned = mask & (seg_map < 0)
    if unassigned.any():
        seg_map[unassigned] = len(targets) - 1

    return seg_map

def assign_segments_angular(mask: np.ndarray, targets: List[int]) -> np.ndarray:
    """
    Assigns segments by angle from center.
    Default Center: Centroid of the mask pixels.
    Sorts pixels by angle, then radius.
    """
    ys, xs = np.where(mask)
    coords = np.stack([ys, xs], axis=1).astype(np.float32)

    if len(coords) == 0:
        return np.full(mask.shape, -1, dtype=np.int32)

    cy, cx = coords.mean(axis=0)

    dy = ys.astype(np.float32) - cy
    dx = xs.astype(np.float32) - cx

    # Angle in [-pi, pi]
    ang = np.arctan2(dy, dx)
    # Normalize to [0, 2pi]
    ang = (ang + 2 * np.pi) % (2 * np.pi)

    # Radius
    r = np.sqrt(dx * dx + dy * dy)

    # Sort: Angle primary, Radius secondary
    # lexsort((secondary, primary))
    order = np.lexsort((r, ang))

    sorted_ys = ys[order]
    sorted_xs = xs[order]

    seg_map = np.full(mask.shape, -1, dtype=np.int32)
    _fill_segments_by_order(seg_map, sorted_ys, sorted_xs, targets)

    unassigned = mask & (seg_map < 0)
    if unassigned.any():
        seg_map[unassigned] = len(targets) - 1

    return seg_map

def assign_segments_contiguous_wedge(mask: np.ndarray, targets: List[int]) -> np.ndarray:
    """
    Similar to Angular, but tries to minimize noise by perhaps smoothing coordinates
    or using a different center?

    Actually, 'Contiguous Wedges' (Radial Sweep) is effectively the Angular sort.
    To make it 'cleaner' (less stained), we can ensure that we strictly sweep.
    The staining comes from pixels having very close angles but alternating due to grid aliasing.

    We can try to quantize the angle? No, that loses precision.

    Let's try using the Bounding Box center instead of Centroid?
    Sometimes Centroid is better.

    Let's implement a version that allows the user to pick a starting angle offset?
    Or maybe simply ensures we handle the 'seam' where 0/360 meet gracefully?

    For now, I will implement it identical to Angular but maybe exposed with a different name
    or slightly different parameterization (e.g. start from -pi/2 (top) instead of 0 (right)).
    Standard math 0 is Right (East). Top is -pi/2 or 3pi/2.
    Let's make "Contiguous Wedge" start from Top (North) which is often preferred in charts (Pie charts).
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
         return np.full(mask.shape, -1, dtype=np.int32)

    coords = np.stack([ys, xs], axis=1).astype(np.float32)
    cy, cx = coords.mean(axis=0)

    dy = ys.astype(np.float32) - cy
    dx = xs.astype(np.float32) - cx

    # Pie chart convention: 0 at Top (North), clockwise.
    # Math: 0 at Right (East), counter-clockwise.
    # atan2(dy, dx) -> y is down (positive), x is right.
    # So dy>0 is Down. dx>0 is Right.
    # atan2(1, 0) -> pi/2 (Down). atan2(-1, 0) -> -pi/2 (Up).

    # We want Top (Up) to be 0.
    # Up is dy < 0.
    # Let's map (dy, dx) -> angle such that Up is 0, Right is 90...
    # angle = atan2(dx, -dy) ?
    # if dy=-1, dx=0 (Up) -> atan2(0, 1) = 0.
    # if dy=0, dx=1 (Right) -> atan2(1, 0) = pi/2.
    # if dy=1, dx=0 (Down) -> atan2(0, -1) = pi.
    # if dy=0, dx=-1 (Left) -> atan2(-1, 0) = -pi/2.

    ang = np.arctan2(dx, -dy)
    ang = (ang + 2 * np.pi) % (2 * np.pi)

    r = np.sqrt(dx * dx + dy * dy)

    order = np.lexsort((r, ang))

    sorted_ys = ys[order]
    sorted_xs = xs[order]

    seg_map = np.full(mask.shape, -1, dtype=np.int32)
    _fill_segments_by_order(seg_map, sorted_ys, sorted_xs, targets)

    unassigned = mask & (seg_map < 0)
    if unassigned.any():
        seg_map[unassigned] = len(targets) - 1

    return seg_map


def _fill_segments_by_order(seg_map: np.ndarray, sorted_ys: np.ndarray, sorted_xs: np.ndarray, targets: List[int]):
    """
    Helper to fill pixels according to a sorted order and targets list.
    """
    start = 0
    # Process all segments
    for seg_idx, t in enumerate(targets):
        if t <= 0:
            continue
        end = start + t

        # Safe bounds check
        if end > len(sorted_ys):
            end = len(sorted_ys)

        # Get indices for this segment
        ys_slice = sorted_ys[start:end]
        xs_slice = sorted_xs[start:end]

        seg_map[ys_slice, xs_slice] = seg_idx

        start = end
        if start >= len(sorted_ys):
            break
