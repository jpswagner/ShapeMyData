import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Any

from mask_utils import (
    normalize_targets,
    hex_to_rgba,
    load_font,
    format_pct,
    upscale_mask_bool
)
from segmentation import (
    assign_segments_vertical,
    assign_segments_horizontal,
    assign_segments_angular,
    assign_segments_contiguous_wedge
)

# =========================
# Label layout
# =========================
def avoid_overlaps_bbox(items: List[dict], min_gap: int, y_min: int, y_max: int, h_canvas: int):
    """
    Adjusts y-coordinates of items to avoid overlap, considering their heights.
    items: list of dicts with 'y' (desired center or top), 'h' (height of text block).
    We'll treat 'y' as the vertical center of the text block for alignment.
    """
    if not items:
        return

    # Sort by original Y
    items.sort(key=lambda d: d["y"])

    # We maintain a list of occupied intervals?
    # Or just stack them?
    # "Force-directed" or simple stacking.
    # Simple stacking:
    # 1. Place first item at its Y or y_min.
    # 2. Place next item at max(its Y, prev_bottom + gap).

    # Pass 1: Push down
    # We use 'y_adj' to store the center Y.

    # Initialize y_adj
    for it in items:
        it["y_adj"] = it["y"]

    # Iterate
    # We need to know half-height to compute edges

    # Let's iterate and push down
    last_bottom = y_min

    for it in items:
        hh = it["h"] / 2
        top = it["y_adj"] - hh

        if top < last_bottom:
            # Push down
            new_top = last_bottom
            it["y_adj"] = new_top + hh

        last_bottom = it["y_adj"] + hh + min_gap

    # Pass 2: Push up if overflow
    last_top = y_max

    # If the last item went too low
    if items:
        last_item = items[-1]
        hh = last_item["h"] / 2
        bottom = last_item["y_adj"] + hh

        overflow = bottom - y_max
        if overflow > 0:
            # We need to shift everything up, but respect gaps
            shift = overflow
            # We can try to shift all up by 'shift'
            # But we must check if we hit y_min

            # Reverse iterate
            for i in range(len(items) - 1, -1, -1):
                it = items[i]
                it["y_adj"] -= shift

                # Check collision with previous (which is 'next' in reverse loop, but we mean the one above)
                # Actually, simpler: just push up from bottom

            # Re-run a rigid push up from bottom
            next_top = y_max
            for i in range(len(items) - 1, -1, -1):
                it = items[i]
                hh = it["h"] / 2
                bottom = it["y_adj"] + hh

                if bottom > next_top:
                    new_bottom = next_top
                    it["y_adj"] = new_bottom - hh

                next_top = it["y_adj"] - hh - min_gap

            # Now check if we pushed too far up (above y_min)
            if items[0]["y_adj"] - items[0]["h"]/2 < y_min:
                # We are squeezed. We can't do much but maybe overlap or clamp.
                pass


def render_shape_area_chart(
    mask_original: np.ndarray,
    labels: List[str],
    values: List[float],
    colors_hex: List[str],
    bg_hex: str,
    segmentation: str,
    direction: str,
    decimal_comma: bool,
    font_label_size: int,
    font_pct_size: int,
    margin_left: int,
    margin_right: int,
    margin_top: int,
    margin_bottom: int,
    upscale_factor: int = 1,
    show_labels: bool = True,
    show_percentages: bool = True,
    internal_labels: bool = False,
    elbow_lines: bool = True
) -> Image.Image:

    # 1. Upscale Mask
    if upscale_factor > 1:
        mask = upscale_mask_bool(mask_original, upscale_factor)
        # Scale layout params
        m_left = margin_left * upscale_factor
        m_right = margin_right * upscale_factor
        m_top = margin_top * upscale_factor
        m_bottom = margin_bottom * upscale_factor
        f_label_sz = font_label_size * upscale_factor
        f_pct_sz = font_pct_size * upscale_factor
    else:
        mask = mask_original
        m_left, m_right, m_top, m_bottom = margin_left, margin_right, margin_top, margin_bottom
        f_label_sz, f_pct_sz = font_label_size, font_pct_size

    h, w = mask.shape
    total_pixels = int(mask.sum())

    if total_pixels == 0:
        # Return a blank image
        bg = hex_to_rgba(bg_hex)
        out = Image.new("RGBA", (m_left + w + m_right, m_top + h + m_bottom), bg)
        return out

    targets = normalize_targets(values, total_pixels)
    n = len(labels)

    # 2. Segmentation
    if segmentation == "Vertical":
        seg_map = assign_segments_vertical(mask, targets, direction)
    elif segmentation == "Horizontal":
        seg_map = assign_segments_horizontal(mask, targets, direction)
    elif segmentation == "Angular":
        seg_map = assign_segments_angular(mask, targets)
    elif segmentation == "Contiguous Wedge": # New mode
        seg_map = assign_segments_contiguous_wedge(mask, targets)
    else: # Fallback
        seg_map = assign_segments_angular(mask, targets)

    colors = [hex_to_rgba(c) for c in colors_hex]
    bg = hex_to_rgba(bg_hex)

    canvas_w = m_left + w + m_right
    canvas_h = m_top + h + m_bottom
    out = Image.new("RGBA", (canvas_w, canvas_h), bg)
    arr = np.array(out)

    # 3. Draw Segments
    # Optimization: Iterate pixels is slow in python.
    # Use numpy assignment.
    # Create an image array of shape (h, w, 4)
    # We can map seg_map indices to colors.

    # Build a palette array: (N+1, 4) -> include background color for -1
    # Actually seg_map has -1 for background.
    # Let's shift indices: -1 -> 0, 0 -> 1 ...

    palette = np.zeros((n + 1, 4), dtype=np.uint8)
    # 0 is background (where seg_map == -1) - wait, mask is boolean.
    # seg_map is defined on mask=True pixels.
    # Where mask=False, seg_map is -1.
    # But where mask=True and unassigned, seg_map might be -1? Should be filled.

    # Colors for segments
    for i in range(n):
        palette[i+1] = colors[i]

    # Apply to seg_map
    # seg_map + 1 gives indices into palette.
    # But we need to handle the canvas offset.

    # Segment region
    region = seg_map + 1
    colored_region = palette[region] # (h, w, 4)

    # Assign to canvas array
    # Note: arr is (H, W, 4).
    # We target: arr[m_top:m_top+h, m_left:m_left+w]

    # Only update where mask is True
    # mask is (h, w) bool.
    # We can use it to index.

    canvas_roi = arr[m_top:m_top+h, m_left:m_left+w]
    canvas_roi[mask] = colored_region[mask]

    # Update array
    arr[m_top:m_top+h, m_left:m_left+w] = canvas_roi

    out = Image.fromarray(arr, mode="RGBA")
    draw = ImageDraw.Draw(out)

    # 4. Labels
    if not show_labels and not show_percentages:
        return out

    f_label = load_font(f_label_sz)
    f_pct = load_font(f_pct_sz)

    vals_sum = float(sum(values))
    seg_info = []

    for k in range(n):
        # Find centroid
        # Using numpy is fast
        ys, xs = np.where(seg_map == k)
        if ys.size == 0:
            continue
        cy = float(ys.mean())
        cx = float(xs.mean())
        pct = (float(values[k]) / vals_sum) * 100.0 if vals_sum > 0 else 0
        seg_info.append({"k": k, "label": labels[k], "pct": pct, "cx": cx, "cy": cy})

    left_items, right_items = [], []

    # Calculate text sizes for collision detection
    for it in seg_info:
        label_text = it["label"]
        pct_text = format_pct(it["pct"], decimal_comma=decimal_comma, decimals=1)

        # Measure text
        # getbbox returns (left, top, right, bottom) relative to (0,0)
        bbox_l = f_label.getbbox(label_text)
        h_l = bbox_l[3] - bbox_l[1] if bbox_l else f_label_sz

        bbox_p = f_pct.getbbox(pct_text)
        h_p = bbox_p[3] - bbox_p[1] if bbox_p else f_pct_sz

        total_h = h_l + h_p + (10 * upscale_factor) if show_labels and show_percentages else (h_l if show_labels else h_p)

        it["h"] = total_h
        it["h_l"] = h_l
        it["h_p"] = h_p

        anchor_x = m_left + it["cx"]
        anchor_y = m_top + it["cy"]

        # Decide side
        side = "left" if it["cx"] < (w / 2) else "right"

        item = {**it, "anchor_x": anchor_x, "anchor_y": anchor_y, "y": int(anchor_y)}
        (left_items if side == "left" else right_items).append(item)

    min_gap = int(f_label_sz * 0.5) # reduced gap
    y_min = m_top
    y_max = m_top + h

    avoid_overlaps_bbox(left_items, min_gap, y_min, y_max, canvas_h)
    avoid_overlaps_bbox(right_items, min_gap, y_min, y_max, canvas_h)

    line_color = (120, 120, 120, 255)
    text_color = (30, 30, 30, 255)
    pct_color = (140, 140, 140, 255)

    def draw_label(item: dict, side: str):
        ax = int(item["anchor_x"])
        ay = int(item["anchor_y"])
        y_text_center = int(item.get("y_adj", item["y"]))

        label = item["label"]
        pct_text = format_pct(item["pct"], decimal_comma=decimal_comma, decimals=1)

        # Text positioning
        # y_text_center is the vertical center of the block.
        # Top of block:
        block_top = y_text_center - item["h"] / 2

        # Draw Line
        if side == "left":
            x_line_end = m_left - (60 * upscale_factor)
            x_text = m_left - (70 * upscale_factor)

            # Elbow logic
            if elbow_lines:
                # Diagonal then Horizontal
                # Or: (ax, ay) -> (mid_x, y_text_center) -> (x_line_end, y_text_center)
                # Let's try: go left/right to a guide line?

                # Simple elbow:
                # 1. Start at ax, ay
                # 2. Go to (x_line_end + constant, ay) ? No.
                # 3. Go to (x_elbow, ay) then (x_line_end, y_text_center)?
                #    Diagonal is better.

                draw.line([(ax, ay), (x_line_end, y_text_center)], fill=line_color, width=max(1, 2*upscale_factor))
                # Wait, straight line is what we had.
                # Elbow: (ax, ay) -> (elbow_point) -> (end)
                # Let's do: (ax, ay) -> (x_line_end + 30, y_text_center) -> (x_line_end, y_text_center)

                elbow_x = x_line_end + (30 * upscale_factor)
                draw.line([(ax, ay), (elbow_x, y_text_center)], fill=line_color, width=max(1, 2*upscale_factor))
                draw.line([(elbow_x, y_text_center), (x_line_end, y_text_center)], fill=line_color, width=max(1, 2*upscale_factor))

            else:
                 draw.line([(ax, ay), (x_line_end, y_text_center)], fill=line_color, width=max(1, 2*upscale_factor))

            # Dot
            r = 4 * upscale_factor
            draw.ellipse((ax - r, ay - r, ax + r, ay + r), fill=line_color)

            # Draw Text (Right aligned)
            # We need width of text to right align
            if show_labels:
                bbox = f_label.getbbox(label)
                tw = bbox[2] - bbox[0]
                draw.text((x_text - tw, block_top), label, font=f_label, fill=text_color)

            if show_percentages:
                bbox = f_pct.getbbox(pct_text)
                tw = bbox[2] - bbox[0]
                y_p = block_top + item["h_l"] + (5 * upscale_factor) if show_labels else block_top
                draw.text((x_text - tw, y_p), pct_text, font=f_pct, fill=pct_color)

        else: # Right side
            x_line_end = m_left + w + (60 * upscale_factor)
            x_text = m_left + w + (70 * upscale_factor)

            if elbow_lines:
                elbow_x = x_line_end - (30 * upscale_factor)
                draw.line([(ax, ay), (elbow_x, y_text_center)], fill=line_color, width=max(1, 2*upscale_factor))
                draw.line([(elbow_x, y_text_center), (x_line_end, y_text_center)], fill=line_color, width=max(1, 2*upscale_factor))
            else:
                draw.line([(ax, ay), (x_line_end, y_text_center)], fill=line_color, width=max(1, 2*upscale_factor))

            r = 4 * upscale_factor
            draw.ellipse((ax - r, ay - r, ax + r, ay + r), fill=line_color)

            # Left aligned
            if show_labels:
                draw.text((x_text, block_top), label, font=f_label, fill=text_color)
            if show_percentages:
                y_p = block_top + item["h_l"] + (5 * upscale_factor) if show_labels else block_top
                draw.text((x_text, y_p), pct_text, font=f_pct, fill=pct_color)

    for it in left_items:
        draw_label(it, "left")
    for it in right_items:
        draw_label(it, "right")

    return out
