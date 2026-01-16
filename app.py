import io
import os
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# =========================
# Paths (repo-based default)
# =========================
APP_DIR = Path(__file__).resolve().parent
DEFAULT_RS_MASK_PATH = APP_DIR / "assets" / "rs_mask.png"


# =========================
# Utilities
# =========================
def hex_to_rgba(h: str) -> Tuple[int, int, int, int]:
    h = h.strip().lstrip("#")
    if len(h) == 6:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)
    if len(h) == 8:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16))
    raise ValueError(f"Invalid hex color: {h}")

def normalize_targets(values: List[float], total_pixels: int) -> List[int]:
    vals = np.array(values, dtype=float)
    if np.any(vals < 0):
        raise ValueError("All values must be non-negative.")
    s = float(vals.sum())
    if s <= 0:
        raise ValueError("Sum of values must be > 0.")
    raw = vals / s * total_pixels
    base = np.floor(raw).astype(int)
    remainder = total_pixels - int(base.sum())
    frac = raw - base
    order = np.argsort(-frac)
    for i in range(remainder):
        base[order[i % len(base)]] += 1
    return base.tolist()

def load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()

def format_pct(x: float, decimal_comma: bool = True, decimals: int = 1) -> str:
    s = f"{x:.{decimals}f}%"
    return s.replace(".", ",") if decimal_comma else s

def resize_mask_bool(mask: np.ndarray, max_side: int) -> np.ndarray:
    h, w = mask.shape
    if max(h, w) <= max_side:
        return mask
    scale = max_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    im = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    im = im.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    arr = np.array(im)
    return arr > 127


# =========================
# Mask extraction helpers
# =========================
def rgb_to_hsv_vec(rgb: np.ndarray):
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn

    h = np.zeros_like(mx)
    mask = diff > 1e-8
    idx = (mx == r) & mask
    h[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6.0
    idx = (mx == g) & mask
    h[idx] = ((b[idx] - r[idx]) / diff[idx]) + 2.0
    idx = (mx == b) & mask
    h[idx] = ((r[idx] - g[idx]) / diff[idx]) + 4.0
    h = (h / 6.0) % 1.0

    s = np.zeros_like(mx)
    nz = mx > 1e-8
    s[nz] = diff[nz] / mx[nz]
    v = mx
    return h, s, v

def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_coords = None
    best_size = 0

    def neighbors(y, x):
        if y > 0: yield y - 1, x
        if y < h - 1: yield y + 1, x
        if x > 0: yield y, x - 1
        if x < w - 1: yield y, x + 1

    ys_all, xs_all = np.where(mask)
    for y0, x0 in zip(ys_all, xs_all):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        coords = [(y0, x0)]
        while stack:
            y, x = stack.pop()
            for ny, nx in neighbors(y, x):
                if mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
                    coords.append((ny, nx))
        if len(coords) > best_size:
            best_size = len(coords)
            best_coords = coords

    out = np.zeros_like(mask, dtype=bool)
    if best_coords:
        ys, xs = zip(*best_coords)
        out[np.array(ys), np.array(xs)] = True
    return out

def load_mask_from_alpha_image(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGBA"))
    alpha = arr[..., 3]
    m = alpha > 0
    return largest_connected_component(m)

def extract_mask_from_shape_image(img: Image.Image, sat_thresh: float = 0.20) -> np.ndarray:
    arr = np.array(img.convert("RGBA")).astype(np.float32) / 255.0
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    _, s, _ = rgb_to_hsv_vec(rgb)
    rough = (alpha > 0.05) & (s > sat_thresh)
    return largest_connected_component(rough)


# =========================
# Segment assignment (pixel-exact)
# =========================
def assign_segments_vertical(mask: np.ndarray, targets: List[int], direction: str) -> np.ndarray:
    h, w = mask.shape
    seg_map = np.full((h, w), -1, dtype=np.int32)
    remaining = targets.copy()
    seg = 0
    n = len(targets)

    x_range = range(w) if direction == "left_to_right" else range(w - 1, -1, -1)
    for x in x_range:
        ys = np.where(mask[:, x])[0]
        if ys.size == 0:
            continue
        i = 0
        while i < ys.size and seg < n:
            take = min(remaining[seg], int(ys.size - i))
            if take > 0:
                seg_map[ys[i:i + take], x] = seg
                remaining[seg] -= take
                i += take
            if seg < n and remaining[seg] == 0:
                seg += 1

    unassigned = mask & (seg_map < 0)
    if unassigned.any():
        seg_map[unassigned] = n - 1
    return seg_map

def assign_segments_horizontal(mask: np.ndarray, targets: List[int], direction: str) -> np.ndarray:
    h, w = mask.shape
    seg_map = np.full((h, w), -1, dtype=np.int32)
    remaining = targets.copy()
    seg = 0
    n = len(targets)

    y_range = range(h) if direction == "top_to_bottom" else range(h - 1, -1, -1)
    for y in y_range:
        xs = np.where(mask[y, :])[0]
        if xs.size == 0:
            continue
        i = 0
        while i < xs.size and seg < n:
            take = min(remaining[seg], int(xs.size - i))
            if take > 0:
                seg_map[y, xs[i:i + take]] = seg
                remaining[seg] -= take
                i += take
            if seg < n and remaining[seg] == 0:
                seg += 1

    unassigned = mask & (seg_map < 0)
    if unassigned.any():
        seg_map[unassigned] = n - 1
    return seg_map

def assign_segments_angular(mask: np.ndarray, targets: List[int]) -> np.ndarray:
    ys, xs = np.where(mask)
    coords = np.stack([ys, xs], axis=1).astype(np.float32)
    cy, cx = coords.mean(axis=0)

    dy = ys.astype(np.float32) - cy
    dx = xs.astype(np.float32) - cx
    ang = np.arctan2(dy, dx)
    ang = (ang + 2 * np.pi) % (2 * np.pi)
    r = np.sqrt(dx * dx + dy * dy)

    order = np.lexsort((r, ang))
    seg_map = np.full(mask.shape, -1, dtype=np.int32)

    start = 0
    for seg, t in enumerate(targets):
        end = start + t
        idx = order[start:end]
        seg_map[ys[idx], xs[idx]] = seg
        start = end

    unassigned = mask & (seg_map < 0)
    if unassigned.any():
        seg_map[unassigned] = len(targets) - 1
    return seg_map


# =========================
# Label layout + render
# =========================
def avoid_overlaps(items: List[dict], min_gap: int, y_min: int, y_max: int) -> None:
    items.sort(key=lambda d: d["y"])
    last = y_min
    for it in items:
        y = max(it["y"], last)
        it["y_adj"] = y
        last = y + min_gap

    if not items:
        return

    overflow = items[-1]["y_adj"] - y_max
    if overflow > 0:
        for it in reversed(items):
            it["y_adj"] -= overflow
        for i in range(1, len(items)):
            if items[i]["y_adj"] < items[i - 1]["y_adj"] + min_gap:
                items[i]["y_adj"] = items[i - 1]["y_adj"] + min_gap

def render_shape_area_chart(
    mask: np.ndarray,
    labels: List[str],
    values: List[float],
    colors_hex: List[str],
    bg_hex: str,
    segmentation: str,
    direction: str,
    decimal_comma: bool,
    font_label: int,
    font_pct: int,
    margin_left: int,
    margin_right: int,
    margin_top: int,
    margin_bottom: int,
) -> Image.Image:
    n = len(labels)
    h, w = mask.shape
    total = int(mask.sum())
    targets = normalize_targets(values, total)

    if segmentation == "Vertical":
        seg_map = assign_segments_vertical(mask, targets, direction)
    elif segmentation == "Horizontal":
        seg_map = assign_segments_horizontal(mask, targets, direction)
    else:
        seg_map = assign_segments_angular(mask, targets)

    colors = [hex_to_rgba(c) for c in colors_hex]
    bg = hex_to_rgba(bg_hex)

    canvas_w = margin_left + w + margin_right
    canvas_h = margin_top + h + margin_bottom
    out = Image.new("RGBA", (canvas_w, canvas_h), bg)
    arr = np.array(out)

    for k in range(n):
        yy, xx = np.where(seg_map == k)
        if yy.size:
            arr[margin_top + yy, margin_left + xx] = np.array(colors[k], dtype=np.uint8)

    out = Image.fromarray(arr, mode="RGBA")
    draw = ImageDraw.Draw(out)

    vals_sum = float(sum(values))
    seg_info = []
    for k in range(n):
        yy, xx = np.where(seg_map == k)
        if yy.size == 0:
            continue
        cy = float(yy.mean())
        cx = float(xx.mean())
        pct = (float(values[k]) / vals_sum) * 100.0
        seg_info.append({"k": k, "label": labels[k], "pct": pct, "cx": cx, "cy": cy})

    f_label = load_font(font_label)
    f_pct = load_font(font_pct)

    left_items, right_items = [], []
    for it in seg_info:
        anchor_x = margin_left + it["cx"]
        anchor_y = margin_top + it["cy"]
        side = "left" if it["cx"] < (w / 2) else "right"
        item = {**it, "anchor_x": anchor_x, "anchor_y": anchor_y, "y": int(anchor_y)}
        (left_items if side == "left" else right_items).append(item)

    min_gap = int(font_label * 1.5)
    y_min = margin_top + 20
    y_max = margin_top + h - 20
    avoid_overlaps(left_items, min_gap, y_min, y_max)
    avoid_overlaps(right_items, min_gap, y_min, y_max)

    line_color = (120, 120, 120, 255)
    text_color = (30, 30, 30, 255)
    pct_color = (140, 140, 140, 255)

    def draw_label(item: dict, side: str):
        ax = int(item["anchor_x"])
        ay = int(item["anchor_y"])
        y_text = int(item.get("y_adj", item["y"]))
        label = item["label"]
        pct_text = format_pct(item["pct"], decimal_comma=decimal_comma, decimals=1)

        if side == "left":
            x_line_end = margin_left - 60
            x_text = 30
            draw.line([(ax, ay), (x_line_end, y_text)], fill=line_color, width=3)
            draw.ellipse((ax - 4, ay - 4, ax + 4, ay + 4), fill=line_color)
            draw.text((x_text, y_text - font_label), label, font=f_label, fill=text_color)
            draw.text((x_text, y_text + 10), pct_text, font=f_pct, fill=pct_color)
        else:
            x_line_end = margin_left + w + 60
            x_text = margin_left + w + 90
            draw.line([(ax, ay), (x_line_end, y_text)], fill=line_color, width=3)
            draw.ellipse((ax - 4, ay - 4, ax + 4, ay + 4), fill=line_color)
            draw.text((x_text, y_text - font_label), label, font=f_label, fill=text_color)
            draw.text((x_text, y_text + 10), pct_text, font=f_pct, fill=pct_color)

    for it in left_items:
        draw_label(it, "left")
    for it in right_items:
        draw_label(it, "right")

    return out


# =========================
# Cached mask loader (file)
# =========================
@st.cache_data
def load_default_rs_mask_from_repo(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Default RS mask not found at: {p}")
    img = Image.open(p).convert("RGBA")
    mask = load_mask_from_alpha_image(img)
    return mask


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Shape Area Chart", layout="wide")
st.title("Shape Area Chart (pixel-exact)")

st.markdown(
    """
Gera um gráfico **com áreas proporcionais** dentro de um shape (pixel-exato).
- Default: **Rio Grande do Sul (RS)** via `assets/rs_mask.png`
- Você pode fazer upload de outro shape (mask PNG com alpha ou imagem para auto-detect)
"""
)

with st.sidebar:
    st.header("1) Shape")
    shape_source = st.radio(
        "Fonte do shape",
        ["Default (RS do repo)", "Upload"],
        index=0,
    )

    sat_thresh = 0.20
    upload_kind = "mask_alpha"
    uploaded = None

    if shape_source == "Upload":
        uploaded = st.file_uploader("Upload (PNG recomendado)", type=["png", "jpg", "jpeg", "webp"])
        kind = st.radio("Tipo de upload", ["Mask PNG (alpha)", "Shape image (auto-detect)"], index=0)
        if kind == "Mask PNG (alpha)":
            upload_kind = "mask_alpha"
        else:
            upload_kind = "shape_auto"
            sat_thresh = st.slider("Auto-detect (saturation threshold)", 0.05, 0.60, 0.20, 0.01)

    st.header("2) Segmentation")
    segmentation = st.selectbox("Modo", ["Vertical", "Horizontal", "Angular"], index=0)
    direction = "left_to_right"
    if segmentation == "Vertical":
        direction = st.selectbox("Direcao", ["left_to_right", "right_to_left"], index=0)
    elif segmentation == "Horizontal":
        direction = st.selectbox("Direcao", ["top_to_bottom", "bottom_to_top"], index=0)

    st.header("3) Render")
    bg_hex = st.color_picker("Background", "#F8F4EE")
    decimal_comma = st.toggle("Usar virgula decimal", value=True)
    max_side = st.slider("Max shape size (performance)", 400, 1800, 900, 50)

    with st.expander("Advanced layout"):
        font_label = st.slider("Label font", 18, 64, 40, 1)
        font_pct = st.slider("Percent font", 16, 64, 38, 1)
        margin_left = st.slider("Left margin", 150, 800, 420, 10)
        margin_right = st.slider("Right margin", 150, 800, 420, 10)
        margin_top = st.slider("Top margin", 20, 250, 60, 5)
        margin_bottom = st.slider("Bottom margin", 20, 250, 60, 5)

# Load mask
try:
    if shape_source == "Default (RS do repo)":
        mask = load_default_rs_mask_from_repo(str(DEFAULT_RS_MASK_PATH))
    else:
        if uploaded is None:
            st.info("Faça upload de um arquivo, ou selecione o default (RS).")
            st.stop()
        img = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGBA")
        if upload_kind == "mask_alpha":
            mask = load_mask_from_alpha_image(img)
        else:
            mask = extract_mask_from_shape_image(img, sat_thresh=sat_thresh)
except Exception as e:
    st.error(str(e))
    st.stop()

mask = resize_mask_bool(mask, max_side=max_side)

# Variables
st.subheader("Variables")
n = st.number_input("How many variables?", min_value=2, max_value=30, value=2, step=1)

# Session state init
if "labels" not in st.session_state or len(st.session_state["labels"]) != int(n):
    st.session_state["labels"] = [f"Var {i+1}" for i in range(int(n))]
if "values" not in st.session_state or len(st.session_state["values"]) != int(n):
    st.session_state["values"] = [1.0 for _ in range(int(n))]
if "colors" not in st.session_state or len(st.session_state["colors"]) != int(n):
    default_palette = [
        "#6AA84C", "#F2C329", "#3C78D8", "#E06666", "#8E7CC3",
        "#76A5AF", "#C27BA0", "#93C47D", "#FFD966", "#A4C2F4",
    ]
    st.session_state["colors"] = [default_palette[i % len(default_palette)] for i in range(int(n))]

# Inputs table
for i in range(int(n)):
    c1, c2 = st.columns([2, 1])
    st.session_state["labels"][i] = c1.text_input(f"Label {i+1}", st.session_state["labels"][i], key=f"label_{i}")
    st.session_state["values"][i] = c2.number_input(
        f"Value {i+1}",
        value=float(st.session_state["values"][i]),
        min_value=0.0,
        step=1.0,
        key=f"value_{i}",
    )

# Color grid
st.markdown("### Colors (grid)")
grid_cols = 6
rows = math.ceil(int(n) / grid_cols)
idx = 0
for r in range(rows):
    cols = st.columns(grid_cols)
    for c in range(grid_cols):
        if idx >= int(n):
            break
        with cols[c]:
            st.caption(st.session_state["labels"][idx])
            st.session_state["colors"][idx] = st.color_picker(
                f"Color {idx+1}",
                st.session_state["colors"][idx],
                key=f"color_{idx}",
            )
        idx += 1

labels = st.session_state["labels"]
values = st.session_state["values"]
colors_hex = st.session_state["colors"]

if sum(values) <= 0:
    st.error("Sum of values must be > 0.")
    st.stop()

# Render
st.subheader("Result")
try:
    out_img = render_shape_area_chart(
        mask=mask,
        labels=labels,
        values=values,
        colors_hex=colors_hex,
        bg_hex=bg_hex,
        segmentation=segmentation,
        direction=direction,
        decimal_comma=decimal_comma,
        font_label=font_label,
        font_pct=font_pct,
        margin_left=margin_left,
        margin_right=margin_right,
        margin_top=margin_top,
        margin_bottom=margin_bottom,
    )
except Exception as e:
    st.exception(e)
    st.stop()

st.image(out_img, use_container_width=True)

buf = io.BytesIO()
out_img.save(buf, format="PNG")
st.download_button(
    "Download PNG",
    data=buf.getvalue(),
    file_name="shape_area_chart.png",
    mime="image/png",
)
