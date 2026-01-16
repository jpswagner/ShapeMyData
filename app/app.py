import io
import os
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Local imports
from mask_utils import (
    load_default_rs_mask_from_repo,
    load_mask_from_alpha_image,
    extract_mask_from_shape_image,
    resize_mask_bool,
    hex_to_rgba
)
from render import render_shape_area_chart

# =========================
# Paths
# =========================
APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
ASSETS_DIR = REPO_DIR / "assets"
DEFAULT_RS_MASK_PATH = ASSETS_DIR / "rs_mask.png"

# =========================
# State Management
# =========================
def reset_defaults():
    st.session_state["labels"] = ["Var 1", "Var 2"]
    st.session_state["values"] = [1.0, 1.0]
    st.session_state["colors"] = ["#6AA84C", "#F2C329"]
    st.session_state["n_vars"] = 2
    # Reset other layout params if needed? Keeping it simple.

if "n_vars" not in st.session_state:
    st.session_state["n_vars"] = 2
if "labels" not in st.session_state:
    st.session_state["labels"] = ["Var 1", "Var 2"]
if "values" not in st.session_state:
    st.session_state["values"] = [1.0, 1.0]
if "colors" not in st.session_state:
    st.session_state["colors"] = ["#6AA84C", "#F2C329"]

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Shape Area Chart", layout="wide")
st.title("Shape Area Chart (pixel-exact)")

st.markdown(
    """
    Generate area-proportional charts within a shape with pixel-exact precision.
    """
)

# Sidebar
with st.sidebar:
    st.header("1. Shape")
    shape_source = st.radio(
        "Source",
        ["Default (RS)", "Upload"],
        index=0,
    )

    sat_thresh = 0.20
    upload_kind = "mask_alpha"
    uploaded = None

    if shape_source == "Upload":
        uploaded = st.file_uploader("Upload Image (PNG/JPG)", type=["png", "jpg", "jpeg", "webp"])
        kind = st.radio("Upload Type", ["Mask PNG (alpha)", "Shape image (auto-detect)"], index=0)
        if kind == "Mask PNG (alpha)":
            upload_kind = "mask_alpha"
        else:
            upload_kind = "shape_auto"
            sat_thresh = st.slider("Saturation Threshold", 0.05, 0.60, 0.20, 0.01)

    st.header("2. Segmentation")
    segmentation = st.selectbox(
        "Mode",
        ["Vertical", "Horizontal", "Angular", "Contiguous Wedge"],
        index=0
    )
    direction = "left_to_right"
    if segmentation == "Vertical":
        direction = st.selectbox("Direction", ["left_to_right", "right_to_left"], index=0)
    elif segmentation == "Horizontal":
        direction = st.selectbox("Direction", ["top_to_bottom", "bottom_to_top"], index=0)

    st.header("3. Appearance")
    bg_hex = st.color_picker("Background", "#F8F4EE")
    decimal_comma = st.toggle("Decimal Comma", value=True)
    show_labels = st.toggle("Show Labels", value=True)
    show_percentages = st.toggle("Show Percentages", value=True)
    elbow_lines = st.toggle("Elbow Leader Lines", value=True)

    with st.expander("Advanced Layout"):
        max_side = st.slider("Processing Resolution (Max Side)", 400, 2000, 900, 100)
        upscale_factor = st.selectbox("Output Upscale (DPI)", [1, 2, 3, 4], index=0)

        font_label = st.slider("Label Font Size", 18, 100, 40, 1)
        font_pct = st.slider("Percent Font Size", 16, 100, 38, 1)
        margin_left = st.slider("Left Margin", 150, 800, 420, 10)
        margin_right = st.slider("Right Margin", 150, 800, 420, 10)
        margin_top = st.slider("Top Margin", 20, 400, 60, 5)
        margin_bottom = st.slider("Bottom Margin", 20, 400, 60, 5)

# Load Mask
try:
    if shape_source == "Default (RS)":
        mask = load_default_rs_mask_from_repo(DEFAULT_RS_MASK_PATH)
    else:
        if uploaded is None:
            st.info("Please upload a file or select Default.")
            st.stop()
        img = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGBA")
        if upload_kind == "mask_alpha":
            mask = load_mask_from_alpha_image(img)
        else:
            mask = extract_mask_from_shape_image(img, sat_thresh=sat_thresh)
except Exception as e:
    st.error(f"Error loading mask: {e}")
    st.stop()

# Resize mask for processing (performance)
# Note: we use this resized mask for everything.
# If upscale_factor > 1, the render function will upscale THIS mask further.
mask = resize_mask_bool(mask, max_side=max_side)

# Mask Stats
h, w = mask.shape
area_px = np.sum(mask)
with st.sidebar.expander("Mask Statistics"):
    st.write(f"Dimensions: {w} x {h}")
    st.write(f"Area: {area_px} pixels")
    # Preview
    preview_img = Image.fromarray((mask.astype(np.uint8) * 255)).resize((150, int(150*h/w)))
    st.image(preview_img, caption="Mask Preview", clamp=True)


# Data Input
st.subheader("Data")
c_data, c_actions = st.columns([3, 1])

with c_actions:
    if st.button("Reset Defaults"):
        reset_defaults()
        st.rerun()

    # CSV Upload
    csv_file = st.file_uploader("Import CSV", type=["csv"], help="Columns: label, value, color(optional)")
    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            # Normalize columns
            df.columns = [c.lower().strip() for c in df.columns]
            if "label" in df.columns and "value" in df.columns:
                st.session_state["labels"] = df["label"].astype(str).tolist()
                st.session_state["values"] = df["value"].astype(float).tolist()
                st.session_state["n_vars"] = len(df)

                # Colors
                if "color" in df.columns:
                     st.session_state["colors"] = df["color"].astype(str).tolist()
                else:
                    # Generate colors if needed
                    default_palette = [
                        "#6AA84C", "#F2C329", "#3C78D8", "#E06666", "#8E7CC3",
                        "#76A5AF", "#C27BA0", "#93C47D", "#FFD966", "#A4C2F4",
                    ]
                    current_len = len(st.session_state["colors"])
                    req_len = len(df)
                    if req_len > current_len:
                        st.session_state["colors"] += [default_palette[i % len(default_palette)] for i in range(current_len, req_len)]
                    else:
                        st.session_state["colors"] = st.session_state["colors"][:req_len]
                st.success("CSV Imported!")
                st.rerun()
            else:
                st.error("CSV must have 'label' and 'value' columns.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

with c_data:
    n = st.number_input("Number of Variables", min_value=2, max_value=50, value=st.session_state["n_vars"], step=1)
    if int(n) != st.session_state["n_vars"]:
        # Adjust lists
        diff = int(n) - st.session_state["n_vars"]
        if diff > 0:
            st.session_state["labels"].extend([f"Var {i+1}" for i in range(st.session_state["n_vars"], int(n))])
            st.session_state["values"].extend([1.0] * diff)
            default_palette = [
                "#6AA84C", "#F2C329", "#3C78D8", "#E06666", "#8E7CC3",
                "#76A5AF", "#C27BA0", "#93C47D", "#FFD966", "#A4C2F4",
            ]
            st.session_state["colors"].extend([default_palette[i % len(default_palette)] for i in range(st.session_state["n_vars"], int(n))])
        else:
            st.session_state["labels"] = st.session_state["labels"][:int(n)]
            st.session_state["values"] = st.session_state["values"][:int(n)]
            st.session_state["colors"] = st.session_state["colors"][:int(n)]
        st.session_state["n_vars"] = int(n)
        st.rerun()

# Inputs Table / Grid
# Using an expander for inputs if N is large?
container = st.container()
if st.session_state["n_vars"] > 10:
    container = st.expander("Edit Variables (Click to expand)", expanded=True)

with container:
    for i in range(st.session_state["n_vars"]):
        c1, c2 = st.columns([2, 1])
        st.session_state["labels"][i] = c1.text_input(f"Label {i+1}", st.session_state["labels"][i], key=f"label_{i}", label_visibility="collapsed")
        st.session_state["values"][i] = c2.number_input(
            f"Value {i+1}",
            value=float(st.session_state["values"][i]),
            min_value=0.0,
            step=1.0,
            key=f"value_{i}",
            label_visibility="collapsed"
        )

# Color Grid
st.markdown("### Colors")
color_container = st.container()
if st.session_state["n_vars"] > 12:
    color_container = st.expander("Edit Colors", expanded=False)

with color_container:
    grid_cols = 6
    rows = math.ceil(st.session_state["n_vars"] / grid_cols)
    idx = 0
    for r in range(rows):
        cols = st.columns(grid_cols)
        for c in range(grid_cols):
            if idx >= st.session_state["n_vars"]:
                break
            with cols[c]:
                # Shorten label for display
                lbl = st.session_state["labels"][idx]
                if len(lbl) > 10: lbl = lbl[:8] + ".."
                st.caption(lbl)
                st.session_state["colors"][idx] = st.color_picker(
                    f"Color {idx}",
                    st.session_state["colors"][idx],
                    key=f"color_{idx}",
                )
            idx += 1

# Render
st.subheader("Result")

labels = st.session_state["labels"]
values = st.session_state["values"]
colors_hex = st.session_state["colors"]

if sum(values) <= 0:
    st.error("Sum of values must be > 0.")
    st.stop()

with st.spinner("Rendering..."):
    try:
        out_img = render_shape_area_chart(
            mask_original=mask,
            labels=labels,
            values=values,
            colors_hex=colors_hex,
            bg_hex=bg_hex,
            segmentation=segmentation,
            direction=direction,
            decimal_comma=decimal_comma,
            font_label_size=font_label,
            font_pct_size=font_pct,
            margin_left=margin_left,
            margin_right=margin_right,
            margin_top=margin_top,
            margin_bottom=margin_bottom,
            upscale_factor=upscale_factor,
            show_labels=show_labels,
            show_percentages=show_percentages,
            elbow_lines=elbow_lines
        )
        st.image(out_img, use_container_width=True, caption=f"Result ({out_img.width}x{out_img.height} px)")

        # Downloads
        c_d1, c_d2 = st.columns(2)
        with c_d1:
            buf_png = io.BytesIO()
            out_img.save(buf_png, format="PNG")
            st.download_button(
                "Download PNG",
                data=buf_png.getvalue(),
                file_name="chart.png",
                mime="image/png",
            )
        with c_d2:
            buf_pdf = io.BytesIO()
            # PDF requires RGB usually, but RGBA handles transparency?
            # PIL saves PDF. If RGBA, it might flatten?
            # "PDF supports transparency... but usually it's better to flatten or ensure background."
            # We have a background color.
            # Convert to RGB with background if needed?
            # But render has bg_hex.
            out_img.convert("RGB").save(buf_pdf, format="PDF")
            st.download_button(
                "Download PDF",
                data=buf_pdf.getvalue(),
                file_name="chart.pdf",
                mime="application/pdf",
            )

    except Exception as e:
        st.exception(e)
