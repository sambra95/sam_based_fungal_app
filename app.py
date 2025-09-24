import io
from contextlib import nullcontext
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates
import tifffile as tiff
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- import helpers ---
from helpers.image_io import load_masks_any
from helpers.masks import (
    polygon_to_mask,
    toggle_at_point,
    composite_over,
    stack_to_instances_binary_first,
    _resize_mask_nearest,
    _attach_masks_to_image,
    zip_all_masks,
)
from helpers.boxes import is_unique_box, boxes_to_fabric_rects, draw_boxes_overlay
from helpers.state_ops import (
    ensure_global_state,
    ordered_keys,
    ensure_image,
    current,
    set_current_by_index,
    add_drawn_mask,
    stem,
)
from helpers import config as cfg  # has SAM2 model and yaml paths

# ============================================================
# ---------- Import App Panels --------
# ============================================================

from panels import upload, mask_editing, classify_cells


st.set_page_config(page_title="Mask Toggle", layout="wide")


# ============================================================
# ---------- Ensure consistant default s --------
# ============================================================

ensure_global_state()

# ============================================================
# ---------------------------- Sidebar -----------------------
# ============================================================


with st.sidebar:

    st.markdown("### App panels for different tasks:")

    panel = st.radio(
        "",
        ["Upload data", "Create and Edit Masks", "Classify Cells"],
        key="side_panel",
    )

    # -------- Create & Edit (combined) --------

    if panel == "Create and Edit Masks":
        mask_editing.render_sidebar(key_ns="side")

    elif panel == "Classify Cells":
        classify_cells.render_sidebar(key_ns="side")


# ============================================================
# --------------------------- Main area ----------------------
# ============================================================

# -------- Upload panel --------
if panel == "Upload data":

    upload.render()

elif panel == "Create and Edit Masks":

    mask_editing.render_main(key_ns="edit")

elif panel == "Classify Cells":

    classify_cells.render_main(key_ns="classify")
