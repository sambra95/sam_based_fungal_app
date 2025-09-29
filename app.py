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

from helpers.state_ops import (
    ensure_global_state,
)

# ============================================================
# ---------- Import App Panels --------
# ============================================================

from panels import (
    upload_panel,
    mask_editing_panel,
    classify_cells_panel,
    cell_metrics_panel,
    fine_tune_panel,
)

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
        [
            "Upload data",
            "Create and Edit Masks",
            "Classify Cells",
            "Fine Tune Models",
            "Cell Metrics",
        ],
        key="side_panel",
    )

    st.divider()

    # -------- Create & Edit (combined) --------

    if panel == "Create and Edit Masks":
        mask_editing_panel.render_sidebar(key_ns="side")

    elif panel == "Classify Cells":
        classify_cells_panel.render_sidebar(key_ns="side")

    elif panel == "Cell Metrics":

        cell_metrics_panel.render_sidebar()

    elif panel == "Fine Tune Models":

        fine_tune_panel.render_sidebar()


# ============================================================
# --------------------------- Main area ----------------------
# ============================================================

# -------- Upload panel --------
if panel == "Upload data":

    upload_panel.render_main()

elif panel == "Create and Edit Masks":

    mask_editing_panel.render_main(key_ns="edit")

elif panel == "Classify Cells":

    classify_cells_panel.render_main(key_ns="classify")

elif panel == "Cell Metrics":

    cell_metrics_panel.render_main()

elif panel == "Fine Tune Models":

    fine_tune_panel.render_main()
