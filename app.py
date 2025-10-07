import streamlit as st

# --- import helpers ---

from helpers.state_ops import (
    ensure_global_state,
)

# ============================================================
# ---------- Import App Panels --------
# ============================================================

# from panels import (
#     upload_panel,
#     mask_editing_panel,
#     classify_cells_panel,
#     cell_metrics_panel,
#     fine_tune_panel,
# )

st.set_page_config(page_title="Mask Toggle", layout="wide")

# ============================================================
# ---------- Ensure consistant default s --------
# ============================================================

ensure_global_state()

# ============================================================
# -------------------- Force TF to CPU -----------------------
# ============================================================

# Mixing frameworks on the same accelerator (CUDA/MPS) often OOMs or hard-crashes.
# Already using Torch (SAM2/Cellpose).
# When DenseNet spins up, TF will try to grab the accelerator too. Causing a crash


@st.cache_resource(show_spinner=False)
def configure_tf_cpu_only():
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    return True


configure_tf_cpu_only()

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

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
            "Downloads",
        ],
        key="side_panel",
    )

    st.divider()

    # -------- Create & Edit (combined) --------

    if panel == "Create and Edit Masks":
        from panels import mask_editing_panel

        mask_editing_panel.render_sidebar(key_ns="side")

    elif panel == "Classify Cells":
        from panels import classify_cells_panel

        classify_cells_panel.render_sidebar(key_ns="side")

    elif panel == "Cell Metrics":
        from panels import cell_metrics_panel

        cell_metrics_panel.render_sidebar()

    elif panel == "Fine Tune Models":
        None

    elif panel == "Downloads":
        None

# ============================================================
# --------------------------- Main area ----------------------
# ============================================================

# -------- Upload panel --------
if panel == "Upload data":

    from panels import upload_panel

    upload_panel.render_main()

elif panel == "Create and Edit Masks":

    from panels import mask_editing_panel

    mask_editing_panel.render_main(key_ns="edit")

elif panel == "Classify Cells":

    from panels import classify_cells_panel

    classify_cells_panel.render_main(key_ns="classify")

elif panel == "Cell Metrics":

    from panels import cell_metrics_panel

    cell_metrics_panel.render_main()

elif panel == "Fine Tune Models":

    from panels import fine_tune_panel

    fine_tune_panel.render_cellpose_train_panel()
    fine_tune_panel.render_densenet_train_panel()

elif panel == "Downloads":

    from panels import download_panel

    download_panel.render_main()
