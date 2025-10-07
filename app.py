import streamlit as st
import os

from helpers.state_ops import (
    ensure_global_state,
)

ensure_global_state()


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


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

configure_tf_cpu_only()

upload = st.Page(
    "pages/1_📥_Upload_data.py", title="Upload Images, Masks and Models", icon="📥"
)
edit = st.Page(
    "pages/2_🎭_Create_and_Edit_Masks.py",
    title="Create & Edit Segmentation Masks",
    icon="🎭",
)
classify = st.Page(
    "pages/3_🧬_Classify_Cells.py",
    title="Create and Edit Mask Classifications",
    icon="🧬",
)
metrics = st.Page(
    "pages/5_📊_Cell_Metrics.py",
    title="Analyze and Compare Cell Characterisatics",
    icon="📊",
)
tune = st.Page(
    "pages/4_🧠_Fine_Tune_Models.py",
    title="Fine Tune Segmentation and Classification Models",
    icon="🧠",
)
dl = st.Page(
    "pages/6_⬇️_Downloads.py", title="Download Datasets and Trained Models", icon="⬇️"
)

nav = st.navigation(
    {
        "Workflow": [upload, edit, classify],
        "Use your datasets": [metrics, tune],
        "Download": [dl],
    }
)
nav.run()
