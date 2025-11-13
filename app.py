# app.py
import os
import streamlit as st

st.set_page_config(page_title="Mycoscope", page_icon="🧬", layout="wide")

# ------------------ Boot steps ------------------ #
from boot import configure_tf_cpu_only
from helpers.state_ops import ensure_global_state

ensure_global_state()
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
configure_tf_cpu_only()


# ------------------ Define pages ------------------ #
pages = [
    st.Page(
        "views/1_home_page.py",
        title="Home",
        icon="🏠",
        default=True,
    ),
    st.Page(
        "views/2_Upload_data.py",
        title="Upload Models and Data",
        icon="📥",
    ),
    st.Page(
        "views/3_Create_and_Edit_Masks.py",
        title="Segment and Classify Cells",
        icon="🎭",
    ),
    st.Page(
        "views/4_Fine_Tune_Models.py",
        title="Train Segmentation and Classification Models",
        icon="🧠",
    ),
    st.Page(
        "views/5_Cell_Metrics.py",
        title="Analyze Cell Groups",
        icon="📊",
    ),
]

# ------------------ TOP navigation ------------------ #
nav = st.navigation(pages, position="top", expanded=False)

# ------------------ Run selected page ------------------ #
nav.run()
