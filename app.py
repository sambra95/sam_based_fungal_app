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

st.html(
    """
<style>

.stAppHeader { background-color: #E9F2FF; !important; }

.stAppHeader span, .stAppHeader div {
    font-weight: 600 !important;
}

.stAppHeader {
    padding: 12px 20px !important;
}

.stAppHeader {
    box-shadow: 0 2px 20px rgba(0,0,0,0.2) !important;
}

/* Increase text size inside the navbar/header */
.stAppHeader span, .stAppHeader h1, .stAppHeader div {
    font-size: 20px !important;
}
</style>
"""
)

# ------------------ Define pages ------------------ #
pages = [
    st.Page(
        "views/1_home_page.py",
        title="Welcome to Mycoscope",
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
        title="Create and Edit Masks",
        icon="🎭",
    ),
    st.Page(
        "views/4_Fine_Tune_Models.py",
        title="Train Models",
        icon="🧠",
    ),
    st.Page(
        "views/5_Cell_Metrics.py",
        title="Visualize Class Attributes",
        icon="📊",
    ),
]

# ------------------ TOP navigation ------------------ #
nav = st.navigation(pages, position="top", expanded=False)

# ------------------ Run selected page ------------------ #
nav.run()
