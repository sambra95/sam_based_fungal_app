# app.py
import streamlit as st
import os
from boot import common_boot, configure_tf_cpu_only
from helpers.state_ops import ensure_global_state

st.set_page_config(page_title="Mycoscope", page_icon="🧬", layout="wide")

ensure_global_state()
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
configure_tf_cpu_only()

common_boot()

pages = {
    "Choose a task from the workflow:": [
        st.Page("views/1_Upload_data.py", title="Upload Models and Data", icon="📥"),
        st.Page(
            "views/2_Create_and_Edit_Masks.py",
            title="Segment and Classify Cells",
            icon="🎭",
        ),
        st.Page(
            "views/4_Fine_Tune_Models.py",
            title="Train Segmentation and Classification Models",
            icon="🧠",
        ),
        st.Page("views/5_Cell_Metrics.py", title="Analyze Cell Groups", icon="📊"),
    ],
}

pg = st.navigation(pages)
pg.run()
