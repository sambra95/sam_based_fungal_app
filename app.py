# app_hydralit.py
import os
import runpy

import streamlit as st

# Your original boot steps
from boot import common_boot, configure_tf_cpu_only
from helpers.state_ops import ensure_global_state

# from hydralit import HydraApp
# import hydralit_components as hc

# ------------------ App setup ------------------ #


ensure_global_state()
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
configure_tf_cpu_only()
# common_boot()


st.set_page_config(page_title="Mycoscope", page_icon="🧬", layout="wide")

# hide Streamlit's sidebar nav (since Hydralit provides top nav)
# st.markdown(
#     """
#     <style>
#       [data-testid="stSidebarNav"] { display: none; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# remove white space above the navbar and around the page
st.markdown(
    """
<style>
/* Remove all outer padding */
div.block-container {
    padding-top: 0rem;
    padding-bottom: 0rem;
    padding-left: 1rem;
    padding-right: 1rem;
    margin-top: -40px !important;

}

</style>
""",
    unsafe_allow_html=True,
)


# Helper to execute a standard Streamlit script as a "page"
def run_view(script_path: str):
    # Execute the target Streamlit script in its own namespace
    # Assumes those scripts do NOT call st.set_page_config again.
    runpy.run_path(script_path, run_name="__main__")


# ------------------ Hydralit app ------------------ #
# app = HydraApp(
#     title="Mycoscope",
#     favicon="🧬",
#     use_loader=False,
#     hide_streamlit_markers=True,  # cleaner header
# )


# Home page (default)
app = st.navigation (
[
    st.Page("views/1_home_page.py", title="", icon="🏠"),



    st.Page ("views/2_Upload_data.py",title="Upload Models and Data", icon="📥"),



    st.Page("views/3_Create_and_Edit_Masks.py",title="Segment and Classify Cells", icon="🎭"),



    st.Page("views/4_Fine_Tune_Models.py",title="Train Segmentation and Classification Models", icon="🧠"),



    st.Page("views/5_Cell_Metrics.py", title="Analyze Cell Groups", icon="📊"),

]
# ------------------ Run ------------------ #
)
app.run()
