# app_hydralit.py
import os
import runpy
import streamlit as st
from hydralit import HydraApp
import hydralit_components as hc

# ------------------ App setup ------------------ #
st.set_page_config(page_title="Mycoscope", page_icon="🧬", layout="wide")

# Your original boot steps
from boot import common_boot, configure_tf_cpu_only
from helpers.state_ops import ensure_global_state

ensure_global_state()
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
configure_tf_cpu_only()
# common_boot()

# hide Streamlit's sidebar nav (since Hydralit provides top nav)
st.markdown(
    """
    <style>
      [data-testid="stSidebarNav"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
app = HydraApp(
    title="Mycoscope",
    favicon="🧬",
    use_loader=False,
    hide_streamlit_markers=True,  # cleaner header
)


# Home page (default)
@app.addapp(title="", icon="🏠")
def page_home():
    run_view("views/1_home_page.py")


@app.addapp(title="Upload Models and Data", icon="📥")
def page_upload():
    run_view("views/2_Upload_data.py")


@app.addapp(title="Segment and Classify Cells", icon="🎭")
def page_segment_classify():
    run_view("views/3_Create_and_Edit_Masks.py")


@app.addapp(title="Train Segmentation and Classification Models", icon="🧠")
def page_train_models():
    run_view("views/4_Fine_Tune_Models.py")


@app.addapp(title="Analyze Cell Groups", icon="📊")
def page_metrics():
    run_view("views/5_Cell_Metrics.py")


# ------------------ Run ------------------ #

app.run()


# import streamlit as st
# import numpy as np
# import plotly.graph_objects as go
# from PIL import Image

# if "boxes" not in st.session_state:
#     st.session_state["boxes"] = []

# if "bg_image" not in st.session_state:
#     st.session_state["bg_image"] = None
#     st.session_state["img_width"] = None
#     st.session_state["img_height"] = None


# def update_boxes():
#     event = st.session_state.get("box_draw_chart")
#     if event is None or event.selection is None:
#         return

#     if event.selection.box:
#         for b in event.selection.box:
#             clean_box = {
#                 "x0": b["x"][0],
#                 "x1": b["x"][1],
#                 "y0": b["y"][0],
#                 "y1": b["y"][1],
#             }
#             if clean_box not in st.session_state["boxes"]:
#                 st.session_state["boxes"].append(clean_box)


# def make_figure():
#     bg_img = st.session_state.get("bg_image")
#     w = st.session_state.get("img_width")
#     h = st.session_state.get("img_height")

#     fig = go.Figure()

#     if bg_img is not None and w is not None and h is not None:
#         fig.add_layout_image(
#             dict(
#                 source=bg_img,
#                 xref="x",
#                 yref="y",
#                 x=0,
#                 y=h,
#                 sizex=w,
#                 sizey=h,
#                 sizing="stretch",
#                 layer="below",
#             )
#         )
#         fig.update_xaxes(visible=False, range=[0, w], constrain="domain")
#         fig.update_yaxes(visible=False, range=[0, h], scaleanchor="x", scaleratio=1)

#     fig.update_layout(
#         dragmode="select",
#         margin=dict(l=0, r=0, t=0, b=0),
#     )

#     # draw stored boxes
#     for box in st.session_state["boxes"]:
#         x0 = box.get("x0")
#         x1 = box.get("x1")
#         y0 = box.get("y0")
#         y1 = box.get("y1")
#         if None in (x0, x1, y0, y1):
#             continue
#         fig.add_shape(
#             type="rect",
#             x0=x0,
#             x1=x1,
#             y0=y0,
#             y1=y1,
#             line=dict(color="red", width=2),
#             fillcolor="rgba(255,0,0,0.15)",
#             layer="above",
#         )

#     return fig


# @st.fragment
# def image_with_boxes():
#     fig = make_figure()
#     event = st.plotly_chart(
#         fig,
#         key="box_draw_chart",
#         on_select=update_boxes,  # updates boxes in session_state
#         selection_mode="box",
#         use_container_width=True,
#     )

#     # (Optional) show live selection object
#     if event is not None and getattr(event, "selection", None) is not None:
#         st.caption("Raw selection:")
#         st.json(event.selection)


# st.title("Image box annotator")

# left, right = st.columns([1, 4])

# with left:
#     # upload image once
#     uploaded_file = st.file_uploader(
#         "Upload background image", type=["png", "jpg", "jpeg", "tif"]
#     )
#     if uploaded_file is not None:
#         img = Image.open(uploaded_file).convert("RGBA")
#         st.session_state["bg_image"] = img
#         st.session_state["img_width"], st.session_state["img_height"] = img.size

#     if st.button("Clear all boxes"):
#         st.session_state["boxes"] = []

# # chart lives in the right column and reruns only as a fragment
# with right:
#     image_with_boxes()

# # Sidebar for bulk extraction
# with st.sidebar:
#     st.header("Box tools")

#     if st.button("Extract all boxes"):
#         # This button causes a *full* rerun when clicked – but you click it rarely.
#         st.subheader("All box coordinates")
#         st.json(st.session_state["boxes"])
