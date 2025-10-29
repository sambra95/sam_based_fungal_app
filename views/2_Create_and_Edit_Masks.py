import streamlit as st
from boot import common_boot
from panels import mask_editing_panel

common_boot()

# Page-specific sidebar
with st.sidebar:

    # common sidebar section for navigating between images
    editing_tab, classifying_tab = st.tabs(["Segment My Cells", "Classify My Cells"])
    with editing_tab:
        mask_editing_panel.render_segment_sidebar(key_ns="edit_side")
    with classifying_tab:
        mask_editing_panel.render_classify_sidebar(key_ns="classify_side")

    mask_editing_panel.render_download_button()

# Page main content
mask_editing_panel.render_main(key_ns="edit")
