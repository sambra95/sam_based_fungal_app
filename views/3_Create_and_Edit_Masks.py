import streamlit as st
from panels import mask_editing_panel


# Page-specific sidebar

col1, col2 = st.columns([2, 5])
with col1:
    with st.container(border=True):

        # common sidebar section for navigating between images
        listTabs = ["Segment Cells", "Classify Cells"]
        whitespace = 9
        editing_tab, classifying_tab = st.tabs(["Segment Cells", "Classify Cells"])
        with editing_tab:
            mask_editing_panel.render_segment_sidebar(key_ns="edit_side")
        with classifying_tab:
            mask_editing_panel.render_classify_sidebar(key_ns="classify_side")

        mask_editing_panel.render_download_button()

# Page main content
with col2:
    mask_editing_panel.render_main(key_ns="edit")
