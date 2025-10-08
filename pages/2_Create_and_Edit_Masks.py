import streamlit as st
from boot import common_boot
from panels import mask_editing_panel

common_boot()

st.title("🎭 Create and Edit Segmentation Masks")

st.divider()

# Page-specific sidebar
with st.sidebar:
    mask_editing_panel.render_sidebar(key_ns="side")

# Page main content
mask_editing_panel.render_main(key_ns="edit")
