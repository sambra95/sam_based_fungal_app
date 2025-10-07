import streamlit as st
from boot import common_boot
from panels import classify_cells_panel

common_boot()

st.title("🧬 Classify Segmentation Masks")

st.divider()

with st.sidebar:
    classify_cells_panel.render_sidebar(key_ns="side")

classify_cells_panel.render_main(key_ns="classify")
