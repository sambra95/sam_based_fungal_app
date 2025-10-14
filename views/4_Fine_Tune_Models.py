import streamlit as st
from boot import common_boot
from panels import fine_tune_panel

common_boot()

st.title("🧠 Fine Tune Segmentation and Classification Models")

st.divider()
col1, col2 = st.columns([1, 1])

# Main content (your two sections)
with col1:
    with st.container(border=True, height=620):
        fine_tune_panel.render_cellpose_train_panel()
with col2:
    with st.container(border=True, height=620):
        fine_tune_panel.render_densenet_train_panel()
