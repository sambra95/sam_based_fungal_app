import streamlit as st
from boot import common_boot
from panels import fine_tune_panel

common_boot()

st.title("🧠 Fine Tune Segmentation and Classification Models")

st.divider()

# Main content (your two sections)
fine_tune_panel.render_cellpose_train_panel()
fine_tune_panel.render_densenet_train_panel()
