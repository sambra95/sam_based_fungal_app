import streamlit as st
from boot import common_boot
from panels import fine_tune_panel

common_boot()

col1, col2 = st.columns([1, 1])

cellpose_tab, densenet_tab = st.tabs(
    [
        "Train a Cellpose model to identify cells",
        "Train a Densenet model to classify cells",
    ]
)

with cellpose_tab:
    with st.container(border=True):
        fine_tune_panel.render_cellpose_train_panel()
    fine_tune_panel.show_cellpose_training_plots()
with densenet_tab:
    with st.container(border=True):
        fine_tune_panel._densenet_options()
        fine_tune_panel.densenet_train_fragment()
    fine_tune_panel.show_densenet_training_plots()
