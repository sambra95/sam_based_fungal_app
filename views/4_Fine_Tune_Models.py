import streamlit as st
from boot import common_boot
from panels import fine_tune_panel

common_boot()

col1, col2 = st.columns([1, 1])

# Main content
with col1:
    with st.container(border=True, height=475):
        fine_tune_panel.render_cellpose_train_panel()
with col2:
    with st.container(border=True, height=475):
        fine_tune_panel.render_densenet_train_panel()


st.divider()
st.header("Training Results:")
cellpose_tab, densenet_tab = st.tabs(
    ["Cellpose Training Results", "Densenet Training Results"]
)

with cellpose_tab:
    fine_tune_panel.show_cellpose_training_plots(height=700)
with densenet_tab:
    fine_tune_panel.show_densenet_training_plots(height=700)
