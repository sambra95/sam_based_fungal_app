import streamlit as st
from boot import common_boot
from panels import cell_metrics_panel

common_boot()

with st.container(border=True):
    cell_metrics_panel.render_plotting_options()

st.divider()

if st.button("Generate Plots"):
    cell_metrics_panel.render_plotting_main()
