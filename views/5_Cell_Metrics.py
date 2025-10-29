import streamlit as st
from boot import common_boot
from panels import cell_metrics_panel

common_boot()

with st.sidebar:
    cell_metrics_panel.render_sidebar()  # if you have one

cell_metrics_panel.render_main()
