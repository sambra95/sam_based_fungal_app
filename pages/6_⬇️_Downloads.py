import streamlit as st
from boot import common_boot
from panels import download_panel

common_boot()

st.title("⬇️ Download Datasets, Models and Visualizations")

st.divider()

download_panel.render_main()
