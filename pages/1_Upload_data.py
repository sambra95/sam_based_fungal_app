import streamlit as st
from boot import common_boot
from panels import upload_panel

common_boot()

st.title("📥 Upload data and models")

st.divider()

# Main area
upload_panel.render_main()
