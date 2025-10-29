import streamlit as st
from boot import common_boot
from panels import upload_panel

common_boot()

# Main area
upload_panel.render_main()
