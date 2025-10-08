# app.py
import streamlit as st
from boot import common_boot

st.set_page_config(page_title="Mycoscope", page_icon="🧬", layout="wide")

common_boot()

pages = {
    "Choose a task from the workflow:": [
        st.Page("pages/1_Upload_data.py", title="Uploads", icon="📥"),
        st.Page("pages/2_Create_and_Edit_Masks.py", title="Segment Cells", icon="🎭"),
        st.Page("pages/3_Classify_Cells.py", title="Classify Cells", icon="🧬"),
        st.Page("pages/4_Fine_Tune_Models.py", title="Train ML Models", icon="🧠"),
        st.Page("pages/5_Cell_Metrics.py", title="Analyze Cell Groups", icon="📊"),
        st.Page("pages/6_Downloads.py", title="Downloads", icon="⬇️"),
    ],
}

pg = st.navigation(pages)
pg.run()
