import streamlit as st
from boot import common_boot
from panels import mask_editing_panel

common_boot()

st.write(
    """
    <style>
    /* Target the tab text container */
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
        font-size: 20px;
        font-weight: 600;
        text-align: center;      /* Center text */
        padding: 10px 30px;      /* Add vertical + horizontal space */
        margin: 5px 10px;        /* Add space around the tab text */
    }

    /* Optional: style the tabs themselves for better spacing */
    button[data-baseweb="tab"] {
        justify-content: center; /* Center tab content */
        padding: 10px 20px;      /* Increase clickable area */
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Page-specific sidebar

col1, col2 = st.columns([2, 5])
with col1:
    st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)
    with st.container(border=True):

        # common sidebar section for navigating between images
        listTabs = ["Segment Cells", "Classify Cells"]
        whitespace = 9
        editing_tab, classifying_tab = st.tabs(["Segment Cells", "Classify Cells"])
        with editing_tab:
            mask_editing_panel.render_segment_sidebar(key_ns="edit_side")
        with classifying_tab:
            mask_editing_panel.render_classify_sidebar(key_ns="classify_side")

        mask_editing_panel.render_download_button()

# Page main content
with col2:
    mask_editing_panel.render_main(key_ns="edit")
