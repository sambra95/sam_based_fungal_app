import streamlit as st
from helpers.cellpose_functions import finetune_cellpose_from_records


def render_sidebar():

    st.radio(
        "Select model to fine tune:",
        ["cyto", "cyto2", "cyto3"],
        key=f"model_to_fine_tune",
        horizontal=True,
    )

    if st.button(
        "Fine tune Cellpose on uploaded data",
        key="btn_segment_cellpose",
        use_container_width=True,
        help="Check all desired cells have masks before training.",
    ):

        finetune_cellpose_from_records(
            recs=st.session_state["images"],
        )


def render_main():
    None
