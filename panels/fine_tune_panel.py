import streamlit as st
from helpers.cellpose_functions import (
    finetune_cellpose_from_records,
    _plot_losses,
    compare_models_mean_iou_plot,
)


def render_sidebar():

    st.radio(
        "Select model to fine tune:",
        ["cyto", "cyto2", "cyto3", "scratch"],
        key=f"model_to_fine_tune",
        horizontal=True,
    )

    if st.button(
        "Fine tune Cellpose on uploaded data",
        key="btn_segment_cellpose",
        use_container_width=True,
        help="Check all desired cells have masks before training.",
    ):

        st.session_state["train_losses"], st.session_state["test_losses"] = (
            finetune_cellpose_from_records(
                recs=st.session_state["images"],
            )
        )


def render_main():
    if len(st.session_state["train_losses"]) != 0:
        _plot_losses(st.session_state["train_losses"], st.session_state["test_losses"])

        images = [rec["image"] for rec in st.session_state["images"].values()]
        masks = [rec["masks"] for rec in st.session_state["images"].values()]
        compare_models_mean_iou_plot(
            images, masks, base_model_name=st.session_state["model_to_fine_tune"]
        )
