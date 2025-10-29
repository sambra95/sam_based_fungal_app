# panels/downloads.py
import streamlit as st

from helpers.state_ops import ordered_keys
from helpers.densenet_functions import load_labeled_patches_from_session
from helpers.upload_download_functions import (
    build_masks_images_zip,
    build_patchset_zip_from_session,
    _build_cell_metrics_zip,
)


def render_main():

    if not ordered_keys():
        st.info("Upload data and label masks first.")
        return False

    images = st.session_state.get("images", {})

    # Morphology plots (expects files or (name,bytes) in session_state['morphology_plots'])
    with st.expander(
        "Download class morphology plots and image cells counts", expanded=False
    ):

        # --- single button ---
        st.download_button(
            "Download cell metrics (.zip)",
            data=_build_cell_metrics_zip(
                tuple(st.session_state.get("analysis_labels") or ())
            ),
            file_name="cell_metrics.zip",
            mime="application/zip",
            use_container_width=True,
            key="dl_cell_metrics_zip",
        )
