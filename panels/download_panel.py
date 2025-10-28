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
    ok = ordered_keys() if images else []

    # Masks & images (with overlay options)
    def masks_images_zip(images, ok, overlay, counts):
        return build_masks_images_zip(images, ok, overlay, counts) or b""

    def patch_zip(ps):
        return build_patchset_zip_from_session(patch_size=ps) or b""

    with st.expander(
        "Download masks, labelled images, cell patches and labels", expanded=True
    ):
        c1, c2 = st.columns(2)
        overlay = c1.checkbox(
            "Include colored mask overlays", True, key="dl_include_overlay"
        )
        counts = c2.checkbox(
            "Overlay per-image class counts", False, key="dl_include_counts"
        )

        if not ok:
            st.info("No images in memory. Upload data first.")
        mz = masks_images_zip(images, ok, overlay, counts) if ok else b""
        st.download_button(
            "Build & Download masks + images (.zip)",
            mz,
            "masks_and_images.zip",
            "application/zip",
            disabled=not mz,
            use_container_width=True,
            key="dl_masks_images_zip_single",
        )

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
