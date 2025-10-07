# panels/downloads.py
import streamlit as st

from helpers.state_ops import ordered_keys
from helpers.densenet_functions import load_labeled_patches_from_session
from helpers.upload_download_functions import (
    build_model_artifacts_zip,
    build_densenet_artifacts_zip,
    build_masks_images_zip,
    build_patchset_zip_from_session,
    _build_cell_metrics_zip,
)


def render_main():

    if not ordered_keys():
        st.info("Upload data and label masks first.")
        return False

    st.markdown("## Downloads")

    images = st.session_state.get("images", {})
    ok = ordered_keys() if images else []

    with st.expander("Download models and performance metrics", expanded=False):

        def _cellpose_zip(fingerprint: tuple) -> bytes:
            return build_model_artifacts_zip("cellpose") or b""

        def _densenet_zip(fingerprint: tuple) -> bytes:
            return build_densenet_artifacts_zip() or b""

        # lightweight fingerprints so cache invalidates when inputs change
        cp_df = st.session_state.get("cp_grid_results_df")
        cp_fp = (
            len(st.session_state.get("cellpose_model_bytes") or b""),
            len(st.session_state.get("cp_losses_png") or b""),
            len(st.session_state.get("cp_compare_iou_png") or b""),
            tuple(getattr(cp_df, "shape", (0, 0))),
        )
        dn_fp = (
            len(st.session_state.get("densenet_model_bytes") or b""),
            len(st.session_state.get("densenet_plot_losses_png") or b""),
            len(st.session_state.get("densenet_plot_confusion_png") or b""),
        )

        c1, c2 = st.columns(2)

        with c1:
            st.download_button(
                "Download Cellpose artifacts (.zip)",
                data=_cellpose_zip(cp_fp),
                file_name="cellpose_artifacts.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_cellpose_zip",
            )

        with c2:
            st.download_button(
                "Download DenseNet artifacts (.zip)",
                data=_densenet_zip(dn_fp),
                file_name="densenet_artifacts.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_densenet_zip",
            )

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

        ps = int(st.session_state.get("densenet_patch_size", 64))
        X_tmp, _, _ = load_labeled_patches_from_session(patch_size=ps)
        has_patches = X_tmp.shape[0] > 0
        if not has_patches:
            st.info("No labeled patches available from the current session.")
        pz = patch_zip(ps) if has_patches else b""
        st.download_button(
            "Build & Download patch set (.zip)",
            pz,
            "cell_patches.zip",
            "application/zip",
            disabled=not pz,
            use_container_width=True,
            key="dl_patch_zip_single",
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
