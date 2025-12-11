# panels/edit_masks.py
import streamlit as st

from helpers.state_ops import ordered_keys
from helpers.mask_editing_functions import (
    render_cellpose_hyperparameters_fragment,
    render_box_tools_fragment,
    render_mask_tools_fragment,
    render_display_and_interact_fragment,
)
from helpers.classifying_functions import (
    classify_actions_fragment,
    class_selection_fragment,
    class_manage_fragment,
)
from helpers.cellpose_functions import (
    segment_current_and_refresh,
    batch_segment_and_refresh,
    segment_current_and_refresh_cellpose_sam,
    batch_segment_current_and_refresh_cellpose_sam,
)

from helpers.upload_download_functions import (
    build_masks_images_zip,
)

# ---------- Rendering functions ----------


def render_segment_sidebar(*, key_ns: str = "side"):
    with st.container(border=True):
        st.subheader("Segmentation controls:")

        # render cellpose controls
        with st.popover(
            "Predict masks for image",
            use_container_width=True,
            help="Segment cells using the loaded Cellpose model.",
            type="primary",
        ):

            model_family = st.selectbox(
                "Select model", ["Cellpose4", "Fine-tuned Model"]
            )

            col1, col2 = st.columns(2)

            if model_family == "Cellpose4":

                with col1:

                    if st.button(
                        "Generate",
                        use_container_width=True,
                        key="segment_image_SAM",
                    ):
                        segment_current_and_refresh_cellpose_sam()

                with col2:
                    if st.button(
                        "Batch generate",
                        use_container_width=True,
                        key="batch_segment_image_sam",
                        help="Segment all uploaded images with Cellpose.",
                    ):
                        batch_segment_current_and_refresh_cellpose_sam()

            else:

                with col1:

                    if st.button(
                        "Generate",
                        use_container_width=True,
                        key="segment_image",
                        help="Segment this image with Cellpose.",
                        disabled=st.session_state["cellpose_model_bytes"] == None,
                    ):
                        segment_current_and_refresh()
                with col2:
                    if st.button(
                        "Batch generate",
                        use_container_width=True,
                        key="batch_segment_image",
                        help="Segment all uploaded images with Cellpose.",
                        disabled=st.session_state["cellpose_model_bytes"] == None,
                    ):
                        batch_segment_and_refresh()

            st.caption("Change hyperparameters to increase accuracy:")

            with st.expander(
                "Cellpose hyperparameters",
            ):
                render_cellpose_hyperparameters_fragment()

        # render SAM2 controls
        with st.popover(
            "Predict masks from boxes",
            use_container_width=True,
            help="Draw boxes and click segment to use SAM2 to segment individual cells.",
            type="primary",
        ):
            render_box_tools_fragment(key_ns)

        # section for selecting tools for directly adding/removing masks
        render_mask_tools_fragment(key_ns)


def render_classify_sidebar(*, key_ns: str = "side"):

    with st.container(border=True):
        st.markdown("### Classification controls:")

        with st.popover(
            label="Manage Labels", use_container_width=True, type="primary"
        ):
            class_manage_fragment(key_ns)  # add/delete/rename

        # Action buttons to classify cells with Densenet
        with st.popover(
            "Classify cells with Densenet", use_container_width=True, type="primary"
        ):

            classify_actions_fragment()

        class_selection_fragment()


def render_main(*, key_ns: str = "edit"):

    render_display_and_interact_fragment(key_ns=key_ns, scale=1.5)


def render_download_button():
    if not ordered_keys():
        st.info("Upload data and label masks first.")
        return False

    images = st.session_state.get("images", {})
    ok = ordered_keys() if images else []

    with st.container(border=True):
        with st.popover(
            label="Download options", use_container_width=True, type="primary"
        ):
            include_overlay = st.checkbox(
                "Include colored mask overlays", True, key="dl_include_overlay"
            )
            include_counts = st.checkbox(
                "Overlay per-image class counts", False, key="dl_include_counts"
            )
            st.checkbox(
                "Normalize downloaded images", False, key="dl_normalize_download"
            )

            include_patches = st.checkbox(
                "Include cell patch images", False, key="dl_include_patches"
            )

            include_summary = st.checkbox(
                "Include table of per image cell counts",
                True,
                key="dl_include_summary",
            )

            # 🔹 Only build the dataset when the user actually clicks the button
            if st.button(
                "Prepare annotated images for download",
                use_container_width=True,
                type="primary",
            ):
                mz = build_masks_images_zip(
                    images,
                    ok,
                    include_overlay,
                    include_counts,
                    include_patches,
                    include_summary,
                )
                st.download_button(
                    "Download dataset",
                    mz,
                    "masks_and_images.zip",
                    "application/zip",
                    use_container_width=True,
                    type="primary",
                )
