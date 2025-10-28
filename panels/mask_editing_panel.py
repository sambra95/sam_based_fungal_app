# panels/edit_masks.py
import streamlit as st

from helpers.state_ops import ordered_keys
from helpers.mask_editing_functions import (
    cellpose_hyperparameters_fragment,
    box_tools_fragment,
    mask_tools_fragment,
    display_and_interact_fragment,
    _segment_current_and_refresh,
    _batch_segment_and_refresh,
)
from helpers.classifying_functions import (
    classify_actions_fragment,
    class_selection_fragment,
    class_manage_fragment,
)


# ---------- Rendering functions ----------


def render_segment_sidebar(*, key_ns: str = "side"):
    with st.container(border=True):
        st.subheader("Segment with Cellpose:")

        col1, col2 = st.columns([1, 1])

        with col1:

            if st.button(
                "Segment with Cellpose",
                use_container_width=True,
                key="segment_image",
                help="Segment this image with Cellpose.",
            ):
                _segment_current_and_refresh()
        with col2:
            if st.button(
                "Batch segment with Cellpose",
                use_container_width=True,
                key="batch_segment_image",
                help="Segment all uploaded images with Cellpose.",
            ):
                _batch_segment_and_refresh()

        with st.popover(
            "Edit Cellpose Hyperparameters",
            use_container_width=True,
            help="Change Cellpose prediction parameters here.",
        ):
            cellpose_hyperparameters_fragment()

        st.subheader("Segment individual cells:")

        with st.popover(
            "Segment cells with SAM2",
            use_container_width=True,
            help="Draw boxes and click segment to use SAM2 to segment individual cells.",
        ):
            box_tools_fragment(key_ns)

        with st.popover(
            "Manually segment cells",
            use_container_width=True,
            help="Manually add and remove cell masks.",
        ):
            mask_tools_fragment(key_ns)


def render_classify_sidebar(*, key_ns: str = "side"):
    ok = ordered_keys()
    if not ok:
        st.warning("Upload an image in **Upload data** first.")
        return

    with st.container(border=True):
        st.markdown("### Classify cells with DenseNet")
        classify_actions_fragment()
        st.markdown("### Classify cells manually")
        class_selection_fragment()
        with st.popover(label="Manage Classes", use_container_width=True):
            class_manage_fragment(key_ns)  # add/delete/rename


def render_main(*, key_ns: str = "edit"):

    display_and_interact_fragment(key_ns=key_ns, mode_ns="side", scale=1.5)
