# panels/edit_masks.py
import streamlit as st

from helpers.state_ops import ordered_keys
from helpers.mask_editing_functions import (
    cellpose_actions_fragment,
    box_tools_fragment,
    mask_tools_fragment,
    display_and_interact_fragment,
)
from helpers.classifying_functions import (
    classify_actions_fragment,
    class_selection_fragment,
    class_manage_fragment,
)


# ---------- Rendering functions ----------


def render_segment_sidebar(*, key_ns: str = "side"):
    st.markdown("### Create and edit cell masks:")
    cellpose_actions_fragment()
    with st.expander("Segment cells with SAM2", expanded=True):
        box_tools_fragment(key_ns)
    with st.expander("Manually segment cells", expanded=True):
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
    with st.container(border=True):
        class_manage_fragment(key_ns)  # add/delete/rename


def render_main(*, key_ns: str = "edit"):

    display_and_interact_fragment(key_ns=key_ns, mode_ns="side", scale=1.5)
