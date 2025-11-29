import streamlit as st
from panels import mask_editing_panel
from helpers.state_ops import ordered_keys, set_current_by_index


# Warning if no images have been uploaded yet
if st.session_state["images"] == {}:
    st.warning("⚠️ Please upload an image on the 'Upload Models and Data' tab first.")
    st.stop()

col1, col2 = st.columns([2, 5])
with col1:

    with st.container(border=True, height=770):

        ok = ordered_keys()
        names = [st.session_state.images[k]["name"] for k in ok]
        reck = st.session_state.current_key
        rec_idx = ok.index(reck) if reck in ok else 0

        st.info(f"**Image {rec_idx+1}/{len(ok)}:** {names[rec_idx]}")

        # --- Navigation buttons ---
        nav_col1, nav_col2, nav_col3 = st.columns([1, 4, 1])

        def go_to_prev():
            set_current_by_index(rec_idx - 1)
            st.rerun()

        def go_to_next():
            set_current_by_index(rec_idx + 1)
            st.rerun()

        with nav_col1:

            if st.button("◀", use_container_width=True):
                go_to_prev()

        with nav_col2:
            # # slider to move between images
            # jump = st.slider(
            #     "Image index",
            #     1,
            #     len(ok),
            #     value=st.session_state["current_key"],
            #     key="slider_jump",
            #     label_visibility="collapsed",
            # )
            # if (jump - 1) != rec_idx:
            #     set_current_by_index(jump - 1)
            #     jump = st.session_state["current_key"]

            st.caption("placeholder for image index slider")

        with nav_col3:

            if st.button("▶", use_container_width=True):
                go_to_next()

        # --- Toggles for overlay and normalization ---
        inner_col1, inner_col2 = st.columns([1, 3])
        with inner_col1:
            # toggle to show/hide masks overlay
            show_overlay_toggle = st.toggle(
                "Show masks",
                key="show_overlay_w",
                value=st.session_state.get("show_overlay", True),
            )
            st.session_state["show_overlay"] = show_overlay_toggle

        with inner_col2:
            # toggle to normalize background image
            normalize_image_toggle = st.toggle(
                "Normalize image",
                key="show_normalized_w",
                value=st.session_state.get("show_normalized", True),
            )
            st.session_state["show_normalized"] = normalize_image_toggle

        # Tabs for editing and classifying masks
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
