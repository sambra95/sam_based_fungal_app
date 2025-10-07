# panels/edit_masks.py
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates

from helpers.cellpose_functions import _has_cellpose_model, segment_rec_with_cellpose
from helpers.state_ops import ordered_keys, set_current_by_index, current
from helpers.mask_editing_functions import (
    _run_sam2_on_boxes,
    is_unique_box,
    boxes_to_fabric_rects,
    draw_boxes_overlay,
    polygon_to_mask,
    composite_over_by_class,
    integrate_new_mask,
    _segment_current_and_refresh,
    _batch_segment_and_refresh,
    _reset_cellpose_hparams_to_defaults,
)
from helpers.classifying_functions import classes_map_from_labels, palette_from_emojis


# ---------- Small utilities ----------


def _ensure_nonces():
    st.session_state.setdefault("edit_canvas_nonce", 0)
    st.session_state.setdefault("pred_canvas_nonce", 0)
    st.session_state.setdefault("side_interaction_mode", "Draw box")


# --- replace your _image_display with this ---
def _image_display(rec, scale):
    disp_w, disp_h = int(rec["W"] * scale), int(rec["H"] * scale)

    M = rec.get("masks")
    has_instances = isinstance(M, np.ndarray) and M.ndim == 2 and M.any()

    if st.session_state.get("show_overlay", False) and has_instances:
        labels = st.session_state.setdefault("all_classes", ["Remove label"])
        palette = palette_from_emojis(labels)
        classes_map = classes_map_from_labels(rec["masks"], rec["labels"])
        base_img = composite_over_by_class(
            rec["image"], rec["masks"], classes_map, palette, alpha=0.35
        )
    else:
        base_img = rec["image"]  # need unaltered image to draw boxes on

    display_for_ui = np.array(
        Image.fromarray(base_img.astype(np.uint8)).resize(
            (disp_w, disp_h), Image.BILINEAR
        )
    )
    return base_img, display_for_ui, disp_w, disp_h


# ---------- Sidebar: navigation ----------


# @st.fragment
def nav_fragment(key_ns="side"):
    ok = ordered_keys()
    if not ok:
        st.warning("Upload an image in **Upload data** first.")
        return

    rec = current()
    names = [st.session_state.images[k]["name"] for k in ok]
    reck = st.session_state.current_key
    rec_idx = ok.index(reck) if reck in ok else 0
    st.markdown(f"**Image {rec_idx+1}/{len(ok)}:** {names[rec_idx]}")

    c1, c2 = st.columns(2)
    if c1.button("◀ Prev", key=f"{key_ns}_prev", use_container_width=True):
        set_current_by_index(rec_idx - 1)
        # st.rerun()
    if c2.button("Next ▶", key=f"{key_ns}_next", use_container_width=True):
        set_current_by_index(rec_idx + 1)
        # st.rerun()

    st.toggle("Show mask overlay", key="show_overlay", value=True)


# ---------- Sidebar: interaction mode ----------


@st.fragment
def interaction_mode_fragment(ns="side"):
    # set a default BEFORE the widget is created
    st.session_state.setdefault(f"{ns}_interaction_mode", "Draw box")

    mode = st.radio(
        "Select action to perform:",
        ["Draw box", "Remove box", "Draw mask", "Remove mask"],
        key=f"{ns}_interaction_mode",
        horizontal=True,
    )
    st.caption(f"Mode: {mode}")


# ---------- Sidebar: Cellpose actions ----------


import streamlit as st


@st.fragment
def cellpose_actions_fragment():
    # --- Hyperparameters (collapsible) ---
    with st.expander("Cellpose hyperparameters", expanded=False):
        # We use a small form so changing values doesn't trigger reruns mid-typing
        with st.form("cellpose_hparams_form", clear_on_submit=False):
            # Channels (two ints)
            c1 = st.number_input(
                "Channel 1",
                value=st.session_state.get("cp_ch1", 0),
                step=1,
                format="%d",
                key="cp_ch1",
            )
            c2 = st.number_input(
                "Channel 2",
                value=st.session_state.get("cp_ch2", 0),
                step=1,
                format="%d",
                key="cp_ch2",
            )

            # Diameter: auto (None) or manual
            diam_mode = st.selectbox(
                "Diameter mode",
                ["Auto (None)", "Manual"],
                index=(
                    0
                    if st.session_state.get("cp_diam_mode", "Auto (None)")
                    == "Auto (None)"
                    else 1
                ),
                key="cp_diam_mode",
                help="Leave as Auto for Cellpose to estimate diameter, or set a manual value.",
            )
            diam_val = None
            if diam_mode == "Manual":
                diam_val = st.number_input(
                    "Manual diameter (pixels)",
                    min_value=0.0,
                    value=float(st.session_state.get("cp_diameter", 0.0) or 0.0),
                    step=1.0,
                    key="cp_diameter",
                )

            # Thresholds & size
            cellprob = st.number_input(
                "Cellprob threshold",
                value=float(st.session_state.get("cp_cellprob_threshold", -0.2)),
                step=0.1,
                key="cp_cellprob_threshold",
                help="Higher -> fewer cells. Default -0.2",
            )
            flowthr = st.number_input(
                "Flow threshold",
                value=float(st.session_state.get("cp_flow_threshold", 0.4)),
                step=0.1,
                key="cp_flow_threshold",
                help="Lower -> more permissive flows. Default 0.4",
            )
            min_size = st.number_input(
                "Minimum size (pixels)",
                value=int(st.session_state.get("cp_min_size", 0)),
                min_value=0,
                step=10,
                key="cp_min_size",
                help="Remove masks smaller than this area.",
            )

            cols = st.columns([1, 1])
            with cols[0]:
                submitted = st.form_submit_button(
                    "Apply changes", use_container_width=True
                )
            with cols[1]:
                if st.form_submit_button("Reset defaults", use_container_width=True):
                    _reset_cellpose_hparams_to_defaults()

        # sync diameter to None when Auto selected
        if st.session_state.get("cp_diam_mode", "Auto (None)") == "Auto (None)":
            st.session_state["cp_diameter"] = None


# ---------- Sidebar: Box utilities ----------


@st.fragment
def box_tools_fragment(key_ns="side"):
    rec = current()
    row = st.container()
    (
        c1,
        c2,
    ) = row.columns([1, 1])

    if c1.button("Draw box", use_container_width=True, key=f"{key_ns}_draw_boxes"):
        st.session_state[f"{key_ns}_interaction_mode"] = "Draw box"
        st.rerun()

    if c2.button("Remove box", use_container_width=True, key=f"{key_ns}_remove_boxes"):
        st.session_state[f"{key_ns}_interaction_mode"] = "Remove box"
        st.rerun()

    row = st.container()
    (
        c1,
        c2,
    ) = row.columns([1, 1])

    if c1.button(
        "Remove all boxes", use_container_width=True, key=f"{key_ns}_clear_boxes"
    ):
        rec["boxes"] = []
        st.session_state["pred_canvas_nonce"] += 1
        st.rerun()

    if c2.button("Remove last box", use_container_width=True, key=f"{key_ns}_undo_box"):
        if rec["boxes"]:
            rec["boxes"].pop()
            st.session_state["pred_canvas_nonce"] += 1
            st.rerun()

    if st.button("Predict Masks", use_container_width=True, key=f"{key_ns}_predict"):
        new_masks = _run_sam2_on_boxes(rec)
        for mask in new_masks:
            inst, new_id = integrate_new_mask(rec["masks"], mask)
            if new_id is not None:
                rec["masks"] = inst
                rec.setdefault("labels", {})[int(new_id)] = rec["labels"].get(
                    int(new_id), None
                )
        rec["boxes"] = []
        st.session_state["pred_canvas_nonce"] += 1
        st.session_state["edit_canvas_nonce"] += 1
        st.rerun()


# ---------- Sidebar: Mask utilities ----------


@st.fragment
def mask_tools_fragment(key_ns="side"):
    rec = current()
    row = st.container()
    c1, c2 = row.columns([1, 1])

    if c1.button("Draw mask", use_container_width=True, key=f"{key_ns}_draw_masks"):
        st.session_state[f"{key_ns}_interaction_mode"] = "Draw mask"
        st.rerun()

    if c2.button("Remove mask", use_container_width=True, key=f"{key_ns}_remove_masks"):
        st.session_state[f"{key_ns}_interaction_mode"] = "Remove mask"
        st.rerun()

    row = st.container()
    c1, c2 = row.columns([1, 1])

    if c1.button("Clear masks", use_container_width=True, key=f"{key_ns}_clear_masks"):
        rec["masks"] = np.zeros((rec["H"], rec["W"]), dtype=np.uint16)
        rec["labels"] = {}
        rec["last_click_xy"] = None
        st.session_state["edit_canvas_nonce"] += 1
        st.rerun()

    if c2.button(
        "Remove last mask", use_container_width=True, key=f"{key_ns}_undo_mask"
    ):
        max_id = int(rec["masks"].max())
        if max_id > 0:
            rec["masks"][rec["masks"] == max_id] = 0
            rec.setdefault("labels", {}).pop(max_id, None)
        st.session_state["pred_canvas_nonce"] += 1
        st.session_state["edit_canvas_nonce"] += 1
        st.rerun()


# ---------- Main: display + interaction canvas ----------


@st.fragment
def display_and_interact_fragment(key_ns="edit", mode_ns="side", scale=1.5):
    _ensure_nonces()
    rec = current()
    if rec is None:
        st.warning("Upload an image in **Upload data** first.")
        return

    base_img, display_for_ui, disp_w, disp_h = _image_display(rec, scale)
    mode = st.session_state.get(f"{mode_ns}_interaction_mode", "Draw box")

    # ---- Draw mask
    if mode == "Draw mask":
        bg = Image.fromarray(display_for_ui).convert("RGBA")
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 255, 0.30)",
            stroke_width=2,
            stroke_color="white",
            background_color="white",
            background_image=bg,
            update_streamlit=True,
            width=disp_w,
            height=disp_h,
            drawing_mode="freedraw",
            point_display_radius=3,
            initial_drawing=None,
            key=f"{key_ns}_canvas_edit_{st.session_state['edit_canvas_nonce']}",
        )
        if canvas_result.json_data:
            added_any = False
            for obj in canvas_result.json_data.get("objects", []):
                if obj.get("type") not in ("path", "polygon"):  # guard
                    continue
                # 1) display-size mask
                mask_disp = polygon_to_mask(obj, disp_h, disp_w).astype(np.uint16)
                # 2) back to full res
                mask_full = (
                    np.array(
                        Image.fromarray(mask_disp).resize(
                            (rec["W"], rec["H"]), Image.NEAREST
                        ),
                        dtype=np.uint16,
                    )
                    > 0
                )
                inst, new_id = integrate_new_mask(rec["masks"], mask_full)
                if new_id is not None:
                    rec["masks"] = inst
                    rec.setdefault("labels", {})[int(new_id)] = rec["labels"].get(
                        int(new_id), None
                    )
                    added_any = True
            if added_any:
                st.session_state["edit_canvas_nonce"] += 1
                st.rerun()

    # ---- Draw box
    elif mode == "Draw box":
        bg = Image.fromarray(display_for_ui).convert("RGB")
        initial_json = boxes_to_fabric_rects(rec["boxes"], scale=scale)
        num_initial = len(initial_json.get("objects", []))
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 255, 0.25)",
            stroke_width=2,
            stroke_color="white",
            background_color="white",
            background_image=bg,
            update_streamlit=True,
            width=disp_w,
            height=disp_h,
            drawing_mode="rect",
            point_display_radius=3,
            initial_drawing=initial_json,
            key=f"{key_ns}_canvas_pred_{st.session_state['pred_canvas_nonce']}",
        )
        if canvas_result.json_data:
            objs = canvas_result.json_data.get("objects", [])
            added_any = False
            for obj in objs[num_initial:]:
                if obj.get("type") != "rect":
                    continue
                left = float(obj.get("left", 0))
                top = float(obj.get("top", 0))
                width = float(obj.get("width", 0)) * float(obj.get("scaleX", 1.0))
                height = float(obj.get("height", 0)) * float(obj.get("scaleY", 1.0))
                # map from display to image coords
                x0 = int(round(left / scale))
                y0 = int(round(top / scale))
                x1 = int(round((left + width) / scale))
                y1 = int(round((top + height) / scale))
                x0 = max(0, min(rec["W"] - 1, x0))
                x1 = max(0, min(rec["W"], x1))
                y0 = max(0, min(rec["H"] - 1, y0))
                y1 = max(0, min(rec["H"], y1))
                if x1 < x0:
                    x0, x1 = x1, x0
                if y1 < y0:
                    y0, y1 = y1, y0
                box = (x0, y0, x1, y1)
                if is_unique_box(box, rec["boxes"]):
                    rec["boxes"].append(box)
                    added_any = True
            if added_any:
                st.rerun()

    # ---- Remove mask
    elif mode == "Remove mask":
        click = streamlit_image_coordinates(
            display_for_ui, key=f"{key_ns}_rm", width=disp_w
        )
        if click:
            x = int(round(int(click["x"]) / scale))
            y = int(round(int(click["y"]) / scale))
            if (
                0 <= x < rec["W"]
                and 0 <= y < rec["H"]
                and (x, y) != rec.get("last_click_xy")
            ):
                inst = rec.get("masks")
                if isinstance(inst, np.ndarray) and inst.ndim == 2 and inst.size:
                    iid = int(inst[y, x])
                    if iid > 0:
                        inst = inst.copy()
                        inst[inst == iid] = 0
                        vals, inv = np.unique(inst, return_inverse=True)
                        if vals.size > 1:
                            new_vals = np.zeros_like(vals)
                            nz = vals != 0
                            new_vals[nz] = np.arange(1, nz.sum() + 1, dtype=inst.dtype)
                            inst = new_vals[inv].reshape(inst.shape)
                            old_labels = rec.setdefault("labels", {})
                            remap = {
                                int(old): int(new)
                                for old, new in zip(vals, new_vals)
                                if old != 0 and new != 0
                            }
                            rec["labels"] = {
                                remap[oid]: old_labels.get(oid) for oid in remap
                            }
                        else:
                            rec["labels"] = {}
                        rec["masks"] = inst
                        rec["last_click_xy"] = (x, y)
                        st.rerun()

    elif mode == "Remove box":
        # draw on full-res base_img with original box coords
        overlay = draw_boxes_overlay(base_img, rec["boxes"], alpha=0.25, outline_px=2)
        overlay_for_ui = np.array(
            Image.fromarray(overlay).resize((disp_w, disp_h), Image.BILINEAR)
        )
        click = streamlit_image_coordinates(
            overlay_for_ui, key=f"{key_ns}_pred_click_remove", width=disp_w
        )
        if click:
            x = int(round(int(click["x"]) / scale))
            y = int(round(int(click["y"]) / scale))
            hits = [
                i
                for i, (x0, y0, x1, y1) in enumerate(rec["boxes"])
                if (x0 <= x < x1) and (y0 <= y < y1)
            ]
            if hits:
                rec["boxes"].pop(hits[-1])
                st.rerun()


# ---------- Rendering functions ----------


def render_sidebar(*, key_ns: str = "side"):
    nav_fragment(key_ns)
    st.divider()
    st.markdown("### Create and edit cell masks:")
    interaction_mode_fragment(ns=key_ns)
    cellpose_actions_fragment()
    with st.expander("Draw boxes and click predict to add masks", expanded=True):
        box_tools_fragment(key_ns)
    with st.expander("Manually draw and remove masks", expanded=True):
        mask_tools_fragment(key_ns)


def render_main(*, key_ns: str = "edit"):
    display_and_interact_fragment(key_ns=key_ns, mode_ns="side", scale=1.5)
