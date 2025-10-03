# panels/edit_masks.py
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates
from helpers.upload_download_functions import zip_all_masks
from helpers.cellpose_functions import (
    _has_cellpose_model,
    segment_rec_with_cellpose,
)
from helpers.state_ops import ordered_keys, set_current_by_index, current
from helpers.mask_editing_functions import (
    _run_sam2_on_boxes,
    is_unique_box,
    boxes_to_fabric_rects,
    draw_boxes_overlay,
    polygon_to_mask,
    composite_over_by_class,
    integrate_new_mask,
)
from helpers.classifying_functions import classes_map_from_labels, palette_from_emojis


def render_sidebar(*, key_ns: str = "side"):
    """
    Renders the sidebar controls for 'Create and Edit Masks'.
    """

    ok = ordered_keys()
    if not ok:
        st.warning("Upload an image in **Upload data** first.")
        # read from state so caller can no-op safely

    # ⬇️ get the current record (this was missing)
    rec = current()

    names = [st.session_state.images[k]["name"] for k in ok]
    reck = st.session_state.current_key
    rec_idx = ok.index(reck) if reck in ok else 0
    st.markdown(f"**Image {rec_idx+1}/{len(ok)}:** {names[rec_idx]}")

    c1, c2 = st.columns(2)
    if c1.button("◀ Prev", key=f"{key_ns}_prev", use_container_width=True):
        set_current_by_index(rec_idx - 1)
        st.rerun()
    if c2.button("Next ▶", key=f"{key_ns}_next", use_container_width=True):
        set_current_by_index(rec_idx + 1)
        st.rerun()

    st.toggle("Show mask overlay", key="show_overlay", value=True)

    st.divider()

    st.markdown("### Create and edit cell masks:")

    st.radio(
        "Select action to perform:",
        ["Draw box", "Remove box", "Draw mask", "Remove mask"],
        key=f"{key_ns}_interaction_mode",
        horizontal=True,
    )

    # segment current image
    if st.button(
        "Segment current image with Cellpose",
        key="btn_segment_cellpose",
        disabled=not _has_cellpose_model(),
        use_container_width=True,
        help=(
            "Warning: this action will reset current mask labels."
            if _has_cellpose_model()
            else "Upload model"
        ),
    ):
        segment_rec_with_cellpose(current())

    # batch segment all images
    if st.button(
        "Batch segment all images with Cellpose",
        key="btn_batch_segment_cellpose",
        disabled=not _has_cellpose_model(),
        use_container_width=True,
        help=(
            "Warning: this action will reset current mask labels."
            if _has_cellpose_model()
            else "Upload model"
        ),
    ):
        n = len(ordered_keys())
        pb = st.progress(0.0, text="Starting…")
        for i, k in enumerate(ordered_keys(), 1):
            segment_rec_with_cellpose(st.session_state.images.get(k))
            pb.progress(i / n, text=f"Segmented {i}/{n}")
        pb.empty()

    # --- Box utilities
    # --- boxes are used to guide cell mask predictions from SAM2
    row = st.container()
    c1, c2, c3 = row.columns([1, 1, 1])

    # resets the boxes list of the record to an empty list
    if c1.button("Clear boxes", use_container_width=True, key=f"{key_ns}_clear_boxes"):
        rec["boxes"] = []
        st.session_state["pred_canvas_nonce"] = (
            st.session_state.get("pred_canvas_nonce", 0) + 1
        )
        st.rerun()

    # removes the last bix from the box list of the record
    if c2.button("Remove last box", use_container_width=True, key=f"{key_ns}_undo_box"):
        if rec["boxes"]:
            rec["boxes"].pop()
            st.session_state["pred_canvas_nonce"] = (
                st.session_state.get("pred_canvas_nonce", 0) + 1
            )
            st.rerun()

    # masks are predicted for any boxes displayed on the image
    if c3.button(
        "Predict masks in boxes", use_container_width=True, key=f"{key_ns}_predict"
    ):
        new_masks = _run_sam2_on_boxes(rec)
        for mask in new_masks:
            inst, new_id = integrate_new_mask(rec["masks"], mask)
            if new_id is not None:
                rec["masks"] = inst
                rec.setdefault("labels", {})[int(new_id)] = rec["labels"].get(
                    int(new_id), None
                )
                # added_any = True
        rec["boxes"] = []
        st.session_state["pred_canvas_nonce"] = (
            st.session_state.get("pred_canvas_nonce", 0) + 1
        )
        st.rerun()

    # --- Mask utilities
    row2 = st.container()
    c4, c5 = row2.columns([1, 1])

    # Remove all masks from the record
    if c4.button("Clear masks", use_container_width=True, key=f"{key_ns}_clear_masks"):
        rec["masks"] = np.zeros((rec["H"], rec["W"]), dtype=np.uint16)
        rec["labels"] = {}
        rec["last_click_xy"] = None
        st.rerun()

    # remove the last added mask from the list
    if c5.button(
        "Remove last mask", use_container_width=True, key=f"{key_ns}_undo_mask"
    ):
        max_id = int(rec["masks"].max())
        if max_id > 0:
            rec["masks"][rec["masks"] == max_id] = 0
            rec.setdefault("labels", {}).pop(max_id, None)

        st.session_state["pred_canvas_nonce"] = (
            st.session_state.get("pred_canvas_nonce", 0) + 1
        )
        st.rerun()

    # Download

    downloadable = any(
        getattr(rec["masks"], "size", 0) > 0 for rec in st.session_state.images.values()
    )
    data_bytes = zip_all_masks(st.session_state.images, ok) if downloadable else b""
    st.download_button(
        "Download all masks (.zip)",
        data=data_bytes,
        file_name="masks.zip",
        mime="application/zip",
        disabled=not downloadable,
        use_container_width=True,
        key=f"{key_ns}_dl",
    )


def render_main(
    *,
    key_ns: str = "edit",
):
    """
    Render the Create/Edit Masks UI for the current image record.
    Required:
      rec          : image record dict (from st.session_state.images[...])
      mode         : one of ["Draw mask","Remove mask","Draw box","Remove box"]
    Optional:
      scale        : float (display scaling)
      key_ns       : widget key namespace to avoid collisions
    """
    rec = current()
    if rec is None:
        st.warning("Upload an image in **Upload data** first.")

    scale = (
        1.5  # hard coded scaling factor so that images show nicely on my laptop screen
    )
    disp_w, disp_h = int(rec["W"] * scale), int(rec["H"] * scale)

    labels = st.session_state.setdefault("all_classes", ["Remove label"])
    palette = palette_from_emojis(labels)
    classes_map = classes_map_from_labels(rec["masks"], rec["labels"])
    display_img = composite_over_by_class(
        rec["image"], rec["masks"], classes_map, palette, alpha=0.35
    )

    # PIL safety: ensure uint8 RGB before resizing (won't work for uint16)
    display_for_ui = np.array(
        Image.fromarray(display_img.astype(np.uint8)).resize(
            (disp_w, disp_h), Image.BILINEAR
        )
    )

    # ensure nonces exist locally to avoid KeyErrors when called from this module
    # ? Where is this set interactively? which widget has that key?
    if st.session_state["side_interaction_mode"] == "Draw mask":
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
                if obj.get("type") not in ("path", "polygon"):
                    continue

                # 1) display-size raster
                mask_disp = polygon_to_mask(obj, disp_h, disp_w).astype(
                    np.uint16
                )  # (disp_h,disp_w) in {0,1}

                # 2) resize to full resolution (nearest keeps labels crisp)
                mask_full = (
                    np.array(
                        Image.fromarray(mask_disp).resize(
                            (rec["W"], rec["H"]), Image.NEAREST
                        ),
                        dtype=np.uint16,
                    )
                    > 0
                )  # (H,W) bool

                # 3) integrate into label image
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

    elif st.session_state["side_interaction_mode"] == "Draw box":
        bg = Image.fromarray(display_for_ui).convert("RGBA")
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

    # remove masks by clicking on them
    elif st.session_state["side_interaction_mode"] == "Remove mask":
        click = streamlit_image_coordinates(
            display_for_ui, key=f"{key_ns}_rm", width=disp_w
        )
        if not click:
            pass
        else:
            x, y = int(round(int(click["x"]) / scale)), int(
                round(int(click["y"]) / scale)
            )
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
                        inst[inst == iid] = 0  # remove clicked instance

                        # relabel to 0,1..K (contiguous) and remap labels dict
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

    # "Remove boxes by clicking on them"
    elif st.session_state["side_interaction_mode"] == "Remove box":

        # render the image and any boxes
        overlay = draw_boxes_overlay(
            display_img, rec["boxes"], alpha=0.25, outline_px=2
        )
        overlay_for_ui = np.array(
            Image.fromarray(overlay).resize((disp_w, disp_h), Image.BILINEAR)
        )
        # record if there has been a click somewhere
        click = streamlit_image_coordinates(
            overlay_for_ui, key=f"{key_ns}_pred_click_remove", width=disp_w
        )
        # if there is a click, check if the click is in any of the boxes, if so, remove the box
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
