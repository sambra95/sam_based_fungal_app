# panels/edit_masks.py
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates
from helpers.mask_editing_functions import zip_all_masks


from helpers.state_ops import ordered_keys, set_current_by_index, current
from helpers.mask_editing_functions import (
    _resize_mask_nearest,
    _run_sam2_on_boxes,
    append_masks_to_rec,
    is_unique_box,
    boxes_to_fabric_rects,
    draw_boxes_overlay,
    polygon_to_mask,
    composite_over,
)


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

    st.toggle(
        "Show mask overlay",
        value=st.session_state[f"{key_ns}_show_overlay"],
        key=f"{key_ns}_show_overlay",
    )

    st.markdown("### Create and edit cell masks:")

    st.radio(
        "Select action to perform:",
        ["Draw box", "Remove box", "Draw mask", "Remove mask"],
        key=f"{key_ns}_interaction_mode",
    )

    # --- Box utilities
    row = st.container()
    c1, c2, c3 = row.columns([1, 1, 1])

    if c1.button("Clear boxes", use_container_width=True, key=f"{key_ns}_clear_boxes"):
        rec["boxes"] = []
        st.session_state["pred_canvas_nonce"] = (
            st.session_state.get("pred_canvas_nonce", 0) + 1
        )
        st.rerun()

    if c2.button("Remove last box", use_container_width=True, key=f"{key_ns}_undo_box"):
        if rec["boxes"]:
            rec["boxes"].pop()
            st.session_state["pred_canvas_nonce"] = (
                st.session_state.get("pred_canvas_nonce", 0) + 1
            )
            st.rerun()

    if c3.button("Predict masks", use_container_width=True, key=f"{key_ns}_predict"):
        new_masks = _run_sam2_on_boxes(rec)
        append_masks_to_rec(
            rec, new_masks
        )  # your existing function that mutates rec and reruns
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
        rec["masks"] = np.zeros((0, rec["H"], rec["W"]), dtype=np.uint8)
        rec["labels"] = []
        rec["last_click_xy"] = None
        st.rerun()

    # remove the last added mask from the list
    if c5.button(
        "Remove last mask", use_container_width=True, key=f"{key_ns}_undo_mask"
    ):
        if getattr(rec["masks"], "size", 0) > 0:
            # drop the last mask
            rec["masks"] = rec["masks"][:-1]
            rec["labels"][:-1]

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

    # visualize once
    display_img = rec["image"]
    if st.session_state["side_show_overlay"] and rec["masks"].shape[0] > 0:
        display_img = composite_over(rec["image"], rec["masks"], alpha=0.35)
    display_for_ui = np.array(
        Image.fromarray(display_img).resize((disp_w, disp_h), Image.BILINEAR)
    )

    # ensure nonces exist locally to avoid KeyErrors when called from this module
    ss = st.session_state
    ss.setdefault("pred_canvas_nonce", 0)
    ss.setdefault("edit_canvas_nonce", 0)

    if st.session_state["side_interaction_mode"] == "Draw mask":
        bg = Image.fromarray(display_for_ui).convert("RGBA")

        # We don't pre-seed strokes; existing masks are already shown in the overlay image.
        num_initial = 0  # exactly like boxes, but we start with an empty canvas

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 255, 0.30)",
            stroke_width=2,
            stroke_color="white",
            background_color="white",
            background_image=bg,
            update_streamlit=True,
            width=disp_w,
            height=disp_h,
            drawing_mode="freedraw",  # keep as freehand drawing
            point_display_radius=3,
            initial_drawing=None,  # important: start empty every run
            key=f"{key_ns}_canvas_edit_{ss['edit_canvas_nonce']}",
        )

        if canvas_result.json_data:
            objs = canvas_result.json_data.get("objects", [])
            added_any = False

            # process only objects drawn this run
            for obj in objs[num_initial:]:
                # accept Fabric "path" (from freedraw) and optional "polygon" if you switch modes later
                if obj.get("type") not in ("path", "polygon"):
                    continue

                # 1) create a display-sized mask from the drawn object
                mask_disp = polygon_to_mask(
                    obj, disp_h, disp_w
                )  # (disp_h, disp_w) uint8 {0,1}

                # 2) resize to image resolution + push into rec["masks"]
                mask_img = (
                    _resize_mask_nearest(mask_disp, rec["H"], rec["W"]) > 0
                ).astype(
                    np.uint8
                )  # (H,W)
                rec["masks"] = np.concatenate(
                    [rec["masks"], mask_img[None, ...]], axis=0
                )  # (N+1,H,W)
                rec["labels"].append(None)

                added_any = True

            # 3) re-render (also clears the canvas because the key/nonce changes)
            if added_any:
                ss["edit_canvas_nonce"] += 1
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
            key=f"{key_ns}_canvas_pred_{ss['pred_canvas_nonce']}",
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

        # record the click
        click = streamlit_image_coordinates(
            display_for_ui, key=f"{key_ns}_mask_click_remove", width=disp_w
        )

        if click:
            x = int(round(int(click["x"]) / scale))
            y = int(round(int(click["y"]) / scale))

            # rec.get("last_click_xy") makes sure that deletion occurs once per click
            if (
                0 <= x < rec["W"]
                and 0 <= y < rec["H"]
                and (x, y) != rec.get("last_click_xy")
            ):

                m = rec.get("masks")
                if isinstance(m, np.ndarray) and m.size:
                    m = np.asarray(m)

                    # Hit-test: which masks cover (y, x)?
                    hits = [i for i in range(m.shape[0]) if m[i, y, x] > 0]

                    if hits:
                        kill = hits[-1]  # delete the topmost/last one that hits
                        rec["masks"] = np.delete(m, kill, axis=0)

                        # keep metadata in sync if present
                        if (
                            isinstance(rec.get("labels"), list)
                            and len(rec["labels"]) > kill
                        ):
                            del rec["labels"][kill]

                        # remember last click to avoid double-fires on st.rerun()
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
