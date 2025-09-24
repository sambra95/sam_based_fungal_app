# panels/edit_masks.py
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates

from helpers.masks import polygon_to_mask, _resize_mask_nearest, composite_over
from helpers.boxes import is_unique_box, boxes_to_fabric_rects, draw_boxes_overlay

# panels/mask_editing.py
import numpy as np
import streamlit as st

from helpers.state_ops import ordered_keys, set_current_by_index, current
from helpers.masks import _resize_mask_nearest, toggle_at_point
from helpers import config as cfg  # CKPT_PATH, CFG_PATH
from contextlib import nullcontext

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def render_sidebar(*, key_ns: str = "side"):
    """
    Renders the sidebar controls for 'Create and Edit Masks'.
    """

    ok = ordered_keys()
    if not ok:
        st.warning("Upload an image in **Upload data** first.")
        # read from state so caller can no-op safely

    # ⬇️ get the current record (this was missing)
    cur = current()

    names = [st.session_state.images[k]["name"] for k in ok]
    curk = st.session_state.current_key
    cur_idx = ok.index(curk) if curk in ok else 0
    st.markdown(f"**Image {cur_idx+1}/{len(ok)}:** {names[cur_idx]}")

    c1, c2 = st.columns(2)
    if c1.button("◀ Prev", key=f"{key_ns}_prev", use_container_width=True):
        set_current_by_index(cur_idx - 1)
        st.rerun()
    if c2.button("Next ▶", key=f"{key_ns}_next", use_container_width=True):
        set_current_by_index(cur_idx + 1)
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
        cur["boxes"] = []
        st.session_state["pred_canvas_nonce"] = (
            st.session_state.get("pred_canvas_nonce", 0) + 1
        )
        st.rerun()

    if c2.button("Remove last box", use_container_width=True, key=f"{key_ns}_undo_box"):
        if cur["boxes"]:
            cur["boxes"].pop()
            st.session_state["pred_canvas_nonce"] = (
                st.session_state.get("pred_canvas_nonce", 0) + 1
            )
            st.rerun()

    if c3.button("Predict Masks", use_container_width=True, key=f"{key_ns}_predict"):
        _run_sam2_on_boxes(cur)  # your existing function that mutates cur and reruns

    # --- Mask utilities
    row2 = st.container()
    c4, c5 = row2.columns([1, 1])

    if c4.button("Clear masks", use_container_width=True, key=f"{key_ns}_clear_masks"):
        cur["masks"] = None
        cur["active"] = []
        cur["history"] = []
        cur["last_click_xy"] = None
        cur.pop("classes", None)
        st.rerun()

    if c5.button(
        "Remove last mask", use_container_width=True, key=f"{key_ns}_undo_mask"
    ):
        if cur["history"]:
            cur["active"] = cur["history"].pop()
            cur["last_click_xy"] = None

    # Download
    from helpers.masks import zip_all_masks

    downloadable = any(
        rec.get("masks") is not None and getattr(rec["masks"], "size", 0) > 0
        for rec in st.session_state.images.values()
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


def _run_sam2_on_boxes(cur: dict):
    """Predict with SAM2 for the current record's boxes; append masks; clear boxes; rerun."""

    boxes = np.array(cur["boxes"], dtype=np.float32)
    if boxes.size == 0:
        st.info("No boxes drawn yet.")
        return
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    boxes = boxes[(w > 0) & (h > 0)]
    if boxes.size == 0:
        st.info("All boxes were empty.")
        return

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    sam = build_sam2(
        cfg.CFG_PATH, cfg.CKPT_PATH, device=device, apply_postprocessing=False
    )
    predictor = SAM2ImagePredictor(sam)
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )
    with torch.inference_mode():
        with amp:
            predictor.set_image(cur["image"])

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes,
        multimask_output=True,
    )
    best_idx = scores.argmax(-1)
    row_idx = np.arange(scores.shape[0])
    masks_best = masks[row_idx, best_idx]  # (B,H,W)

    # normalize + resize to image size
    H, W = cur["H"], cur["W"]
    m = np.asarray(masks_best)
    if m.ndim == 4 and m.shape[-1] in (1, 3):
        m = m[..., 0]
    if m.ndim == 3 and m.shape[-1] in (1, 3):
        m = m[..., 0][None, ...]
    if m.ndim == 2:
        m = m[None, ...]
    if m.shape[-2:] != (H, W):
        m = np.stack(
            [_resize_mask_nearest(mi.astype(np.uint8), H, W) for mi in m], axis=0
        )
    m = (m > 0).astype(np.uint8)

    if cur.get("masks") is None or cur["masks"].size == 0:
        cur["masks"] = m
        cur["active"] = [True] * m.shape[0]
        cur["history"] = []
    else:
        cur["masks"] = np.concatenate([cur["masks"], m], axis=0)
        cur["active"].extend([True] * m.shape[0])

    cur["boxes"] = []
    st.session_state["pred_canvas_nonce"] = (
        st.session_state.get("pred_canvas_nonce", 0) + 1
    )
    st.success(
        f"Added {m.shape[0]} masks (total: {cur['masks'].shape[0]}) for {cur['name']}."
    )
    st.rerun()


def render_main(
    *,
    key_ns: str = "edit",
):
    """
    Render the Create/Edit Masks UI for the current image record.
    Required:
      cur          : image record dict (from st.session_state.images[...])
      mode         : one of ["Draw mask","Remove mask","Draw box","Remove box"]
    Optional:
      scale        : float (display scaling)
      key_ns       : widget key namespace to avoid collisions
    """
    cur = current()
    if cur is None:
        st.warning("Upload an image in **Upload data** first.")

    scale = (
        1.5  # hard coded scaling factor so that images show nicely on my laptop screen
    )
    H, W = cur["H"], cur["W"]
    disp_w, disp_h = int(W * scale), int(H * scale)

    # visualize once
    display_img = cur["image"]
    if (
        st.session_state["side_show_overlay"]
        and cur.get("masks") is not None
        and cur["masks"].size
    ):
        display_img = composite_over(
            cur["image"], cur["masks"], cur["active"], alpha=0.35
        )
    display_for_ui = np.array(
        Image.fromarray(display_img).resize((disp_w, disp_h), Image.BILINEAR)
    )

    # ensure nonces exist locally to avoid KeyErrors when called from this module
    ss = st.session_state
    ss.setdefault("pred_canvas_nonce", 0)
    ss.setdefault("edit_canvas_nonce", 0)

    if st.session_state["side_interaction_mode"] == "Draw mask":
        bg = Image.fromarray(display_for_ui).convert("RGBA")
        canvas_state = cur["canvas"]  # holds closed_json + processed_count

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 255, 0.3)",
            stroke_width=1,
            stroke_color="white",
            background_color="white",
            background_image=bg,
            update_streamlit=True,
            width=disp_w,
            height=disp_h,
            drawing_mode="freedraw",
            point_display_radius=3,
            initial_drawing=canvas_state.get("closed_json"),
            key=f"{key_ns}_canvas_edit_{ss['edit_canvas_nonce']}",
        )

        if canvas_result.json_data:
            data = canvas_result.json_data
            objs = data.get("objects", [])
            start_idx = canvas_state.get("processed_count", 0)
            new_objs = objs[start_idx:]
            changed = False
            added_any = False

            for obj in new_objs:
                if obj.get("type") != "path":
                    continue
                p = obj.get("path", [])
                if p and p[-1][0] != "Z":
                    p.append(["Z"])
                    obj["path"] = p
                    changed = True

                obj["fill"] = "rgba(0, 0, 255, 0.3)"
                obj["stroke"] = obj.get("stroke", "black")
                obj["strokeWidth"] = obj.get("strokeWidth", 3)

                mask_disp = polygon_to_mask(obj, disp_h, disp_w)
                mask_orig = _resize_mask_nearest(mask_disp, H, W)

                # append to masks stack
                mask_orig = (mask_orig > 0).astype(np.uint8)[None, ...]
                if cur.get("masks") is None or cur["masks"].size == 0:
                    cur["masks"] = mask_orig
                    cur["active"] = [True]
                    cur["history"] = []
                else:
                    cur["masks"] = np.concatenate([cur["masks"], mask_orig], axis=0)
                    cur["active"].append(True)
                added_any = True

            canvas_state["processed_count"] = len(objs)
            if added_any:
                # clear drawn strokes so they don’t persist
                canvas_state["closed_json"] = None
                canvas_state["processed_count"] = 0
                ss["edit_canvas_nonce"] += 1
                st.rerun()
            elif changed:
                canvas_state["closed_json"] = data

    elif st.session_state["side_interaction_mode"] == "Draw box":
        bg = Image.fromarray(display_for_ui).convert("RGBA")
        initial_json = boxes_to_fabric_rects(cur["boxes"], scale=scale)
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

                x0 = max(0, min(W - 1, x0))
                x1 = max(0, min(W, x1))
                y0 = max(0, min(H - 1, y0))
                y1 = max(0, min(H, y1))
                if x1 < x0:
                    x0, x1 = x1, x0
                if y1 < y0:
                    y0, y1 = y1, y0

                box = (x0, y0, x1, y1)
                if is_unique_box(box, cur["boxes"]):
                    cur["boxes"].append(box)
                    added_any = True
            if added_any:
                st.rerun()

    elif st.session_state["side_interaction_mode"] == "Remove mask":
        click = streamlit_image_coordinates(
            display_for_ui, key=f"{key_ns}_img_click", width=disp_w
        )
        if click:
            x0 = int(round(int(click["x"]) / scale))
            y0 = int(round(int(click["y"]) / scale))
            if 0 <= x0 < W and 0 <= y0 < H and (x0, y0) != cur["last_click_xy"]:
                if cur.get("masks") is not None and cur["masks"].size:

                    # --- find which instance you clicked (BEFORE toggling) ---
                    m = np.asarray(cur["masks"])
                    a = cur.get("active", [True] * (m.shape[0] if m.ndim == 3 else 1))
                    if m.ndim == 2:
                        m, a = m[None, ...], [True]
                    elif m.ndim == 3 and m.shape[-1] == 1:
                        m = m[..., 0]
                    keep = [i for i, t in enumerate(a) if t]
                    m_active = (m[keep] if keep else np.zeros((0, H, W), np.uint8)) > 0
                    from helpers.masks import stack_to_instances_binary_first

                    inst = (
                        stack_to_instances_binary_first(m_active)
                        if m_active.size
                        else np.zeros((H, W), np.uint16)
                    )
                    inst_id = int(inst[y0, x0])

                    # --- toggle the mask at the click ---
                    cur["history"].append(cur["active"].copy())
                    cur["active"] = toggle_at_point(cur["active"], cur["masks"], x0, y0)
                    cur["last_click_xy"] = (x0, y0)

                    # --- remove its label if it had one ---
                    if inst_id > 0:
                        cur.setdefault("classes", {}).pop(inst_id, None)

                    # if nothing left active, clear masks AND labels
                    if (
                        cur["masks"] is not None
                        and cur["masks"].size
                        and not any(cur["active"])
                    ):
                        cur["masks"] = None
                        cur["active"] = []
                        cur["history"] = []
                        cur.pop("classes", None)

                    st.rerun()

    else:  # "Remove box"
        overlay = draw_boxes_overlay(
            display_img, cur["boxes"], alpha=0.25, outline_px=2
        )
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
                for i, (x0, y0, x1, y1) in enumerate(cur["boxes"])
                if (x0 <= x < x1) and (y0 <= y < y1)
            ]
            if hits:
                cur["boxes"].pop(hits[-1])
                st.rerun()
