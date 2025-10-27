import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from contextlib import nullcontext
from typing import Dict, Any
from scipy import ndimage as ndi
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates

from helpers import config as cfg  # CKPT_PATH, CFG_PATH
from helpers.state_ops import ordered_keys, current
from helpers.classifying_functions import classes_map_from_labels, palette_from_emojis
from helpers.cellpose_functions import segment_rec_with_cellpose

# -----------------------------------------------------#
# --------------- MASK HELPERS SIDEBAR --------------- #
# -----------------------------------------------------#


# masks can be cut when adding them to the existing array (new masks lose priority).
# therefore, add mask, rextract to see if it is cut, if so, take it out and re-add the largest section
def keep_largest_part(mask: np.ndarray) -> np.ndarray:
    """Return only the largest connected component of a boolean mask."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    lab, n = ndi.label(mask)
    if n == 1:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def integrate_new_mask(original: np.ndarray, new_binary: np.ndarray):
    """
    Add a new mask into a label image.
    - original: (H,W) int labels, 0=background, 1..N instances
    - new_binary: (H,W) boolean mask
    Returns (updated_label_image, new_id or None)
    """
    out = original
    nb = new_binary.astype(bool)
    if nb.ndim != 2 or not nb.any():
        return out, None

    # write only where background
    write = (out == 0) & nb
    if not write.any():
        return out, None

    max_id = int(out.max(initial=0))
    new_id = max_id + 1

    # upcast if needed
    if new_id > np.iinfo(out.dtype).max:
        out = out.astype(np.uint32)
    else:
        out = out.copy()

    out[write] = new_id

    # --- check contiguity: keep only the largest surviving component ---
    mask_new = out == new_id
    mask_new = keep_largest_part(mask_new)
    if not mask_new.any():
        return original, None  # nothing left after check

    out[out == new_id] = 0  # clear possibly cut version
    out[mask_new] = new_id  # reapply only largest part

    return out, new_id


# -----------------------------------------------------#
# ---------- SEGMENTATION SIDEBAR (CELLPOSE) --------- #
# -----------------------------------------------------#


def _segment_current_and_refresh():
    """calls cellpose to segment the current image"""
    rec = current()
    if rec is not None:
        params = _read_cellpose_hparams_from_state()
        segment_rec_with_cellpose(rec, **params)
        st.session_state["edit_canvas_nonce"] += 1
    st.rerun()


def _batch_segment_and_refresh():
    """calls cellpose to segment all images with progress bar"""
    ok = ordered_keys()
    params = _read_cellpose_hparams_from_state()
    n = len(ok)
    pb = st.progress(0.0, text="Starting…")
    for i, k in enumerate(ok, 1):
        segment_rec_with_cellpose(st.session_state.images.get(k), **params)
        pb.progress(i / n, text=f"Segmented {i}/{n}")
    pb.empty()
    st.session_state["edit_canvas_nonce"] += 1
    st.rerun()


def _read_cellpose_hparams_from_state():
    """calls hparam values from session state"""
    # Build kwargs matching segment_rec_with_cellpose signature
    ch1 = int(st.session_state.get("cp_ch1", 0))
    ch2 = int(st.session_state.get("cp_ch2", 0))
    diameter = st.session_state.get("cp_diameter", None)
    # ensure None if 0.0 when Auto
    if st.session_state.get("cp_diam_mode", "Auto (None)") == "Auto (None)":
        diameter = None

    return dict(
        channels=(ch1, ch2),
        diameter=diameter,
        cellprob_threshold=float(st.session_state.get("cp_cellprob_threshold", -0.2)),
        flow_threshold=float(st.session_state.get("cp_flow_threshold", 0.4)),
        min_size=int(st.session_state.get("cp_min_size", 0)),
    )


def _reset_cellpose_hparams_to_defaults():
    """resets hparam values in session state"""
    st.session_state["cp_ch1"] = 0
    st.session_state["cp_ch2"] = 0
    st.session_state["cp_diam_mode"] = "Auto (None)"
    st.session_state["cp_diameter"] = None
    st.session_state["cp_cellprob_threshold"] = -0.2
    st.session_state["cp_flow_threshold"] = 0.4
    st.session_state["cp_min_size"] = 0
    st.toast("Cellpose hyperparameters reset to defaults")


# ============================================================
# --------- SEGMENTATION SIDEBAR (BOXES AND SAM2) ------------
# ============================================================


def boxes_to_fabric_rects(boxes, scale=1.0) -> Dict[str, Any]:
    """draw box for sam2 mask predictions on canvas rendering"""
    rects = []
    for x0, y0, x1, y1 in boxes:
        rects.append(
            {
                "type": "rect",
                "left": x0 * scale,
                "top": y0 * scale,
                "width": (x1 - x0) * scale,
                "height": (y1 - y0) * scale,
                "fill": "rgba(0, 0, 255, 0.25)",
                "stroke": "white",
                "strokeWidth": 2,
                "selectable": False,
                "evented": False,
                "hasControls": False,
                "lockMovementX": True,
                "lockMovementY": True,
                "hoverCursor": "crosshair",
            }
        )
    return {"objects": rects}


def draw_boxes_overlay(image_u8, boxes, alpha=0.25, outline_px=2):
    """overlay drawn boxes the image"""
    base = Image.fromarray(image_u8).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    fill = (0, 0, 255, int(alpha * 255))
    for x0, y0, x1, y1 in boxes:
        d.rectangle(
            [x0, y0, x1, y1], fill=fill, outline=(255, 255, 255, 255), width=outline_px
        )
    return np.array(Image.alpha_composite(base, overlay).convert("RGB"), dtype=np.uint8)


def _prep_for_sam(img: np.ndarray) -> np.ndarray:
    """preprocess image into correct format for mask prediction with sam2"""
    a = img
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    elif a.ndim == 3 and a.shape[2] == 4:
        a = np.array(Image.fromarray(a).convert("RGB"))
    a = a.astype(np.float32)
    mx = a.max() if a.size else 1.0
    if mx > 1.0:
        a /= 255.0 if mx <= 255 else (65535.0 if mx <= 65535 else mx)
    return a


def _run_sam2_on_boxes(cur: dict):
    """input is record for prediction. boxes to guide prediction will be extracted wtih "boxes" key.
    Return a list of (H,W) boolean masks (best mask per box."""
    boxes = np.asarray(cur.get("boxes", []), dtype=np.float32)
    if boxes.size == 0:
        st.info("No boxes drawn yet.")
        return []
    if boxes.ndim == 1:
        boxes = boxes[None, :]  # (1,4)

    # drop degenerate
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    boxes = boxes[(w > 0) & (h > 0)]
    if boxes.size == 0:
        st.info("All boxes were empty.")
        return []

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    sam = build_sam2(
        cfg.CFG_PATH,
        cfg.CKPT_PATH,
        device=device,
        apply_postprocessing=False,  # post-processing not supported with MPS :(
    )
    predictor = SAM2ImagePredictor(sam)

    img_float = _prep_for_sam(cur["image"])
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), amp:
        predictor.set_image(img_float)
        masks, scores, _ = predictor.predict(
            point_coords=None, point_labels=None, box=boxes, multimask_output=True
        )

    # to numpy
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    # normalize shapes:
    if masks.ndim == 3:
        masks = masks[None, ...]
    if scores.ndim == 1:
        scores = scores[None, ...]

    B = scores.shape[0]
    best = scores.argmax(-1)  # (B,)
    masks_best = masks[np.arange(B), best]  # (B,H,W)

    H, W = int(cur["H"]), int(cur["W"])
    out = []
    for mi in masks_best:
        mi = mi > 0
        if mi.shape != (H, W):
            mi = np.array(
                Image.fromarray(mi.astype(np.uint8)).resize((W, H), Image.NEAREST),
                dtype=bool,
            )
        out.append(mi)
    return out  # list of boalean mask (best mask for each box)


# ============================================================
# ----------- MANUAL MASK EDITING SIDEBAR --------------------
# ============================================================


def polygon_to_mask(obj, h, w):
    """drawing function for adding masks by freehand drawing on rendered canvas"""
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    pts = []
    for cmd in obj.get("path", []):
        if (
            len(cmd) >= 3
            and isinstance(cmd[1], (int, float))
            and isinstance(cmd[2], (int, float))
        ):
            pts.append((cmd[1], cmd[2]))
    if pts:
        draw.polygon(pts, outline=1, fill=1)
    return np.array(mask_img, dtype=np.uint8)


# -----------------------------------------------------#
# ------------------ RENDER SIDE BAR ----------------- #
# -----------------------------------------------------#


def set_current_by_index(idx: int):
    ok = ordered_keys()
    if not ok:
        return
    st.session_state.current_key = ok[idx % len(ok)]


def nav_fragment(key_ns="side"):
    ok = ordered_keys()
    if not ok:
        st.warning("Upload an image in **Upload data** first.")
        return

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


@st.fragment
def cellpose_actions_fragment():
    # --- Hyperparameters (collapsible) ---
    with st.expander("Segment cells with Cellpose", expanded=False):

        if st.button("Segment image with Cellpose", use_container_width=True):
            _segment_current_and_refresh()

        if st.button("Segment all images with Cellpose", use_container_width=True):
            _batch_segment_and_refresh()

        # We use a small form so changing values doesn't trigger reruns mid-typing
        with st.form("cellpose_hparams_form", clear_on_submit=False):
            # Channels (two ints)
            st.number_input(
                "Channel 1",
                value=st.session_state.get("cp_ch1", 0),
                step=1,
                format="%d",
                key="cp_ch1",
            )
            st.number_input(
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
                st.form_submit_button("Apply changes", use_container_width=True)
            with cols[1]:
                if st.form_submit_button("Reset defaults", use_container_width=True):
                    _reset_cellpose_hparams_to_defaults()

        # sync diameter to None when Auto selected
        if st.session_state.get("cp_diam_mode", "Auto (None)") == "Auto (None)":
            st.session_state["cp_diameter"] = None


def box_tools_fragment(key_ns="side"):
    rec = current()
    row = st.container()
    c1, c2 = row.columns([1, 1])

    if c1.button("Draw box", use_container_width=True, key=f"{key_ns}_draw_boxes"):
        st.session_state[f"interaction_mode"] = "Draw box"
        st.rerun()

    if c2.button("Remove box", use_container_width=True, key=f"{key_ns}_remove_boxes"):
        st.session_state[f"interaction_mode"] = "Remove box"
        st.rerun()

    row = st.container()
    c1, c2 = row.columns([1, 1])

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

    if st.button(
        "Segment cells in boxes", use_container_width=True, key=f"{key_ns}_predict"
    ):
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


def mask_tools_fragment(key_ns="side"):
    rec = current()
    row = st.container()
    c1, c2 = row.columns([1, 1])

    if c1.button("Draw mask", use_container_width=True, key=f"{key_ns}_draw_masks"):
        st.session_state[f"interaction_mode"] = "Draw mask"
        st.rerun()

    if c2.button("Remove mask", use_container_width=True, key=f"{key_ns}_remove_masks"):
        st.session_state[f"interaction_mode"] = "Remove mask"
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


# -----------------------------------------------------#
# ---------------- RENDER MAIN DISPLAY --------------- #
# -----------------------------------------------------#


def composite_over_by_class(image_u8, label_inst, classes_map, palette, alpha=0.5):
    """
    image_u8:  uint8 RGB image, shape (H, W, 3)
    label_inst: uint{8,16,32} label image, shape (H, W), 0=background, 1..N=instances
    classes_map: dict[int -> class_name]
    palette: dict[class_name -> (r,g,b) in 0..1]
    alpha: overlay opacity for filled region
    """
    H, W = image_u8.shape[:2]
    inst = np.asarray(label_inst)

    if inst.ndim != 2:
        raise ValueError("label_inst must be a 2D label image (H, W)")
    if inst.shape != (H, W):
        # nearest to preserve integer labels
        inst = np.array(
            Image.fromarray(inst).resize((W, H), Image.NEAREST), dtype=inst.dtype
        )

    if inst.size == 0 or not np.any(inst):
        return image_u8

    out = image_u8.astype(np.float32) / 255.0

    ids = np.unique(inst)
    ids = ids[ids != 0]  # skip background

    for iid in ids:
        cls = classes_map.get(int(iid), "__unlabeled__")
        color = np.array(palette.get(cls, palette["__unlabeled__"]), dtype=np.float32)

        mm = inst == iid

        # fill
        a = (mm.astype(np.float32) * alpha)[..., None]
        out = out * (1 - a) + color[None, None, :] * a

        # 1px white edge (simple interior test)
        interior = (
            mm
            & np.roll(mm, 1, 0)
            & np.roll(mm, -1, 0)
            & np.roll(mm, 1, 1)
            & np.roll(mm, -1, 1)
        )
        edge = mm & ~interior
        out[edge] = 1.0

    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


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


@st.fragment
def display_and_interact_fragment(key_ns="edit", mode_ns="side", scale=1.5):

    rec = current()
    if rec is None:
        st.warning("Upload an image in **Upload data** first.")
        return

    # ensure class defaults exist (safe if already set by sidebar)
    st.session_state.setdefault("all_classes", ["Remove label"])
    st.session_state.setdefault("side_current_class", "Remove label")

    base_img, display_for_ui, disp_w, disp_h = _image_display(rec, scale)
    mode = st.session_state.get(f"interaction_mode", "Draw box")

    M = rec.get("masks")
    has_instances = isinstance(M, np.ndarray) and M.ndim == 2 and M.any()

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
            drawing_mode="freedraw",  # new arguement
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
                st.rerun(scope="fragment")

    # click and hold to draw boxes on the image
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
            drawing_mode="rect",  # new arguement
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
                rec["boxes"].append(box)
                added_any = True
            if added_any:
                st.rerun(scope="fragment")

    # click a mask in the image to remove the mask
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
                        st.rerun(scope="fragment")

    # click on a box in the image to remove the box
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

    # click on a mask in the image to assign it the current class
    elif mode == "Assign class":
        click = streamlit_image_coordinates(
            display_for_ui, key=f"{key_ns}_class_click", width=disp_w
        )
        if click and has_instances:
            x0 = int(round(int(click["x"]) / scale))
            y0 = int(round(int(click["y"]) / scale))
            if (
                0 <= x0 < rec["W"]
                and 0 <= y0 < rec["H"]
                and (x0, y0) != rec.get("last_click_xy")
            ):
                iid = int(M[y0, x0])
                if iid > 0:
                    cur_class = st.session_state.get("side_current_class")
                    if cur_class == "Remove label":
                        rec.setdefault("labels", {}).pop(iid, None)
                    else:
                        rec.setdefault("labels", {})[iid] = cur_class
                    rec["last_click_xy"] = (x0, y0)
                    # force overlay to refresh if it's on
                    st.session_state["edit_canvas_nonce"] += 1
                    st.rerun()
                else:
                    rec["last_click_xy"] = (x0, y0)
