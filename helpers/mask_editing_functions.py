import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from contextlib import nullcontext
from helpers import config as cfg  # CKPT_PATH, CFG_PATH
from typing import Dict, Any
from scipy import ndimage as ndi

from helpers.cellpose_functions import segment_rec_with_cellpose
from helpers.state_ops import ordered_keys, current


# ============================================================
# ---------------- MAKE AND EDIT BOXES -----------------------
# ============================================================


def is_unique_box(box, boxes):
    """checks that the box doesnt already exist. avoids duplicate mask predictions"""
    x0, y0, x1, y1 = box
    for b in boxes:
        if b == (x0, y0, x1, y1):
            return False
    return True


def boxes_to_fabric_rects(boxes, scale=1.0) -> Dict[str, Any]:
    """draw box for mask predictions on canvas rendering"""
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
    """render the draw boxes in front of the image"""
    base = Image.fromarray(image_u8).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    fill = (0, 0, 255, int(alpha * 255))
    for x0, y0, x1, y1 in boxes:
        d.rectangle(
            [x0, y0, x1, y1], fill=fill, outline=(255, 255, 255, 255), width=outline_px
        )
    return np.array(Image.alpha_composite(base, overlay).convert("RGB"), dtype=np.uint8)


# ============================================================
# ---------------- MAKE AND EDIT MASKS -----------------------
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


def _reset_cellpose_hparams_to_defaults():
    st.session_state["cp_ch1"] = 0
    st.session_state["cp_ch2"] = 0
    st.session_state["cp_diam_mode"] = "Auto (None)"
    st.session_state["cp_diameter"] = None
    st.session_state["cp_cellprob_threshold"] = -0.2
    st.session_state["cp_flow_threshold"] = 0.4
    st.session_state["cp_min_size"] = 0
    st.session_state["cp_do_normalize"] = True
    st.toast("Cellpose hyperparameters reset to defaults")


def _segment_current_and_refresh():
    rec = current()
    if rec is not None:
        params = _read_cellpose_hparams_from_state()
        segment_rec_with_cellpose(rec, **params)
        st.session_state["edit_canvas_nonce"] += 1
    st.rerun()


def _batch_segment_and_refresh():
    ok = ordered_keys()
    if not ok:
        return
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
        do_normalize=bool(st.session_state.get("cp_do_normalize", True)),
    )
