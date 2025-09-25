import numpy as np
from PIL import Image, ImageDraw
import io
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import tifffile as tiff
import streamlit as st
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from contextlib import nullcontext
from helpers import config as cfg  # CKPT_PATH, CFG_PATH
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict, Any


# ============================================================
# ---------------- MAKE AND EDIT BOXES -----------------------
# ============================================================


def is_unique_box(box, boxes):
    x0, y0, x1, y1 = box
    for b in boxes:
        if b == (x0, y0, x1, y1):
            return False
    return True


def boxes_to_fabric_rects(boxes, scale=1.0) -> Dict[str, Any]:
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


def _resize_mask_nearest(mask_u8, out_h, out_w):
    return np.array(
        Image.fromarray(mask_u8).resize((out_w, out_h), resample=Image.NEAREST),
        dtype=np.uint8,
    )


def polygon_to_mask(obj, h, w):
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


def composite_over(image_u8, masks_u8, alpha=0.5):
    H, W = image_u8.shape[:2]
    m = np.asarray(masks_u8)

    # Coerce to (N,H,W) WITHOUT transpose guesses
    if m.ndim == 2:
        m = m[None, ...]
    elif m.ndim == 3:
        if m.shape[-1] in (1, 3) and m.shape[:2] == (H, W):
            m = m[..., 0][None, ...]
    elif m.ndim == 4:
        if m.shape[-1] in (1, 3):
            m = m[..., 0]
        elif m.shape[1] in (1, 3):
            m = m[:, 0, ...]

    if m.ndim == 2:
        m = m[None, ...]
    if m.shape[-2:] != (H, W):
        m = np.stack([_resize_mask_nearest(mi.astype(np.uint8), H, W) for mi in m], 0)

    masks = (m > 0).astype(np.uint8)
    N = masks.shape[0]

    out = image_u8.astype(np.float32) / 255.0
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    for i in range(N):  # no active filtering; draw all remaining masks
        mi = (masks[i] > 0).astype(np.float32)
        a = (mi * alpha)[..., None]
        out = out * (1 - a) + red[None, None, :] * a

        mb = mi.astype(bool)
        interior = (
            mb
            & np.roll(mb, 1, 0)
            & np.roll(mb, -1, 0)
            & np.roll(mb, 1, 1)
            & np.roll(mb, -1, 1)
        )
        edge = mb & ~interior
        out[edge] = 1.0

    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _run_sam2_on_boxes(cur: dict):
    """Predict with SAM2 for the current record's boxes and return a (B,H,W) uint8 stack."""
    boxes = np.array(cur["boxes"], dtype=np.float32)
    if boxes.size == 0:
        st.info("No boxes drawn yet.")
        return np.zeros((0, cur["H"], cur["W"]), dtype=np.uint8)

    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    boxes = boxes[(w > 0) & (h > 0)]
    if boxes.size == 0:
        st.info("All boxes were empty.")
        return np.zeros((0, cur["H"], cur["W"]), dtype=np.uint8)

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

    # Handle torch tensors or numpy
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    # masks is (B, M, H, W); pick best per box -> (B, H, W)
    best_idx = scores.argmax(-1)  # (B,)
    row_idx = np.arange(scores.shape[0])  # (B,)
    masks_best = masks[row_idx, best_idx, ...]  # (B,H,W)

    # --- Normalize to canonical stack: uint8 {0,1}, correct (H,W) ---
    H, W = cur["H"], cur["W"]
    # If SAM already returns bool, astype handles it; if float, >0 handles it.
    masks_best = (masks_best > 0).astype(np.uint8)

    if masks_best.shape[-2:] != (H, W):
        masks_best = np.stack(
            [_resize_mask_nearest(mi, H, W) for mi in masks_best], axis=0
        ).astype(np.uint8)

    # Ensure contiguous (helps later concatenations)
    return np.ascontiguousarray(masks_best, dtype=np.uint8)


def append_masks_to_rec(rec: dict, new_masks: np.ndarray):
    H, W = rec["H"], rec["W"]
    if not isinstance(rec.get("masks"), np.ndarray) or rec["masks"].ndim != 3:
        rec["masks"] = np.zeros((0, H, W), dtype=np.uint8)
    rec.setdefault("labels", [])

    nm = np.asarray(new_masks)
    if nm.size == 0:
        return rec
    if nm.ndim == 2:
        nm = nm[None, ...]  # (1,H,W)
    if nm.shape[-2:] != (H, W):
        nm = np.stack([_resize_mask_nearest(mi.astype(np.uint8), H, W) for mi in nm], 0)
    nm = (nm > 0).astype(np.uint8)  # binary uint8

    rec["masks"] = np.concatenate([rec["masks"], nm], axis=0)
    rec["labels"].extend([None] * nm.shape[0])
    return rec


def zip_all_masks(images: dict, keys: list[int]) -> bytes:
    buf = io.BytesIO()
    with ZipFile(buf, "w") as zf:
        for k in keys:
            rec = images[k]
            H, W = rec["H"], rec["W"]
            m = rec.get("masks")

            if m is None or getattr(m, "size", 0) == 0:
                inst = np.zeros((H, W), np.uint16)
            else:
                m = np.asarray(m)
                if m.size == 0:
                    inst = np.zeros((H, W), np.uint16)
                else:
                    if m.ndim == 2:
                        m = m[None, ...]
                    elif m.ndim == 3 and m.shape[-1] == 1:
                        m = m[..., 0]
                    m = (m > 0).astype(np.uint8)
                    if m.shape[-2:] != (H, W):
                        m = np.stack([_resize_mask_nearest(mi, H, W) for mi in m], 0)
                    inst = stack_to_instances_binary_first(m)

            b = io.BytesIO()
            tiff.imwrite(
                b, inst, dtype=np.uint16, photometric="minisblack", compression="zlib"
            )
            zf.writestr(f"{Path(rec['name']).stem}_mask.tif", b.getvalue())

    buf.seek(0)
    return buf.getvalue()  # <-- make sure this line exists


def stack_to_instances_binary_first(m: np.ndarray) -> np.ndarray:
    """
    Robust stack (N,H,W,[...]) -> instance labels (H,W) uint16.
    - Treats >0 as 1
    - Resolves overlaps by descending area (largest wins), matching your UI behavior.
    """
    m = np.asarray(m)
    if m.ndim == 4 and m.shape[-1] in (1, 3):
        m = m[..., 0]
    if m.ndim == 3 and m.shape[-1] in (1, 3):  # (H,W,1) sneaking in as 'stack'
        m = m[..., 0][None, ...]
    if m.ndim == 2:
        m = m[None, ...]
    bin_stack = (m > 0).astype(np.uint8)

    N, H, W = bin_stack.shape
    areas = bin_stack.reshape(N, -1).sum(axis=1)
    order = np.argsort(-areas)  # largest first

    inst = np.zeros((H, W), dtype=np.uint16)
    curr_id = 1
    for ch in order:
        mm = bin_stack[ch] > 0
        if mm.sum() == 0:
            continue
        write_here = mm & (inst == 0)
        if write_here.any():
            inst[write_here] = curr_id
            curr_id += 1
    return inst


def get_class_palette(labels, *, ss_key="class_colors"):
    """Persistent {class -> RGB float tuple} in session_state."""
    import streamlit as st

    base = [
        (0.90, 0.10, 0.10),
        (0.10, 0.60, 0.95),
        (0.10, 0.75, 0.25),
        (0.95, 0.70, 0.10),
        (0.55, 0.10, 0.80),
        (0.05, 0.80, 0.70),
        (0.85, 0.30, 0.30),
        (0.30, 0.85, 0.30),
    ]
    pal = st.session_state.setdefault(ss_key, {})
    for i, c in enumerate(labels):
        pal.setdefault(c, base[i % len(base)])
    pal.setdefault("__unlabeled__", (0.95, 0.95, 0.95))  # light gray
    return pal


def composite_over_by_class(image_u8, masks_u8, classes_map, palette, alpha=0.5):
    H, W = image_u8.shape[:2]
    m = np.asarray(masks_u8)
    if m is None or m.size == 0:
        return image_u8

    if m.ndim == 2:
        m = m[None, ...]
    elif m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    if m.shape[-2:] != (H, W):
        m = np.stack([_resize_mask_nearest(mi.astype(np.uint8), H, W) for mi in m], 0)
    m = (m > 0).astype(np.uint8)

    # *** no active filtering here ***
    inst = stack_to_instances_binary_first(m)
    out = image_u8.astype(np.float32) / 255.0

    ids = np.unique(inst)
    ids = ids[ids != 0]
    for iid in ids:
        cls = classes_map.get(int(iid), "__unlabeled__")
        color = np.array(palette.get(cls, palette["__unlabeled__"]), dtype=np.float32)
        mm = inst == iid
        a = (mm.astype(np.float32) * alpha)[..., None]
        out = out * (1 - a) + color[None, None, :] * a

        mb = mm
        interior = (
            mb
            & np.roll(mb, 1, 0)
            & np.roll(mb, -1, 0)
            & mb
            & np.roll(mb, 1, 1)
            & np.roll(mb, -1, 1)
        )
        edge = mb & ~interior
        out[edge] = 1.0

    return (np.clip(out, 0, 1) * 255).astype(np.uint8)
