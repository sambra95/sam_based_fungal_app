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
from typing import Dict, Any
from scipy import ndimage as ndi


# ============================================================
# ---------------- MAKE AND EDIT BOXES -----------------------
# ============================================================


def is_unique_box(box, boxes):
    x0, y0, x1, y1 = box
    for b in boxes:
        if b == (x0, y0, x1, y1):
            return False
    return True


def keep_largest_part(mask: np.ndarray) -> np.ndarray:
    """Return only the largest connected component of a boolean mask."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    lab, n = ndi.label(mask)
    if n == 1:
        return mask.astype(bool)

    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # background
    return lab == sizes.argmax()


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


def _prep_for_sam(img: np.ndarray) -> np.ndarray:
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
    """Return a list of (H,W) boolean masks — best mask per box."""
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
    return out


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
