import numpy as np
from PIL import Image, ImageDraw
import io
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import tifffile as tiff


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


def toggle_at_point(active, masks_u8, x, y):
    if masks_u8 is None or not masks_u8.size:
        return active
    hits = [i for i in range(masks_u8.shape[0]) if active[i] and masks_u8[i, y, x] > 0]
    if hits:
        active[hits[-1]] = not active[hits[-1]]
    return active


def composite_over(image_u8, masks_u8, active, alpha=0.5):
    """
    Overlay active masks in red + white outline onto image_u8.
    Accepts masks shaped (H,W), (N,H,W), (N,H,W,1/3), or (H,W,1/3).
    Does NOT auto-transpose on square images.
    """
    H, W = image_u8.shape[:2]
    m = np.asarray(masks_u8)

    # Coerce to (N,H,W) WITHOUT any transpose guesses
    if m.ndim == 2:
        m = m[None, ...]
    elif m.ndim == 3:
        # (H,W,1/3) -> drop channel; else assume already (N,H,W)
        if m.shape[-1] in (1, 3) and m.shape[:2] == (H, W):
            m = m[..., 0][None, ...]
    elif m.ndim == 4:
        # (N,H,W,1/3) -> drop channel; (N,1/3,H,W) -> take first channel
        if m.shape[-1] in (1, 3):
            m = m[..., 0]
        elif m.shape[1] in (1, 3):
            m = m[:, 0, ...]
        # else: assume already (N,H,W) after channel handling

    # Final shape guard
    if m.ndim == 2:
        m = m[None, ...]
    if m.shape[-2:] != (H, W):
        m = np.stack(
            [_resize_mask_nearest(mi.astype(np.uint8), H, W) for mi in m], axis=0
        )

    masks = (m > 0).astype(np.uint8)
    N = masks.shape[0]
    if not isinstance(active, (list, tuple)) or len(active) != N:
        active = [True] * N

    out = image_u8.astype(np.float32) / 255.0
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    for i in range(N):
        if not active[i]:
            continue
        mi = (masks[i] > 0).astype(np.float32)  # (H,W)
        a = (mi * alpha)[..., None]  # (H,W,1)
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


def _attach_masks_to_image(rec, new_masks):
    """Normalize/resize and attach masks to the given image record."""
    m = np.asarray(new_masks)
    # normalize shape to (N,H,W) without transpose guesses
    if m.ndim == 4 and m.shape[-1] in (1, 3):
        m = m[..., 0]
    if m.ndim == 3 and m.shape[-1] in (1, 3):
        m = m[..., 0][None, ...]
    if m.ndim == 2:
        m = m[None, ...]
    # resize to image size
    H, W = rec["H"], rec["W"]
    if m.shape[-2:] != (H, W):
        m = np.stack(
            [_resize_mask_nearest(mi.astype(np.uint8), H, W) for mi in m], axis=0
        )
    m = (m > 0).astype(np.uint8)

    # append instead of overwrite (match your Predict behavior)
    if rec.get("masks") is None or rec["masks"].size == 0:
        rec["masks"] = m
        rec["active"] = [True] * m.shape[0]
        rec["history"] = []
    else:
        rec["masks"] = np.concatenate([rec["masks"], m], axis=0)
        rec["active"].extend([True] * m.shape[0])
        rec["history"] = []


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
                a = rec.get("active")
                if isinstance(a, list) and len(a) == m.shape[0]:
                    m = m[[i for i, t in enumerate(a) if t]]
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


import numpy as np


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


def composite_over_by_class(
    image_u8, masks_u8, active, classes_map, palette, alpha=0.5
):
    """
    Overlay active instances colored by class.
    - classes_map: {instance_id:int -> class:str}
    - palette: {class:str -> (r,g,b) floats 0..1}
    """
    from .masks import (
        _resize_mask_nearest,
        stack_to_instances_binary_first,
    )  # reuse your helpers

    H, W = image_u8.shape[:2]
    m = np.asarray(masks_u8)
    if m is None or m.size == 0:
        return image_u8

    # normalize to (N,H,W)
    if m.ndim == 2:
        m = m[None, ...]
    elif m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    if m.shape[-2:] != (H, W):
        m = np.stack([_resize_mask_nearest(mi.astype(np.uint8), H, W) for mi in m], 0)
    m = (m > 0).astype(np.uint8)

    # filter active
    if not isinstance(active, (list, tuple)) or len(active) != m.shape[0]:
        active = [True] * m.shape[0]
    idx = [i for i, a in enumerate(active) if a]
    if not idx:
        return image_u8
    m = m[idx]

    inst = stack_to_instances_binary_first(m)  # (H,W) uint16, 0=bg
    out = image_u8.astype(np.float32) / 255.0

    # draw each instance with its class color + white outline
    ids = np.unique(inst)
    ids = ids[ids != 0]
    for iid in ids:
        cls = classes_map.get(int(iid), "__unlabeled__")
        color = np.array(palette.get(cls, palette["__unlabeled__"]), dtype=np.float32)
        mm = inst == iid
        a = (mm.astype(np.float32) * alpha)[..., None]
        out = out * (1 - a) + color[None, None, :] * a

        # white outline
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
