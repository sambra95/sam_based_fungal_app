# panels/classify_cells.py
import numpy as np
import streamlit as st
from PIL import Image
import io
import pandas as pd
from pathlib import Path
from zipfile import ZipFile
import cv2


from helpers.state_ops import ordered_keys

from helpers.mask_editing_functions import stack_to_instances_binary_first

# --- builder: fills session_state["classifier_records_df"] and ["classifier_patches"] ---


def classes_map_from_labels(masks, labels):
    m = np.asarray(masks)
    if m.ndim == 2:
        m = m[None, ...]
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    m = (m > 0).astype(np.uint8)
    inst = stack_to_instances_binary_first(m)  # (H,W) uint16
    classes_map = {}
    ids = np.unique(inst)
    ids = ids[ids != 0]
    for iid in ids:
        mm = inst == iid
        # pick mask index with max overlap for this instance
        ov = [(i, int((m[i] & mm).sum())) for i in range(m.shape[0])]
        owner = max(ov, key=lambda t: t[1])[0]
        cls = labels[owner] if owner < len(labels) else None
        if cls is not None:
            classes_map[int(iid)] = cls
    return classes_map


def _stem(name: str) -> str:
    return Path(name).stem


def _to_square_patch(rgb: np.ndarray, patch_size: int = 256) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    s = patch_size / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    resized = np.array(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
    canvas = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    y0, x0 = (patch_size - nh) // 2, (patch_size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


import numpy as np, cv2


def extract_masked_cell_patch(
    image: np.ndarray, mask: np.ndarray, size: int | tuple[int, int] = 64
):
    im, m = np.asarray(image), np.asarray(mask, bool)
    if im.shape[:2] != m.shape:
        raise ValueError("image/mask size mismatch")
    if not m.any():
        return None
    if im.ndim == 3 and im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)

    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop, mc = im[y0:y1, x0:x1], m[y0:y1, x0:x1]
    crop = (crop * mc[..., None] if crop.ndim == 3 else crop * mc).astype(im.dtype)

    tw, th = (size, size) if isinstance(size, int) else map(int, size)
    h, w = crop.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    resized = cv2.resize(
        crop, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    )

    canvas = np.zeros(
        (th, tw) if resized.ndim == 2 else (th, tw, resized.shape[2]), dtype=im.dtype
    )  # black pad
    yx = ((th - nh) // 2, (tw - nw) // 2)
    canvas[yx[0] : yx[0] + nh, yx[1] : yx[1] + nw, ...] = resized
    return canvas


def _u8(a):
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float32)
    return np.clip(a * 255 if a.max() <= 1 else a, 0, 255).astype(np.uint8)


def make_classifier_zip(patch_size: int = 64) -> bytes | None:
    rows, buf = [], io.BytesIO()
    with ZipFile(buf, "w") as zf:
        for k in ordered_keys():
            rec = st.session_state.images[k]
            img, M = rec.get("image"), rec.get("masks")
            if (
                img is None
                or not isinstance(M, np.ndarray)
                or M.ndim != 3
                or M.shape[0] == 0
            ):
                continue
            M = (M > 0).astype(np.uint8)
            labs = list(rec.get("labels", [])) + [None] * max(
                0, M.shape[0] - len(rec.get("labels", []))
            )
            inst = stack_to_instances_binary_first(M)
            base = _stem(rec["name"])
            for iid in np.unique(inst)[1:]:
                mm = (inst == int(iid)).astype(np.uint8)
                if not mm.any():
                    continue
                owner = int(np.argmax(M[:, mm.astype(bool)].sum(axis=1)))
                cls = labs[owner]
                if cls in (None, "Remove label"):
                    continue
                patch = extract_masked_cell_patch(img, mm, size=patch_size)
                if patch is None:
                    continue
                name = f"{base}_mask{int(iid)}.png"
                bio = io.BytesIO()
                Image.fromarray(_u8(patch)).save(bio, "PNG")
                zf.writestr(f"images/{name}", bio.getvalue())
                rows.append({"image": name, "mask number": int(iid), "class": cls})
        if rows:
            zf.writestr("labels.csv", pd.DataFrame(rows).to_csv(index=False))
    return buf.getvalue() if rows else None
