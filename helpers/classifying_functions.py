# panels/classify_cells.py
import numpy as np
import streamlit as st
from PIL import Image
import io
import pandas as pd
from pathlib import Path
from zipfile import ZipFile


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


def make_classifier_zip(patch_size: int = 256) -> bytes | None:
    rows = []
    buf = io.BytesIO()
    with ZipFile(buf, "w") as zf:
        for k in ordered_keys():
            rec = st.session_state.images[k]
            img, m = rec.get("image"), rec.get("masks")
            if (
                img is None
                or not isinstance(m, np.ndarray)
                or m.ndim != 3
                or m.shape[0] == 0
            ):
                continue
            m = (m > 0).astype(np.uint8)  # (N,H,W)
            N, H, W = m.shape
            labs = list(rec.get("labels", []))
            if len(labs) < N:
                labs.extend([None] * (N - len(labs)))

            inst = stack_to_instances_binary_first(m)  # (H,W) uint16
            ids = np.unique(inst)
            ids = ids[ids != 0]
            base = _stem(rec["name"])

            for iid in ids:
                mm = inst == int(iid)
                if not mm.any():
                    continue
                ys, xs = np.where(mm)
                y0, y1 = ys.min(), ys.max() + 1
                x0, x1 = xs.min(), xs.max() + 1

                # owner mask index by max overlap
                owner = int(np.argmax(m[:, mm].sum(axis=1)))
                cls = labs[owner]
                if cls == "Remove label":
                    cls = None
                if cls is None:  # only include labeled instances
                    continue

                crop_rgb = img[y0:y1, x0:x1]
                crop_mask = mm[y0:y1, x0:x1][..., None]
                patch_rgb = _to_square_patch(
                    (crop_rgb * crop_mask).astype(np.uint8), patch_size
                )

                fname = f"{base}_mask{int(iid)}.png"
                bio = io.BytesIO()
                Image.fromarray(patch_rgb).save(bio, "PNG")
                zf.writestr(f"images/{fname}", bio.getvalue())
                rows.append({"image": fname, "mask number": int(iid), "class": cls})

        if not rows:
            return None
        df = pd.DataFrame(rows)
        zf.writestr("labels.csv", df.to_csv(index=False).encode("utf-8"))

    buf.seek(0)
    return buf.getvalue()
