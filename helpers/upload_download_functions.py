# helpers/image_io.py
from PIL import Image
import io
import numpy as np
import tifffile as tiff
import streamlit as st
from zipfile import ZipFile
from pathlib import Path


def load_npy_mask(file, rec):
    """Read Cellpose *_seg.npy and return a (H,W) label matrix with 0 background, 1..N instances."""
    file = file.read()
    arr = np.load(io.BytesIO(file), allow_pickle=True).item()
    # Cellpose stores masks in dict under 'masks'
    mask = arr["masks"].astype(np.uint16)
    H, W = rec["H"], rec["W"]
    if mask.shape != (H, W):
        # resize if needed
        from PIL import Image

        mask = np.array(
            Image.fromarray(mask).resize((W, H), resample=Image.NEAREST),
            dtype=np.uint16,
        )
    return mask


def load_tif_mask(file, rec):
    """Read a label TIFF and return a (H,W) label matrix with 0 background, 1..N instances."""
    file = file.read()
    mask = tiff.imread(io.BytesIO(file)).astype(np.uint16)

    H, W = rec["H"], rec["W"]
    if mask.shape != (H, W):
        mask = np.array(
            Image.fromarray(mask).resize((W, H), resample=Image.NEAREST),
            dtype=np.uint16,
        )
    return mask


def create_new_record_with_image(uploaded_file):
    name = uploaded_file.name
    m = st.session_state.name_to_key
    imgs = st.session_state.images

    # already have it → focus it
    if name in m:
        st.session_state.current_key = m[name]
        return

    # new record
    img_np = np.array(Image.open(uploaded_file).convert("RGB"), dtype=np.uint16)
    H, W = img_np.shape[:2]
    k = st.session_state.next_ord
    st.session_state.next_ord += 1

    imgs[k] = {
        "name": name,
        "image": img_np,
        "H": H,
        "W": W,
        "masks": np.zeros((H, W), dtype=np.uint16),
        "labels": {},
        "boxes": [],
        "last_click_xy": None,
        "canvas": {"closed_json": None, "processed_count": 0},
    }
    m[name] = k
    st.session_state.current_key = k


def zip_all_masks(images: dict, keys: list[int]) -> bytes:
    buf = io.BytesIO()
    with ZipFile(buf, "w") as zf:
        for k in keys:
            rec = images[k]
            H, W = rec["H"], rec["W"]
            inst = rec.get("masks")

            # fallback to empty label image
            if not isinstance(inst, np.ndarray) or inst.ndim != 2 or inst.size == 0:
                inst = np.zeros((H, W), np.uint16)

            # ensure correct size (nearest keeps integer IDs)
            if inst.shape != (H, W):
                inst = np.array(
                    Image.fromarray(inst).resize((W, H), Image.NEAREST),
                    dtype=inst.dtype,
                )

            # write the single 2D label image unchanged
            b = io.BytesIO()
            tiff.imwrite(
                b,
                inst,
                # dtype=np.uint16,
                photometric="minisblack",
                compression="zlib",
                metadata={"axes": "YX"},  # hint it's a 2D image
            )
            zf.writestr(f"{Path(rec['name']).stem}_masks.tif", b.getvalue())

    buf.seek(0)
    return buf.getvalue()
