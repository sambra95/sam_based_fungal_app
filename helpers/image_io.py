from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st


def load_image(file) -> np.ndarray:
    img = Image.open(file).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _normalize_masks(m):
    m = np.asarray(m)
    if m.ndim == 2:
        m = m[None, ...]
    elif m.ndim == 4:
        m = m[..., 0] if m.shape[-1] in (1, 3) else m[:, 0, ...]
    return (m > 0).astype(np.uint8)


from pathlib import Path
import numpy as np
import tifffile as tiff
import streamlit as st


# helpers/image_io.py
import io
import numpy as np
import tifffile as tiff

_POSSIBLE_STACK_AXES = ((0, 1, 2), (2, 0, 1))  # try N,H,W then H,W,N


def _as_uint8_binary(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == bool:
        return x.astype(np.uint8)
    # Heuristic: treat any nonzero as 1 (covers 0/255 and arbitrary logits >0)
    return (x > 0).astype(np.uint8)


def _maybe_reorder_to_nhw(arr: np.ndarray) -> np.ndarray:
    """
    Accept (N,H,W), (H,W,N), (H,W,1), (N,H,W,1), (H,W) — return (N,H,W).
    Raises if ambiguous but extremely unlikely with square-ish images.
    """
    a = np.asarray(arr)

    # Squeeze trailing singleton channel
    if a.ndim == 4 and a.shape[-1] == 1:
        a = a[..., 0]

    # 2D → single mask as stack of one
    if a.ndim == 2:
        return a[None, ...]  # (1,H,W)

    # 3D cases
    if a.ndim == 3:
        # (H,W,1)
        if a.shape[-1] == 1:
            return a[..., 0][None, ...]
        # Try as (N,H,W)
        if a.shape[0] < 128 and a.shape[1] >= 8 and a.shape[2] >= 8:
            return a
        # Else assume (H,W,N)
        return np.transpose(a, (2, 0, 1))

    raise ValueError(f"Unsupported mask array shape {a.shape}")


def _load_tiff(fp) -> np.ndarray:
    arr = tiff.imread(fp)
    # Instance map?
    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.integer):
        return _instances_to_stack(arr)
    # Otherwise assume a stack
    arr = _maybe_reorder_to_nhw(arr)
    return _as_uint8_binary(arr)


def _load_npy(fp) -> np.ndarray:
    arr = np.load(fp)  # returns ndarray
    arr = _maybe_reorder_to_nhw(arr)
    return _as_uint8_binary(arr)


def _load_npz(fp) -> np.ndarray:
    z = np.load(fp)
    # Prefer common keys
    for k in ("masks", "mask", "arr_0"):
        if k in z:
            arr = z[k]
            break
    else:
        # fallback: first array in archive
        first_key = list(z.files)[0]
        arr = z[first_key]
    arr = _maybe_reorder_to_nhw(arr)
    return _as_uint8_binary(arr)


def load_masks_any(uploaded_file) -> np.ndarray:
    """
    Read masks from .tif/.tiff (instance label or stack), .npy, or .npz and return (N,H,W) uint8 in {0,1}.
    """
    name = uploaded_file.name.lower()
    data = uploaded_file if hasattr(uploaded_file, "read") else uploaded_file
    # Streamlit UploadedFile is file-like; tifffile/numpy can consume it directly.
    if name.endswith((".tif", ".tiff")):
        return _load_tiff(data)
    if name.endswith(".npy"):
        return _load_npy(data)
    if name.endswith(".npz"):
        return _load_npz(data)
    raise ValueError(f"Unsupported mask file type: {uploaded_file.name}")
