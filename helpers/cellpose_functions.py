import os, tempfile, hashlib
import numpy as np
import streamlit as st
import cv2
from cellpose import models
from skimage.exposure import rescale_intensity
from helpers.state_ops import current


# --- small helper: normalization similar to your earlier pipeline ---
def _normalize_for_cellpose(image: np.ndarray) -> np.ndarray:
    im = image.astype(np.float32)
    mean_val = float(np.mean(im)) if im.size else 0.0
    if mean_val <= 0:
        # safe fallback: just scale to [0,1]
        rng = float(im.max() - im.min())
        return (im - im.min()) / rng if rng > 0 else im * 0.0
    meannorm = im * (0.5 / mean_val)
    transformed = 1.0 / (1.0 + meannorm)
    return rescale_intensity(transformed, in_range="image", out_range=(0, 1)).astype(
        np.float32
    )


# --- materialize session model bytes to a stable temp path ---
def _materialize_cellpose_weights_from_session() -> str | None:
    ss = st.session_state
    b = ss.get("cellpose_model_bytes", None)
    name = ss.get("cellpose_model_name", None)
    if not b or not name:
        return None

    h = hashlib.sha1(b).hexdigest()[:12]
    suffix = os.path.splitext(name)[1] or ".npy"
    path = os.path.join(tempfile.gettempdir(), f"cellpose_{h}{suffix}")

    # write once; if the file exists, assume it matches the hash
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(b)
    return path


# --- cache the loaded Cellpose model so we don't reload every call ---
def _get_cellpose_model_cached():
    ss = st.session_state
    # tag tracks which bytes are loaded
    tag = (
        hashlib.sha1(ss["cellpose_model_bytes"]).hexdigest()[:12]
        if ss.get("cellpose_model_bytes")
        else "cyto2"
    )

    if ss.get("cellpose_model_obj") is not None and ss.get("cellpose_model_tag") == tag:
        return ss["cellpose_model_obj"]

    weights_path = _materialize_cellpose_weights_from_session()
    if weights_path:
        model = models.CellposeModel(pretrained_model=weights_path)
    else:
        # fallback built-in weights
        model = models.CellposeModel(pretrained_model="cyto2")

    ss["cellpose_model_obj"] = model
    ss["cellpose_model_tag"] = tag
    return model


def segment_rec_with_cellpose(
    rec: dict,
    *,
    channels=(0, 0),  # grayscale images
    diameter=None,  # let Cellpose estimate if None
    cellprob_threshold=-0.2,
    flow_threshold=0.4,
    min_size=0,
    do_normalize=True,
) -> dict:
    """
    Runs Cellpose using the model stored in st.session_state on rec['image'] and
    overwrites rec['masks'] with a (N,H,W) uint8 stack. Resets rec['labels'].

    Expects:
      rec['image'] : np.ndarray, (H,W) or (H,W,3/4)
      rec['H'], rec['W'] : ints (optional; inferred if missing)

    Returns:
      rec (mutated and also returned for convenience).
    """
    if rec is None or "image" not in rec:
        raise ValueError(
            "segment_rec_with_cellpose: rec must contain an 'image' ndarray"
        )

    img = rec["image"]
    if img.ndim == 3:
        # convert RGB/RGBA to grayscale for channels=[0,0]
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim != 2:
        raise ValueError(
            f"Unsupported image shape {img.shape}; expected (H,W) or (H,W,C)"
        )

    H, W = img.shape[:2]
    rec["H"] = H
    rec["W"] = W

    im_in = _normalize_for_cellpose(img) if do_normalize else img.astype(np.float32)

    # get (cached) model
    try:
        cell_model = _get_cellpose_model_cached()
    except Exception as e:
        st.error(f"Failed to load Cellpose model: {e}")
        return rec

    # run inference (handle both list and single-array inputs)
    try:
        masks_out, flows, styles = cell_model.eval(
            [im_in],  # list form is the most forgiving across versions
            channels=list(channels),
            diameter=diameter,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            min_size=min_size,
        )
        mask_lbl = masks_out[0] if isinstance(masks_out, (list, tuple)) else masks_out
    except Exception as e:
        st.error(f"Cellpose inference failed: {e}")
        return rec

    # convert labeled mask -> (N,H,W) binary stack
    if mask_lbl is None or mask_lbl.size == 0:
        nm = np.zeros((0, H, W), dtype=np.uint8)
    else:
        labels = np.unique(mask_lbl)
        labels = labels[labels > 0]
        if labels.size == 0:
            nm = np.zeros((0, H, W), dtype=np.uint8)
        else:
            # vectorized expansion
            nm = (labels[:, None, None] == mask_lbl[None, :, :]).astype(np.uint8)

    # overwrite masks + reset labels
    rec["masks"] = nm  # shape (N,H,W), uint8 {0,1}
    rec["labels"] = [None] * int(nm.shape[0])  # reset/realign


def _segment_current_rec():
    rec = current()
    if rec is None:
        st.warning("Upload an image first.")
        return
    with st.spinner("Running Cellpose…"):
        segment_rec_with_cellpose(rec)  # overwrites rec['masks'], resets rec['labels']
    # bump any canvas nonce you use so the UI refreshes
    st.session_state["pred_canvas_nonce"] = (
        st.session_state.get("pred_canvas_nonce", 0) + 1
    )
    st.rerun()


def _has_cellpose_model():
    # require both bytes and a filename
    return bool(st.session_state.get("cellpose_model_bytes")) and bool(
        st.session_state.get("cellpose_model_name")
    )
