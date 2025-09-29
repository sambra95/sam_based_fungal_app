import os, tempfile, hashlib
import numpy as np
import streamlit as st
import cv2
from skimage.exposure import rescale_intensity
from cellpose import core, io, models, train
import torch
from PIL import Image
import io as IO


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
    channels=(0, 0),
    diameter=None,
    cellprob_threshold=-0.2,
    flow_threshold=0.4,
    min_size=0,
    do_normalize=True,
) -> dict:
    """
    Runs Cellpose on rec['image'] and overwrites rec['masks'] with a single (H,W)
    integer label image (0=background, 1..N=instances). Resets rec['labels'].
    """
    if rec is None or "image" not in rec:
        raise ValueError(
            "segment_rec_with_cellpose: rec must contain an 'image' ndarray"
        )

    img = rec["image"]
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim != 2:
        raise ValueError(
            f"Unsupported image shape {img.shape}; expected (H,W) or (H,W,C)"
        )

    H, W = img.shape[:2]
    rec["H"], rec["W"] = H, W
    im_in = _normalize_for_cellpose(img) if do_normalize else img.astype(np.float32)

    try:
        cell_model = _get_cellpose_model_cached()
    except Exception as e:
        st.error(f"Failed to load Cellpose model: {e}")
        return rec

    try:
        masks_out, flows, styles = cell_model.eval(
            [im_in],
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

    # ---- convert to single (H,W) label image with contiguous ids 1..N ----
    if mask_lbl is None or mask_lbl.size == 0:
        inst = np.zeros((H, W), dtype=np.uint8)
        K = 0
    else:
        a = np.asarray(mask_lbl)
        if a.shape != (H, W):
            # (rare) ensure correct size; nearest preserves labels
            a = np.array(
                Image.fromarray(a).resize((W, H), Image.NEAREST), dtype=a.dtype
            )

        vals = np.unique(a)
        ids = vals[vals > 0]
        if ids.size == 0:
            inst = np.zeros((H, W), dtype=np.uint8)
            K = 0
        else:
            # remap old ids -> 1..K (contiguous)
            K = int(ids.size)
            max_old = int(a.max())
            lut_dtype = np.uint32 if K > np.iinfo(np.uint16).max else np.uint16
            lut = np.zeros(max_old + 1, dtype=lut_dtype)
            lut[ids] = np.arange(1, K + 1, dtype=lut_dtype)
            inst = lut[a]

    rec["masks"] = inst  # (H,W) integer labels
    rec["labels"] = {
        int(i): None for i in np.unique(rec["masks"]) if i != 0
    }  # reset/realign


def _has_cellpose_model():
    # require both bytes and a filename
    return bool(st.session_state.get("cellpose_model_bytes")) and bool(
        st.session_state.get("cellpose_model_name")
    )


def finetune_cellpose_from_records(
    recs: dict,
):

    images = [recs[k]["image"].astype("uint8") for k in recs.keys()]
    images = [np.array(Image.fromarray(img).convert("L")) for img in images]
    masks = [recs[k]["masks"].astype("uint16") for k in recs.keys()]

    use_gpu = core.use_gpu()
    channels = [0, 0]
    _ = io.logger_setup()
    cell_model = models.CellposeModel(
        gpu=use_gpu, model_type=st.session_state["model_to_fine_tune"]
    )

    st.write("model loaded")
    init_model = st.session_state["model_to_fine_tune"]
    model_name = f"{init_model}_finetuned.pt"

    new_path, train_losses, test_losses = train.train_seg(
        cell_model.net,
        train_data=images,
        train_labels=masks,
        test_data=images,
        test_labels=masks,
        channels=channels,
        n_epochs=10,
        learning_rate=0.0005,
        weight_decay=0.1,
        SGD=True,
        nimg_per_epoch=32,
        model_name=model_name,
    )

    st.write("fine tuning complete")

    # also place into session state like the uploader expects

    buf = IO.BytesIO()
    torch.save(cell_model.net.state_dict(), buf)
    st.session_state["cellpose_model_bytes"] = buf.getvalue()
    st.session_state["cellpose_model_name"] = model_name

    st.write("model saved")
