import os, tempfile, hashlib
import numpy as np
import streamlit as st
import cv2
from cellpose import core, io, models, train, metrics
import torch
from PIL import Image
import io as IO
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# --- small helper: normalization similar to your earlier pipeline ---
def _normalize_for_cellpose(image: np.ndarray) -> np.ndarray:
    im = image.astype(np.float32)
    mean_val = float(np.mean(im)) if im.size else 0.0
    if mean_val <= 0:
        # safe fallback: just scale to [0,1]
        rng = float(im.max() - im.min())
        return (im - im.min()) / rng if rng > 0 else im * 0.0
    meannorm = im * (0.5 / mean_val)
    return meannorm


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


def _plot_losses(train_losses, test_losses):
    fig = plt.figure(figsize=(6, 3))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="train")

    if test_losses is not None and len(test_losses) == len(train_losses):
        # keep only nonzero test losses (ignore exact zeros, keep negatives)
        test_epochs = [e for e, v in zip(epochs, test_losses) if v != 0]
        test_vals = [v for v in test_losses if v != 0]
        plt.plot(test_epochs, test_vals, label="test")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cellpose training and test losses during fine-tuning")
    plt.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)


def compare_models_mean_iou_plot(
    images, masks, base_model_name="cyto2", channels=(0, 0)
):
    """
    Compare mean IoU between base and fine-tuned Cellpose models
    and plot a bar chart with SD error bars and individual points.
    """
    use_gpu = core.use_gpu()

    # Base (original) model
    base_model = models.CellposeModel(gpu=use_gpu, model_type=base_model_name)
    base_preds, _, _ = base_model.eval(images, channels=channels)

    # Fine-tuned model (from session state)
    tuned_model = models.CellposeModel(
        gpu=use_gpu, pretrained_model=st.session_state["cellpose_model_bytes"]
    )
    tuned_preds, _, _ = tuned_model.eval(images, channels=channels)

    # Compute IoUs
    base_ious = [
        metrics.average_precision([m], [p])[0][:, 0].mean()
        for m, p in zip(masks, base_preds)
    ]
    tuned_ious = [
        metrics.average_precision([m], [p])[0][:, 0].mean()
        for m, p in zip(masks, tuned_preds)
    ]

    # Plot
    fig, ax = plt.subplots(figsize=(4, 4))
    data = [base_ious, tuned_ious]
    labels = ["Base", "Fine-tuned"]
    means = [np.mean(d) for d in data]
    stds = [np.std(d) for d in data]

    ax.bar(labels, means, yerr=stds, capsize=5, alpha=0.7)
    for i, vals in enumerate(data):
        ax.scatter([labels[i]] * len(vals), vals, color="black", zorder=10)

    ax.set_ylabel("Mean IoU")
    ax.set_title("Base vs Fine-tuned Cellpose")
    st.pyplot(fig, use_container_width=True)


def finetune_cellpose_from_records(
    recs: dict,
):

    images = [_normalize_for_cellpose(recs[k]["image"]) for k in recs.keys()]
    masks = [recs[k]["masks"].astype("uint16") for k in recs.keys()]

    train_images, test_images, train_masks, test_masks = train_test_split(
        images, masks, test_size=0.2, random_state=42, shuffle=True
    )

    use_gpu = core.use_gpu()
    channels = [0, 0]
    _ = io.logger_setup()

    init_model = st.session_state["model_to_fine_tune"]
    if init_model == "scratch":
        init_model = None
    cell_model = models.CellposeModel(gpu=use_gpu, model_type=init_model)

    st.write("model loaded")

    if init_model == None:
        init_model = "scratch"
    model_name = f"{init_model}_finetuned.pt"

    new_path, train_losses, test_losses = train.train_seg(
        cell_model.net,
        train_data=train_images,
        train_labels=train_masks,
        test_data=test_images,
        test_labels=test_masks,
        channels=channels,
        n_epochs=20,
        learning_rate=0.00005,
        weight_decay=0.1,
        SGD=True,
        nimg_per_epoch=32,
        model_name=model_name,
        save_path=None,
    )

    st.write("fine tuning complete")

    # also place into session state like the uploader expects

    buf = IO.BytesIO()
    torch.save(cell_model.net.state_dict(), buf)
    st.session_state["cellpose_model_bytes"] = buf.getvalue()
    st.session_state["cellpose_model_name"] = model_name

    st.write("model saved")

    return train_losses, test_losses
