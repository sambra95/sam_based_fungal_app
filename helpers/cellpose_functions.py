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
from sklearn.metrics import r2_score, mean_absolute_error
import zipfile
from datetime import datetime
import pandas as pd
from helpers.state_ops import ordered_keys

# -----------------------------------------------------#
# ---------------- IMAGE PREPROCESSING --------------- #
# -----------------------------------------------------#


# --- small helper: normalization similar to your earlier pipeline ---
def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize uint8 images so output stays in [0, 255] with consistent mean."""
    im = image.astype(np.float32)
    if im.size == 0:
        return im

    mean_val = float(im.mean())
    if mean_val <= 0:
        # fallback: scale to full uint8 range
        rng = float(im.max() - im.min())
        im = (im - im.min()) / rng * 255.0 if rng > 0 else im * 0.0
    else:
        # scale by ratio so mean intensity ≈ 127.5 (mid-gray)
        im = im * (127.5 / mean_val)

    # ensure valid uint8 range
    im = np.clip(im, 0, 255)
    return im.astype(np.uint8)


def preprocess_image_for_cellpose(rec):
    """takes record input and prepares the stored image for cellpose"""

    img = rec["image"]
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim != 2:
        raise ValueError(
            f"Unsupported image shape {img.shape}; expected (H,W) or (H,W,C)"
        )

    im_in = normalize_image(img)

    return im_in


def process_cellpose_mask_outputs_to_single_array(mask_output, H, W):
    """takes mask output from cellpose and converts to a single matrix where different masks are defined by different integers"""

    # ---- convert to single (H,W) label image with contiguous ids 1..N ----
    if mask_output is None or mask_output.size == 0:
        inst = np.zeros((H, W), dtype=np.uint8)
        K = 0
    else:
        a = np.asarray(mask_output)
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

        return inst


# -----------------------------------------------------#
# ---------------- CELLPOSE INFERENCE ---------------- #
# -----------------------------------------------------#


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
    niter=0,
) -> dict:
    """
    Runs Cellpose on rec['image'] and overwrites rec['masks'] with a single (H,W)
    integer label image (0=background, 1..N=instances). Resets rec['labels'].
    """

    im_in = preprocess_image_for_cellpose(rec)

    cell_model = _get_cellpose_model_cached()

    masks_out, flows, styles = cell_model.eval(
        [im_in],
        channels=list(channels),
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size,
        niter=niter,
    )
    mask_output = masks_out[0] if isinstance(masks_out, (list, tuple)) else masks_out

    # set record masks to new predicted mask matrix
    rec["masks"] = process_cellpose_mask_outputs_to_single_array(
        mask_output, rec["H"], rec["W"]
    )
    # clear any labels in the record (no new masks are labelled)
    rec["labels"] = {
        int(i): None for i in np.unique(rec["masks"]) if i != 0
    }  # reset/realign


# -----------------------------------------------------#
# ----------------- CELLPOSE FIGURES ----------------- #
# -----------------------------------------------------#


def _save_fig_to_session(fig, key_prefix: str, dpi: int = 200):
    buf = IO.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    st.session_state[f"{key_prefix}_png"] = buf.getvalue()


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

    # 🔸 Save to session state for later download/use
    _save_fig_to_session(fig, key_prefix="cp_losses", dpi=300)


def _count_instances(lbl):
    return int(np.count_nonzero(np.unique(lbl)))  # #unique nonzero ids


def compare_models_mean_iou_plot(
    images, masks, base_model_name="cyto2", channels=(0, 0)
):
    use_gpu = core.use_gpu()

    # Base/original model
    base = models.CellposeModel(gpu=use_gpu, model_type=base_model_name)
    base_preds, _, _ = base.eval(list(images), channels=list(channels))

    # Fine-tuned model from session BYTES
    tuned = models.CellposeModel(gpu=use_gpu, model_type=base_model_name)
    ft_bytes = st.session_state.get("cellpose_model_bytes")
    if ft_bytes:
        sd = torch.load(IO.BytesIO(ft_bytes), map_location="cpu")
        tuned.net.load_state_dict(sd)
    tuned_preds, _, _ = tuned.eval(list(images), channels=list(channels))

    # IoU per image
    base_ious = [
        metrics.average_precision([gt], [pr])[0][:, 0].mean()
        for gt, pr in zip(masks, base_preds)
    ]
    tuned_ious = [
        metrics.average_precision([gt], [pr])[0][:, 0].mean()
        for gt, pr in zip(masks, tuned_preds)
    ]

    # --- IoU data ---
    labels = ["Original", "Fine-tuned"]
    means = [np.mean(base_ious), np.mean(tuned_ious)]
    sds = [
        np.std(base_ious, ddof=1) if len(base_ious) > 1 else 0.0,
        np.std(tuned_ious, ddof=1) if len(tuned_ious) > 1 else 0.0,
    ]

    # --- Count instances ---
    gt_counts = [_count_instances(m) for m in masks]
    base_counts = [_count_instances(pr) for pr in base_preds]
    tuned_counts = [_count_instances(pr) for pr in tuned_preds]

    lim = max(1, max(gt_counts + base_counts + tuned_counts)) if gt_counts else 1

    # --- Plots ---
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 1: Mean IoU
    ax0.bar(np.arange(2), means, yerr=sds, capsize=5, alpha=0.8, edgecolor="black")
    jitter = 0.12
    ax0.scatter(
        np.zeros(len(base_ious)) + (np.random.rand(len(base_ious)) - 0.5) * jitter,
        base_ious,
        s=18,
        color="k",
        zorder=10,
    )
    ax0.scatter(
        np.ones(len(tuned_ious)) + (np.random.rand(len(tuned_ious)) - 0.5) * jitter,
        tuned_ious,
        s=18,
        color="k",
        zorder=10,
    )
    ax0.set_xticks([0, 1])
    ax0.set_xticklabels(labels)
    ax0.set_ylim(0, 1.05)
    ax0.set_ylabel("Mean IoU")
    ax0.set_title("IoU Comparison")
    ax0.grid(alpha=0.3)

    # Panel 2: Original counts vs GT
    ax1.scatter(gt_counts, base_counts, s=28, alpha=0.85, label="Original")
    ax1.plot([0, lim], [0, lim], ls="--", lw=1, color="gray")
    ax1.set_title("Original vs GT counts")
    ax1.set_xlabel("Ground-truth #instances")
    ax1.set_ylabel("Predicted #instances")
    ax1.set_xlim(-0.5, lim + 0.5)
    ax1.set_ylim(-0.5, lim + 0.5)
    ax1.grid(alpha=0.3)

    # R² & MAE (base)
    if len(gt_counts) > 1:
        r2_base = r2_score(gt_counts, base_counts)
        mae_base = mean_absolute_error(gt_counts, base_counts)
        ax1.text(
            0.05,
            0.95,
            f"R² = {r2_base:.3f}\nMAE = {mae_base:.3f}",
            transform=ax1.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    # Panel 3: Fine-tuned counts vs GT
    ax2.scatter(
        gt_counts,
        tuned_counts,
        s=28,
        alpha=0.85,
        color="tab:orange",
        label="Fine-tuned",
    )
    ax2.plot([0, lim], [0, lim], ls="--", lw=1, color="gray")
    ax2.set_title("Fine-tuned vs GT counts")
    ax2.set_xlabel("Ground-truth #instances")
    ax2.set_xlim(-0.5, lim + 0.5)
    ax2.set_ylim(-0.5, lim + 0.5)
    ax2.grid(alpha=0.3)

    # R² & MAE (tuned)
    if len(gt_counts) > 1:
        r2_tuned = r2_score(gt_counts, tuned_counts)
        mae_tuned = mean_absolute_error(gt_counts, tuned_counts)
        ax2.text(
            0.05,
            0.95,
            f"R² = {r2_tuned:.3f}\nMAE = {mae_tuned:.3f}",
            transform=ax2.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    fig.suptitle("Original vs Fine-tuned Model Comparison", fontsize=13)
    plt.tight_layout()

    # 🔸 Save to session state for later download/use
    _save_fig_to_session(fig, key_prefix="cp_compare_iou", dpi=300)


# -----------------------------------------------------#
# ---------------- FINE TUNE CELLPOSE ---------------- #
# -----------------------------------------------------#


def finetune_cellpose_from_records(
    recs: dict,
    base_model: str,
    epochs=100,
    learning_rate=0.1,
    weight_decay=0.0001,
    nimg_per_epoch=32,
    channels=[0, 0],
):
    images, masks = [], []
    for k in recs.keys():
        images.append(preprocess_image_for_cellpose(recs[k]))
        masks.append(recs[k]["masks"].astype("uint16"))

    train_images, test_images, train_masks, test_masks = train_test_split(
        images, masks, test_size=0.2, random_state=42, shuffle=True
    )

    st.info(
        f"Training on **{len(train_images)} images** "
        f"(+ {len(test_images)} validation images)."
    )

    use_gpu = core.use_gpu()
    _ = io.logger_setup()

    init_model = None if base_model == "scratch" else base_model
    cell_model = models.CellposeModel(gpu=use_gpu, model_type=init_model)
    model_name = f"{base_model}_finetuned.pt"

    with st.spinner("Fine-tuning Cellpose…"):
        new_path, train_losses, test_losses = train.train_seg(
            cell_model.net,
            train_data=train_images,
            train_labels=train_masks,
            test_data=test_images,
            test_labels=test_masks,
            channels=channels,
            n_epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            SGD=True,
            nimg_per_epoch=nimg_per_epoch,
            model_name=model_name,
            save_path=None,
        )

    # stash in session
    buf = IO.BytesIO()
    torch.save(cell_model.net.state_dict(), buf)
    st.session_state["cellpose_model_bytes"] = buf.getvalue()
    st.session_state["cellpose_model_name"] = model_name
    st.session_state["model_to_fine_tune"] = base_model

    return train_losses, test_losses, model_name


def download_cellpose_training_record():
    ss, ok = st.session_state, ordered_keys()
    n_masks = sum(
        (
            int(np.max(m))
            if (
                isinstance(m := ss["images"][k].get("masks"), np.ndarray)
                and m.ndim == 2
                and m.size
            )
            else 0
        )
        for k in ok
    )
    params = dict(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        base_model=ss.get("cp_base_model", "cyto2"),
        epochs=int(ss.get("cp_max_epoch", 100)),
        learning_rate=float(ss.get("cp_lr", 5e-5)),
        weight_decay=float(ss.get("cp_wd", 0.1)),
        batch_size=int(ss.get("cp_batch_size", 1)),
        images_used=len(ok),
        masks_used=n_masks,
    )

    buf = IO.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if ss.get("cellpose_model_bytes"):
            z.writestr("cellpose_model.pt", ss["cellpose_model_bytes"])
        z.writestr(
            "params.csv",
            pd.Series(params)
            .rename_axis("parameter")
            .reset_index(name="value")
            .to_csv(index=False),
        )
        for k in ok:
            r = ss["images"][k]
            img = np.asarray(r["image"])
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            b = IO.BytesIO()
            Image.fromarray(img).save(b, "TIFF")
            name = r.get("name", f"{k}.tif")
            name = name if name.lower().endswith(".tif") else f"{name}.tif"
            z.writestr(f"images/{name}", b.getvalue())
        for k, p in [
            ("cp_losses_png", "plots/cp_losses.png"),
            ("cp_compare_iou_png", "plots/cp_compare_iou.png"),
        ]:
            if k in ss:
                z.writestr(p, ss[k])

    st.download_button(
        "Download Cellpose model, dataset and training metrics (ZIP)",
        data=buf.getvalue(),
        file_name=f"cellpose_training_{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip",
        mime="application/zip",
        use_container_width=True,
        type="primary",
    )
