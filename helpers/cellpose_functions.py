import os
import tempfile
import hashlib
import numpy as np
import streamlit as st
import cv2
from cellpose import core, io, models, train, metrics
import torch
from PIL import Image
import io as IO
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import zipfile
import pandas as pd
from helpers.state_ops import ordered_keys, get_current_rec
from pathlib import Path
import plotly.io as pio
import plotly.graph_objects as go
import subprocess

# -----------------------------------------------------#
# ---------------- IMAGE PREPROCESSING --------------- #
# -----------------------------------------------------#


# --- small helper: normalization similar to your earlier pipeline ---
def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizes image intensities for Cellpose input.
    Scales mean intensity to ~127.5 or full uint8 range if mean <= 0.
    """
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


def preprocess_for_cellpose(rec):
    """takes record input and prepares the stored image for cellpose"""

    img = rec["image"]

    # convert to grayscale if needed
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim != 2:
        raise ValueError(
            f"Unsupported image shape {img.shape}; expected (H,W) or (H,W,C)"
        )

    # normalize
    im_in = normalize_image(img)

    return im_in


def convert_cellpose_mask_to_single_array(mask_output, H, W):
    """Converts Cellpose output mask to single (H,W) label image with contiguous ids 1..N"""

    # handle empty mask case
    if mask_output is None or mask_output.size == 0:
        inst = np.zeros((H, W), dtype=np.uint8)
        K = 0
    # handle standard case
    else:
        a = np.asarray(mask_output)
        if a.shape != (H, W):
            # (rare) ensure correct size; nearest preserves labels
            a = np.array(
                Image.fromarray(a).resize((W, H), Image.NEAREST), dtype=a.dtype
            )
        # remap ids to contiguous 1..K
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
def get_cellpose_weights() -> str | None:
    """writes Cellpose model bytes from session state to a temp file and returns the path"""
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


def get_cellpose_model():
    ss = st.session_state
    tag = (
        hashlib.sha1(ss["cellpose_model_bytes"]).hexdigest()[:12]
        if ss.get("cellpose_model_bytes")
        else "cyto2"
    )

    if ss.get("cellpose_model_obj") is not None and ss.get("cellpose_model_tag") == tag:
        return ss["cellpose_model_obj"]

    weights_path = get_cellpose_weights()
    if weights_path:
        model = models.CellposeModel(
            gpu=core.use_gpu,
            pretrained_model=weights_path,
        )
    else:
        model = models.CellposeModel(
            gpu=core.use_gpu,
            pretrained_model="cyto2",
        )

    ss["cellpose_model_obj"] = model
    ss["cellpose_model_tag"] = tag

    return model


def segment_with_cellpose(
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

    im_in = preprocess_for_cellpose(rec)

    cell_model = get_cellpose_model()

    # reset diameter to None for automatic estimation
    # this is necessary be in the online version of the app only
    if diameter == 0:
        diameter = None

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
    rec["masks"] = convert_cellpose_mask_to_single_array(
        mask_output, rec["H"], rec["W"]
    )
    # clear any labels in the record (no new masks are labelled)
    rec["labels"] = {
        int(i): None for i in np.unique(rec["masks"]) if i != 0
    }  # reset/realign


def segment_with_cellpose_sam(
    rec: dict,
    *,
    channels=(0, 0),
    diameter=None,
    cellprob_threshold=-0.2,
    flow_threshold=0.4,
    min_size=0,
    niter=0,
    use_gpu=core.use_gpu,  # control GPU usage for Cellpose-SAM
) -> dict:
    """
    Runs Cellpose-SAM on rec['image'] and overwrites rec['masks'] with a single (H,W)
    integer label image (0=background, 1..N=instances). Resets rec['labels'].
    """

    # prepare input image for Cellpose
    im_in = preprocess_for_cellpose(rec)

    # handle diameter=0 as "auto" (same behavior as plain Cellpose function)
    if diameter == 0:
        diameter = None

    # create Cellpose-SAM model instance
    cell_model = load_cellpose_sam_model(use_gpu)

    # run model with explicit hyperparameters
    masks_out, flows, styles = cell_model.eval(
        [im_in],
        channels=list(channels),
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size,
        niter=niter,
    )

    # handle list/tuple output
    mask_output = masks_out[0] if isinstance(masks_out, (list, tuple)) else masks_out

    # set record masks to new predicted mask matrix
    rec["masks"] = convert_cellpose_mask_to_single_array(
        mask_output, rec["H"], rec["W"]
    )

    # clear any labels in the record (no new masks are labelled)
    rec["labels"] = {
        int(i): None for i in np.unique(rec["masks"]) if i != 0
    }  # reset/realign

    return rec


HERE = Path(__file__).resolve().parent

# worker is one level above helpers/
WORKER_SCRIPT = str((HERE.parent / "segment_with_cellpose_sam_worker.py").resolve())

# your cellpose4 python (this is what you already used)
CELLPOSE4_PYTHON = "/Users/sambra/miniforge3/envs/cellpose4/bin/python"


def segment_with_cellpose_sam_v4_bridge(
    rec: dict,
    *,
    channels=(0, 0),
    diameter=None,
    cellprob_threshold=-0.2,
    flow_threshold=0.4,
    min_size=0,
    niter=0,
    use_gpu=True,
) -> dict:
    """Call Cellpose-SAM (v4) from fungal_app_env via a worker in cellpose4 env."""

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "input.npz"
        out_path = Path(tmpdir) / "output.npz"

        kwargs = dict(
            channels=channels,
            diameter=diameter,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            min_size=min_size,
            niter=niter,
            use_gpu=use_gpu,
        )

        # Save input record + parameters for the worker
        np.savez_compressed(in_path, rec=rec, kwargs=kwargs)

        cmd = [CELLPOSE4_PYTHON, WORKER_SCRIPT, str(in_path), str(out_path)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Cellpose-SAM worker failed\n"
                f"  return code: {result.returncode}\n"
                f"  command: {result.args}\n\n"
                f"--- STDOUT ---\n{result.stdout}\n"
                f"--- STDERR ---\n{result.stderr}\n"
            )

        # Load updated record from worker *before* the temp dir is deleted
        out = np.load(out_path, allow_pickle=True)
        rec_out = out["rec"].item()

        # mutate the original rec in-place so existing code still works
        rec.clear()
        rec.update(rec_out)

    return rec


# -----------------------------------------------------#
# ----------------- CELLPOSE FIGURES ----------------- #
# -----------------------------------------------------#


def compute_prediction_ious(images, masks, model, channels):
    """
    Evaluate a Cellpose model on a list of images/masks and return the IoU values per image.

    Parameters
    ----------
    images : list of np.ndarray
        Input images to evaluate.
    masks : list of np.ndarray
        Corresponding ground-truth segmentation masks.
    model : cellpose.models.CellposeModel
        A loaded Cellpose model (base or fine-tuned).

    Returns
    -------
    list of float
        IoU value for each image.
    """
    preds, _, _ = model.eval(images, channels=channels)
    ious = [
        metrics.average_precision([gt], [pr])[0][:, 0].mean()
        for gt, pr in zip(masks, preds)
    ]
    return ious


def plot_iou_comparison(base_ious, tuned_ious):
    """Plots a bar chart comparing mean IoU of base and fine-tuned Cellpose models with error bars."""

    # prepare data
    labels = ["Base Model", "Fine-tuned"]
    x = [0, 1]
    means = [np.mean(base_ious), np.mean(tuned_ious)]
    sds = [
        np.std(base_ious, ddof=1) if len(base_ious) > 1 else 0.0,
        np.std(tuned_ious, ddof=1) if len(tuned_ious) > 1 else 0.0,
    ]

    # create figure
    fig = go.Figure(layout=dict(barcornerradius=10))
    fig.add_bar(
        x=x,
        y=means,
        width=0.6,
        error_y=dict(type="data", array=sds, visible=True),
        marker=dict(
            color=[
                "#EBF1F8",
                "#EBF1F8",
            ],
            line=dict(color="#004280", width=2),
        ),
    )

    # add individual data points with jitter
    j = 0.12
    fig.add_scatter(
        x=(np.full(len(base_ious), x[0]) + (np.random.rand(len(base_ious)) - 0.5) * j),
        y=base_ious,
        mode="markers",
        marker=dict(color="#004280", size=6),
    )
    fig.add_scatter(
        x=(
            np.full(len(tuned_ious), x[1]) + (np.random.rand(len(tuned_ious)) - 0.5) * j
        ),
        y=tuned_ious,
        mode="markers",
        marker=dict(color="#004280", size=6),
    )

    # layout settings
    fig.update_layout(
        title="IoU Comparison",
        xaxis=dict(tickmode="array", tickvals=x, ticktext=labels, range=[-0.6, 1.6]),
        yaxis=dict(title="Mean IoU", range=[0, 1.05]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")

    return fig


def plot_pred_vs_true_counts(gt_counts, base_counts, title):
    """Plots predicted vs true counts scatter plot with R² and MAE annotations."""

    # determine plot limits
    lim = max(1, max(gt_counts + base_counts))

    # create figure
    fig = go.Figure()
    fig.add_scatter(
        x=gt_counts,
        y=base_counts,
        mode="markers",
        marker=dict(size=8, opacity=0.85, color="#004280"),
        name="Original",
    )
    fig.add_scatter(
        x=[0, lim],
        y=[0, lim],
        mode="lines",
        line=dict(dash="dash", width=1, color="gray"),
        showlegend=False,
    )
    # add annotations if more than one data point
    if len(gt_counts) > 1:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.05,
            y=0.95,
            showarrow=False,
            text=f"R² = {r2_score(gt_counts, base_counts):.3f}<br>MAE = {mean_absolute_error(gt_counts, base_counts):.3f}",
            bgcolor="white",
            opacity=0.7,
            align="left",
        )

    # layout settings
    fig.update_layout(
        title=title,
        xaxis_title="True number of masks",
        yaxis_title="Predicted number of masks",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )

    # set axes ranges and grid
    fig.update_xaxes(range=[-0.5, lim + 0.5], showgrid=True)
    fig.update_yaxes(
        range=[-0.5, lim + 0.5], showgrid=True, gridcolor="rgba(0,0,0,0.1)"
    )
    return fig


# -----------------------------------------------------#
# ---------------- FINE TUNE CELLPOSE ---------------- #
# -----------------------------------------------------#


@st.cache_resource
def load_base_cellpose_model(base_model: str):
    """Loads a base Cellpose model for fine-tuning."""
    init_model = None if base_model == "scratch" else base_model
    cell_model = models.CellposeModel(gpu=core.use_gpu, model_type=init_model)
    return cell_model


@st.cache_resource
def load_cellpose_sam_model(_use_gpu):  # _ stops streamlit hashing the argument
    return models.CellposeModel(gpu=_use_gpu)


def finetune_cellpose(
    recs: dict,
    base_model: str,
    epochs=100,
    learning_rate=0.1,
    weight_decay=0.0001,
    nimg_per_epoch=32,
    channels=[0, 0],
):
    """Fine-tunes Cellpose on the given records and returns training losses, test losses, and model name."""

    images, masks = [], []
    for k in recs.keys():
        images.append(preprocess_for_cellpose(recs[k]))
        masks.append(recs[k]["masks"].astype("uint16"))

    train_images, test_images, train_masks, test_masks = train_test_split(
        images, masks, test_size=0.2, random_state=42, shuffle=True
    )

    st.info(
        f"Training on **{len(train_images)} images** "
        f"(+ {len(test_images)} validation images)."
    )

    _ = io.logger_setup()

    init_model = None if base_model == "scratch" else base_model
    cell_model = load_base_cellpose_model(init_model)
    model_name = f"{base_model}_finetuned.pt"

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


def is_not_empty_mask(m):
    """returns True if mask is a non-empty numpy array"""
    return isinstance(m, np.ndarray) and m.any()


# Plots
def add_plotly_as_png_to_zip(fig_key, zip_file, out_path, default_w=900, default_h=400):
    """Adds a plotly figure stored in st.session_state[fig_key] as a PNG to the given zip file."""
    fig = st.session_state[fig_key]
    png = pio.to_image(
        fig,
        format="png",
        scale=3,
        width=int(getattr(fig.layout, "width", default_w) or default_w),
        height=int(getattr(fig.layout, "height", default_h) or default_h),
    )
    zip_file.writestr(out_path, png)


def build_cellpose_zip_bytes():
    """Builds a zip file containing the fine-tuned Cellpose model, training parameters,
    images, masks, and plots. Returns the zip file as bytes."""

    ok = ordered_keys()
    ss = st.session_state
    n_masks = sum((int(len(np.unique(ss["images"][k])) - 1)) for k in ok)

    # extract training parameters
    params = dict(
        base_model=ss.get("cp_base_model"),
        epochs=int(ss.get("cp_max_epoch")),
        learning_rate=float(ss.get("cp_learning_rate")),
        weight_decay=float(ss.get("cp_weight_decay")),
        batch_size=int(ss.get("cp_batch_size")),
        images_used=len(ok),
        masks_used=n_masks,
    )

    # Build zip in memory
    buf = IO.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("cellpose_model.pt", ss["cellpose_model_bytes"])

        # Optional: grid search results
        if ss.get("cp_do_gridsearch") and ss.get("cp_grid_results_df") is not None:
            z.writestr(
                "hyperparameter_search.csv",
                ss["cp_grid_results_df"].to_csv(index=False),
            )

        # Training parameters table
        z.writestr(
            "params.csv",
            pd.Series(params)
            .rename_axis("parameter")
            .reset_index(name="value")
            .to_csv(index=False),
        )

        # Images and masks
        for k in ok:
            rec = ss["images"][k]
            img_name = Path(rec.get("name")).stem

            b = IO.BytesIO()
            Image.fromarray(np.asarray(rec["image"])).save(b, "TIFF")
            z.writestr(f"images/{img_name}.tif", b.getvalue())

            c = IO.BytesIO()
            Image.fromarray(np.asarray(rec["masks"])).save(c, "TIFF")
            z.writestr(f"masks/{img_name}_masks.tif", c.getvalue())

        # Plots
        add_plotly_as_png_to_zip(
            "cellpose_training_losses", z, "plots/cellpose_training_losses.png"
        )
        add_plotly_as_png_to_zip(
            "cellpose_iou_comparison", z, "plots/cellpose_iou_comparison.png"
        )
        add_plotly_as_png_to_zip(
            "cellpose_original_counts_comparison",
            z,
            "plots/cellpose_original_counts_comparison.png",
        )
        add_plotly_as_png_to_zip(
            "cellpose_tuned_counts_comparison",
            z,
            "plots/cellpose_tuned_counts_comparison.png",
        )

    return buf.getvalue()


# -----------------------------------------------------#
# ----------     SEGMENTATION FUNCTIONS.     --------- #
# -----------------------------------------------------#


def segment_current_and_refresh():
    """calls cellpose to segment the current image"""
    rec = get_current_rec()
    if rec is not None:
        params = get_cellpose_hparams_from_state()
        segment_with_cellpose(rec, **params)
        st.session_state["edit_canvas_nonce"] += 1
    st.rerun()


def batch_segment_and_refresh():
    """calls cellpose to segment all images with progress bar"""
    ok = ordered_keys()
    params = get_cellpose_hparams_from_state()
    n = len(ok)
    pb = st.progress(0.0, text="Starting…")
    for i, k in enumerate(ok, 1):
        segment_with_cellpose(st.session_state.images.get(k), **params)
        pb.progress(i / n, text=f"Segmented {i}/{n}")


def segment_current_and_refresh_cellpose_sam():
    """calls cellpose to segment the current image"""
    rec = get_current_rec()
    if rec is not None:
        params = get_cellpose_hparams_from_state()
        segment_with_cellpose_sam_v4_bridge(rec, **params)
        st.session_state["edit_canvas_nonce"] += 1
    st.rerun()


def batch_segment_current_and_refresh_cellpose_sam():
    """calls cellpose to segment the current image"""
    ok = ordered_keys()
    n = len(ok)
    params = get_cellpose_hparams_from_state()
    pb = st.progress(0.0, text="Starting…")
    for i, k in enumerate(ok, 1):
        segment_with_cellpose_sam_v4_bridge(st.session_state.images.get(k), **params)
        pb.progress(i / n, text=f"Segmented {i}/{n}")


def get_cellpose_hparams_from_state():
    """calls hparam values from session state"""
    # Build kwargs matching segment_rec_with_cellpose signature
    ch1 = int(st.session_state.get("cp_ch1"))
    ch2 = int(st.session_state.get("cp_ch2"))
    diameter = st.session_state.get("cp_diameter")

    return dict(
        channels=(ch1, ch2),
        diameter=diameter,
        cellprob_threshold=float(st.session_state.get("cp_cellprob_threshold")),
        flow_threshold=float(st.session_state.get("cp_flow_threshold")),
        min_size=int(st.session_state.get("cp_min_size")),
        niter=int(st.session_state.get("cp_niter")),
    )
