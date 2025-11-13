from contextlib import nullcontext
import numpy as np
import streamlit as st
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import plotly.graph_objects as go


# duplicated function from mas_editing_functions to avoid circular import
def _make_base_figure(bg_img, disp_w: int, disp_h: int, dragmode: str) -> go.Figure:
    """
    Create a Plotly figure with a background image and fixed pixel size.
    Used by both 'Draw box' and 'Draw mask' modes.
    """
    fig = go.Figure()

    fig.add_layout_image(
        dict(
            source=bg_img,
            xref="x",
            yref="y",
            x=0,
            y=disp_h,
            sizex=disp_w,
            sizey=disp_h,
            sizing="stretch",
            layer="below",
        )
    )

    fig.update_xaxes(visible=False, range=[0, disp_w], constrain="domain")
    fig.update_yaxes(
        visible=False,
        range=[0, disp_h],
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        dragmode=dragmode,
        margin=dict(l=0, r=0, t=0, b=0),
        width=disp_w,
        height=disp_h,
    )

    return fig


def _update_boxes(chart_key: str, rec: dict):
    """Callback run when a selection is made on the Plotly chart."""
    event = st.session_state.get(chart_key)
    sel = getattr(event, "selection", None)
    if not sel or not sel.box:
        return

    # For drawing (display coordinates)
    boxes_display = rec.setdefault("boxes_display", [])
    # For SAM (original image coordinates)
    boxes_orig = rec.setdefault("boxes", [])

    # Try to recover display geometry + scale
    disp_w = st.session_state.get("disp_w")
    if disp_w is not None and rec.get("W"):
        scale = float(disp_w / rec["W"])
        disp_h = int(round(rec["H"] * scale))
    else:
        scale = None
        disp_h = None

    for b in sel.box:
        x0_plot, x1_plot = map(float, b["x"])
        y0_plot, y1_plot = map(float, b["y"])

        # --- 1) Store display-space box for visualization ---
        box_disp = {"x0": x0_plot, "x1": x1_plot, "y0": y0_plot, "y1": y1_plot}
        if box_disp not in boxes_display:
            boxes_display.append(box_disp)

        # --- 2) Also compute original image coordinates, if we know the scale ---
        if scale is None or disp_h is None:
            continue

        # Normalize ordering
        if x1_plot < x0_plot:
            x0_plot, x1_plot = x1_plot, x0_plot
        if y1_plot < y0_plot:
            y0_plot, y1_plot = y1_plot, y0_plot

        # Flip Y: Plotly (0 bottom) -> display image (0 top)
        y0_disp = disp_h - y1_plot
        y1_disp = disp_h - y0_plot
        x0_disp, x1_disp = x0_plot, x1_plot

        # Scale back to original image coordinates
        x0 = int(round(x0_disp / scale))
        x1 = int(round(x1_disp / scale))
        y0 = int(round(y0_disp / scale))
        y1 = int(round(y1_disp / scale))

        # Clamp to image bounds
        x0 = max(0, min(rec["W"] - 1, x0))
        x1 = max(0, min(rec["W"], x1))
        y0 = max(0, min(rec["H"] - 1, y0))
        y1 = max(0, min(rec["H"], y1))

        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        box_orig = (x0, y0, x1, y1)
        if box_orig not in boxes_orig:
            boxes_orig.append(box_orig)


def _make_figure_with_boxes(bg_img, disp_w, disp_h, rec: dict):
    fig = _make_base_figure(bg_img, disp_w, disp_h, dragmode="select")

    for box in rec.get("boxes_display", []):
        fig.add_shape(
            type="rect",
            x0=box["x0"],
            x1=box["x1"],
            y0=box["y0"],
            y1=box["y1"],
            line=dict(color="red", width=2),
            fillcolor="rgba(255,0,0,0.15)",
            layer="above",
        )

    return fig


def _clear_boxes(rec: dict):
    """Clear all boxes for the current record (UI + original coords)."""
    rec["boxes"] = []
    rec["boxes_display"] = []


@st.fragment
def box_draw_fragment(bg_img, disp_w, disp_h, chart_key: str, rec: dict):
    fig = _make_figure_with_boxes(bg_img, disp_w, disp_h, rec)
    st.plotly_chart(
        fig,
        key=chart_key,
        selection_mode="box",
        on_select=lambda: _update_boxes(chart_key, rec),
        use_container_width=False,  # 🔒 respect fig.width / fig.height
    )


def prep_image_for_sam2(img: np.ndarray) -> np.ndarray:
    """preprocess image into correct format for mask prediction with sam2"""
    a = img
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    elif a.ndim == 3 and a.shape[2] == 4:
        a = np.array(Image.fromarray(a).convert("RGB"))
    a = a.astype(np.float32)
    mx = a.max() if a.size else 1.0
    if mx > 1.0:
        a /= 255.0 if mx <= 255 else (65535.0 if mx <= 65535 else mx)
    return a


@st.cache_resource(show_spinner="Loading SAM2 weights…")
def _load_sam2():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    # use package config string (resolved by the installed sam2 package)
    CFG_PATH = "configs/sam2.1/sam2.1_hiera_l.yaml"
    # download checkpoint to local HF cache and use its path
    CKPT_PATH = hf_hub_download(
        repo_id="facebook/sam2.1-hiera-large",
        filename="sam2.1_hiera_large.pt",
    )
    # --- minimal additions end ---

    sam = build_sam2(
        CFG_PATH,
        CKPT_PATH,
        device=device,
        apply_postprocessing=False,  # post-processing not supported with MPS :(
    )
    predictor = SAM2ImagePredictor(sam)
    return predictor, device


def segment_with_sam2(rec: dict):
    """input is record for prediction. boxes to guide prediction will be extracted wtih "boxes" key.
    Return a list of (H,W) boolean masks (best mask per box."""
    boxes = np.asarray(rec.get("boxes", []), dtype=np.float32)
    if boxes.size == 0:
        st.info("No boxes drawn yet.")
        return []
    if boxes.ndim == 1:
        boxes = boxes[None, :]  # (1,4)

    # drop degenerate
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    boxes = boxes[(w > 0) & (h > 0)]
    if boxes.size == 0:
        st.info("All boxes were empty.")
        return []

    predictor, device = _load_sam2()

    img_float = prep_image_for_sam2(rec["image"])
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), amp:
        predictor.set_image(img_float)
        masks, scores, _ = predictor.predict(
            point_coords=None, point_labels=None, box=boxes, multimask_output=True
        )

    # to numpy
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    # normalize shapes:
    if masks.ndim == 3:
        masks = masks[None, ...]
    if scores.ndim == 1:
        scores = scores[None, ...]

    B = scores.shape[0]
    best = scores.argmax(-1)  # (B,)
    masks_best = masks[np.arange(B), best]  # (B,H,W)

    H, W = int(rec["H"]), int(rec["W"])
    out = []
    for mi in masks_best:
        mi = mi > 0
        if mi.shape != (H, W):
            mi = np.array(
                Image.fromarray(mi.astype(np.uint8)).resize((W, H), Image.NEAREST),
                dtype=bool,
            )
        out.append(mi)

    _clear_boxes(rec)
    return out  # list of boalean mask (best mask for each box)
