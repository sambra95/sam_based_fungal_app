"""Segmentation and interactive mask editing for the Streamlit app."""

import hashlib
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from helpers.state_ops import ordered_keys, get_current_rec, set_current_by_index
from helpers.classifying_functions import classes_map_from_labels, create_colour_palette
from helpers.cellpose_functions import normalize_image
from helpers.sam2_functions import segment_with_sam2, _clear_boxes, box_draw_fragment

ImageArray = np.ndarray  # (H, W, 3) uint8/float32
MaskArray = np.ndarray  # (H, W) int / bool
Record = dict[str, any]
Box = dict[str, float]


# -----------------------------------------------------#
# --------------- IMAGE HELPERS  --------------- #
# -----------------------------------------------------#


def create_image_mask_overlay_inner(
    image_bytes,
    mask_bytes,
    image_shape,
    mask_shape,
    classes_items,
    palette_items,
    alpha,
):
    image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(image_shape)
    mask = np.frombuffer(mask_bytes, dtype=np.uint16).reshape(mask_shape)
    classes_map = dict(classes_items)
    palette = dict(palette_items)
    return create_image_mask_overlay(image, mask, classes_map, palette, alpha)


def create_image_mask_overlay(image, mask, classes_map, palette, alpha=0.5):
    """
    image_u8:  uint8 RGB image, shape (H, W, 3)
    mask: uint{8,16,32} label image, shape (H, W), 0=background, 1..N=instances
    classes_map: dict[int -> class_name]
    palette: dict[class_name -> (r,g,b) in 0..1]
    alpha: overlay opacity for filled region
    """
    H, W = image.shape[:2]
    inst = np.asarray(mask)

    if inst.ndim != 2:
        raise ValueError("label_inst must be a 2D label image (H, W)")
    if inst.shape != (H, W):
        # nearest to preserve integer labels
        inst = np.array(
            Image.fromarray(inst).resize((W, H), Image.NEAREST), dtype=inst.dtype
        )

    if inst.size == 0 or not np.any(inst):
        return image

    out = image.astype(np.float32) / 255.0

    ids = np.unique(inst)
    ids = ids[ids != 0]  # skip background

    for iid in ids:
        cls = classes_map.get(int(iid), "__unlabeled__")
        color = np.array(palette.get(cls, palette["__unlabeled__"]), dtype=np.float32)

        mm = inst == iid

        # fill
        a = (mm.astype(np.float32) * alpha)[..., None]
        out = out * (1 - a) + color[None, None, :] * a

        # 1px white edge (simple interior test)
        interior = (
            mm
            & np.roll(mm, 1, 0)
            & np.roll(mm, -1, 0)
            & np.roll(mm, 1, 1)
            & np.roll(mm, -1, 1)
        )
        edge = mm & ~interior
        out[edge] = 1.0

    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


@st.cache_data(show_spinner=False)
def cached_image_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    classes_map: dict,
    palette: dict,
    alpha: float,
) -> np.ndarray:
    # Convert unhashable types to something cacheable
    image_key = image.tobytes()
    mask_key = mask.tobytes()
    classes_key = tuple(sorted(classes_map.items()))
    palette_key = tuple(sorted(palette.items()))

    return create_image_mask_overlay_inner(
        image_key, mask_key, image.shape, mask.shape, classes_key, palette_key, alpha
    )


def create_image_display(rec, scale):
    disp_w, disp_h = int(rec["W"] * scale), int(rec["H"] * scale)

    mask = rec.get("masks")

    if st.session_state.get("show_overlay", False) and mask is not None and mask.any():
        labels = st.session_state.setdefault("all_classes", ["No label"])
        palette = create_colour_palette(labels)
        classes_map = classes_map_from_labels(rec["masks"], rec["labels"])

        base_img = cached_image_mask_overlay(
            rec["image"], rec["masks"], classes_map, palette, alpha=0.35
        )
    else:
        base_img = rec["image"]

    display_for_ui = np.array(
        Image.fromarray(base_img.astype(np.uint8)).resize(
            (disp_w, disp_h), Image.BILINEAR
        )
    )
    return base_img, display_for_ui, disp_w, disp_h


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


def _handle_draw_mask_mode(
    rec: Record,
    display_for_ui: ImageArray,
    disp_w: int,
    disp_h: int,
    key_ns: str,
) -> None:
    """Handle interactions when in 'Draw mask' mode."""
    bg = Image.fromarray(display_for_ui).convert("RGBA")

    # Track display shape per record so we can clear any stale display-space data
    disp_shape = (disp_h, disp_w)
    if rec.get("display_shape") != disp_shape:
        rec["display_shape"] = disp_shape
        # reset any display-space storage tied to the old size
        st.session_state["boxes"] = []

    # unique key per image so Streamlit doesn't reuse chart state incorrectly
    img_hash = hashlib.md5(bg.tobytes()).hexdigest()[:8]
    chart_key = f"{key_ns}_plotly_mask_{img_hash}"

    if "boxes" not in st.session_state:
        st.session_state["boxes"] = []  # if you still want raw polys

    def update_boxes() -> None:
        event = st.session_state.get(chart_key)
        if event is None or event.selection is None:
            return

        if getattr(event.selection, "lasso", None):
            for l in event.selection.lasso:
                xs_plot = l["x"]
                ys_plot = l["y"]

                # (optional) keep polygons for debugging / later use
                clean_poly = {"x": xs_plot, "y": ys_plot}
                if clean_poly not in st.session_state["boxes"]:
                    st.session_state["boxes"].append(clean_poly)

                # Plotly coords (0 at bottom) -> display coords (0 at top)
                xs_disp = xs_plot
                ys_disp = [disp_h - y for y in ys_plot]

                # 1) polygon (display space) -> mask in display resolution
                mask_disp = polygon_xy_to_mask(xs_disp, ys_disp, disp_h, disp_w)

                # 2) resize to ORIGINAL image resolution
                mask_full = np.array(
                    Image.fromarray(mask_disp.astype(np.uint8)).resize(
                        (rec["W"], rec["H"]), Image.NEAREST
                    ),
                    dtype=bool,
                )

                # 3) integrate into rec["masks"]
                inst, new_id = integrate_new_mask(rec["masks"], mask_full)
                if new_id is not None:
                    rec["masks"] = inst
                    rec.setdefault("labels", {})[int(new_id)] = rec["labels"].get(
                        int(new_id), None
                    )

    # Build figure using the shared helper: same size as box mode
    fig = _make_base_figure(bg, disp_w, disp_h, dragmode="lasso")

    st.plotly_chart(
        fig,
        key=chart_key,
        on_select=update_boxes,
        selection_mode="lasso",
        use_container_width=False,  # same behaviour as box mode
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["lasso2d", "select2d", "zoom2d", "pan2d"],
        },
    )


def _handle_draw_box_mode(
    rec: Record,
    display_for_ui: ImageArray,
    disp_w: int,
    disp_h: int,
    key_ns: str,
) -> None:
    """Handle interactions when in 'Draw box' mode."""
    bg = Image.fromarray(display_for_ui).convert("RGBA")
    img_hash = hashlib.md5(bg.tobytes()).hexdigest()[:8]
    chart_key = f"{key_ns}_plotly_{img_hash}"

    box_draw_fragment(
        bg_img=bg,
        disp_w=disp_w,
        disp_h=disp_h,
        chart_key=chart_key,
        rec=rec,
    )


def _handle_remove_mask_mode(base_img: ImageArray, disp_w: int) -> None:
    """Handle interactions when in 'Remove mask' mode."""
    streamlit_image_coordinates(
        base_img,
        key="remove_click",
        width=disp_w,
        on_click=remove_clicked,
    )


def _handle_assign_class_mode(base_img: ImageArray, disp_w: int) -> None:
    """Handle interactions when in 'Assign class' mode."""
    streamlit_image_coordinates(
        base_img,
        key="class_click",
        width=disp_w,
        on_click=assign_clicked,
    )


# -----------------------------------------------------#
# --------------- MASK HELPERS  --------------- #
# -----------------------------------------------------#


# masks can be cut when adding them to the existing array (new masks lose priority).
# therefore, add mask, rextract to see if it is cut, if so, take it out and re-add the largest section
def keep_largest_part(mask: np.ndarray) -> np.ndarray:
    """Return only the largest connected component of a boolean mask."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    lab, n = ndi.label(mask)
    if n == 1:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def integrate_new_mask(original: np.ndarray, new_binary: np.ndarray):
    """
    Add a new mask into a label image.
    - original: (H,W) int labels, 0=background, 1..N instances
    - new_binary: (H,W) boolean mask
    Returns (updated_label_image, new_id or None)
    """
    out = original
    nb = new_binary.astype(bool)
    if nb.ndim != 2 or not nb.any():
        return out, None

    # write only where background
    write = (out == 0) & nb
    if not write.any():
        return out, None

    max_id = int(out.max(initial=0))
    new_id = max_id + 1

    # upcast if needed
    if new_id > np.iinfo(out.dtype).max:
        out = out.astype(np.uint32)
    else:
        out = out.copy()

    out[write] = new_id

    # --- check contiguity: keep only the largest surviving component ---
    mask_new = out == new_id
    mask_new = keep_largest_part(mask_new)
    if not mask_new.any():
        return original, None  # nothing left after check

    out[out == new_id] = 0  # clear possibly cut version
    out[mask_new] = new_id  # reapply only largest part

    return out, new_id


def polygon_xy_to_mask(xs, ys, height, width):
    """Rasterize a polygon given x,y coords into a (height,width) bool mask."""
    img = Image.new("L", (width, height), 0)
    xy = list(zip(xs, ys))
    ImageDraw.Draw(img).polygon(xy, outline=1, fill=1)
    return np.array(img, dtype=bool)


def remove_clicked():
    if not st.session_state["remove_click"]:
        return

    rec = get_current_rec()
    disp_w = st.session_state["disp_w"]

    s = float(disp_w / rec["W"])
    xy = (
        int(round(st.session_state["remove_click"]["x"] / s)),
        int(round(st.session_state["remove_click"]["y"] / s)),
    )

    # ignore click from previous run
    if xy == st.session_state["last_remove_xy"]:
        return

    st.session_state["last_remove_xy"] = xy

    x, y = xy
    m = rec.get("masks")

    iid = int(m[y, x])
    if iid == 0:
        return

    m = m.copy()
    m[m == iid] = 0
    gt = m > iid
    if gt.any():
        m[gt] -= 1

    rec["masks"] = m
    rec["labels"] = {
        (k - 1 if k > iid else k): v
        for k, v in rec.get("labels", {}).items()
        if k != iid
    }
    st.session_state["remove_click"] = False  # prevent reprocessing on rerun


def assign_clicked():
    if not st.session_state["class_click"]:
        return

    rec = get_current_rec()
    disp_w = st.session_state["disp_w"]
    s = float(disp_w / rec["W"])
    xy = (
        int(round(st.session_state["class_click"]["x"] / s)),
        int(round(st.session_state["class_click"]["y"] / s)),
    )

    # ignore click from previous run
    if xy == st.session_state["last_class_xy"]:
        return

    st.session_state["last_class_xy"] = xy

    x, y = xy
    m = rec.get("masks")

    iid = int(m[y, x])
    if iid == 0:
        return

    cur = st.session_state.get("side_current_class")
    labels = rec.setdefault("labels", {})
    if cur == "No label" or cur is None:
        labels.pop(iid, None)
    else:
        labels[iid] = cur

    st.session_state["class_click"] = False  # prevent reprocessing on rerun


# -----------------------------------------------------#
# ------------------ RENDER SIDE BAR ----------------- #
# -----------------------------------------------------#


@st.fragment
def render_cellpose_hyperparameters_fragment():

    # Channels (two ints)
    st.number_input(
        "Channel 1",
        value=st.session_state.get("cp_ch1"),
        step=1,
        format="%d",
        key="w_cp_ch1",
    )
    st.session_state["cp_ch1"] = st.session_state.get("w_cp_ch1")
    st.number_input(
        "Channel 2",
        value=st.session_state.get("cp_ch2"),
        step=1,
        format="%d",
        key="w_cp_ch2",
    )
    st.session_state["cp_ch2"] = st.session_state["w_cp_ch2"]

    # Diameter: auto (None) or manual
    diam_mode = st.selectbox(
        "Diameter mode",
        ["Auto (None)", "Manual"],
        index=(
            0
            if st.session_state.get("cp_diam_mode", "Auto (None)") == "Auto (None)"
            else 1
        ),
        key="w_cp_diam_mode",
        help="Leave as Auto for Cellpose to estimate diameter, or set a manual value.",
    )
    st.session_state["cp_diam_mode"] = diam_mode
    # diam_val = None
    if diam_mode == "Manual":
        diam_val = st.number_input(
            "Manual diameter (pixels)",
            min_value=0.0,
            value=float(st.session_state.get("cp_diameter", 0.0)),
            step=1.0,
            key="w_cp_diameter",
        )
        st.session_state["cp_diameter"] = diam_val

    # Thresholds & size
    cellprob = st.number_input(
        "Cellprob threshold",
        value=float(st.session_state.get("cp_cellprob_threshold")),
        step=0.1,
        key="w_cp_cellprob_threshold",
        help="Higher -> fewer cells.",
    )
    st.session_state["cp_cellprob_threshold"] = cellprob
    flowthr = st.number_input(
        "Flow threshold",
        value=float(st.session_state.get("cp_flow_threshold")),
        step=0.1,
        key="w_cp_flow_threshold",
        help="Lower -> more permissive flows.",
    )
    st.session_state["cp_flow_threshold"] = flowthr
    min_size = st.number_input(
        "Minimum size (pixels)",
        value=int(st.session_state.get("cp_min_size")),
        min_value=0,
        step=10,
        key="w_cp_min_size",
        help="Remove masks smaller than this area.",
    )
    st.session_state["cp_min_size"] = min_size

    niter = st.number_input(
        "Niter",
        value=int(st.session_state["cp_niter"]),
        min_value=0,
        step=10,
        key="w_cp_niter",
        help="Higher values favour longer, stringier, cells.",
    )
    st.session_state["cp_niter"] = niter

    # sync diameter to None when Auto selected
    if st.session_state.get("cp_diam_mode", "Auto (None)") == "Auto (None)":
        st.session_state["cp_diameter"] = None


def render_box_tools_fragment(key_ns="side"):
    rec = get_current_rec()
    c1, c2 = st.columns([1, 1])

    if c1.button("Draw box", use_container_width=True, key=f"{key_ns}_draw_boxes"):
        st.session_state["interaction_mode"] = "Draw box"
        st.rerun()

    if c2.button("Clear boxes", use_container_width=True, key="clear_boxes_button"):
        _clear_boxes(rec)

    if st.button(
        "Generate masks from boxes",
        use_container_width=True,
        key=f"{key_ns}_predict",
        help="Remember to click complete first!",
    ):
        new_masks = segment_with_sam2(rec)
        for mask in new_masks:
            inst, new_id = integrate_new_mask(rec["masks"], mask)
            if new_id is not None:
                rec["masks"] = inst
                rec.setdefault("labels", {})[int(new_id)] = rec["labels"].get(
                    int(new_id), None
                )

        st.session_state["pred_canvas_nonce"] += 1
        st.session_state["edit_canvas_nonce"] += 1
        st.rerun()


def render_mask_tools_fragment(key_ns="side"):
    rec = get_current_rec()
    row = st.container()
    c1, c2 = row.columns([1, 1])

    if c1.button(
        "Draw mask",
        use_container_width=True,
        key=f"{key_ns}_draw_masks",
        help="Click and hold to draw masks",
    ):
        st.session_state["interaction_mode"] = "Draw mask"
        st.rerun()

    if c2.button(
        "Remove mask",
        use_container_width=True,
        key=f"{key_ns}_remove_masks",
        help="Click masks to remove them",
    ):
        st.session_state["interaction_mode"] = "Remove mask"
        st.rerun()

    row = st.container()
    c1, c2 = row.columns([1, 1])

    if c1.button(
        "Clear masks",
        use_container_width=True,
        key=f"{key_ns}_clear_masks",
        help="Remove all masks from image",
    ):
        rec["masks"] = np.zeros((rec["H"], rec["W"]), dtype=np.uint16)
        rec["labels"] = {}
        rec["last_click_xy"] = None
        st.session_state["edit_canvas_nonce"] += 1
        st.rerun()

    if c2.button(
        "Remove last mask", use_container_width=True, key=f"{key_ns}_undo_mask"
    ):
        max_id = int(rec["masks"].max())
        if max_id > 0:
            rec["masks"][rec["masks"] == max_id] = 0
            rec.setdefault("labels", {}).pop(max_id, None)
        st.session_state["pred_canvas_nonce"] += 1
        st.session_state["edit_canvas_nonce"] += 1
        st.rerun()


# -----------------------------------------------------#
# ---------------- RENDER MAIN DISPLAY --------------- #
# -----------------------------------------------------#


@st.fragment
def render_display_and_interact_fragment(key_ns="edit", scale=1.5):
    rec = get_current_rec()
    if rec is None:
        st.warning("Upload an image in **Upload data** first.")
        return

    ok = ordered_keys()
    names = [st.session_state.images[k]["name"] for k in ok]
    reck = st.session_state.current_key
    rec_idx = ok.index(reck) if reck in ok else 0

    slider_key = f"{key_ns}_jump"  # define once, reuse

    c1, c2 = st.columns([1, 4])
    with c1:
        st.toggle("Show masks", key="show_overlay", value=True)
        st.toggle("Normalize image", key="show_normalized", value=False)

    with c2:
        st.info(f"**Image {rec_idx+1}/{len(ok)}:** {names[rec_idx]}")

        # Just read the slider
        jump = st.slider(
            "Image index",
            1,
            len(ok),
            key=slider_key,
            label_visibility="collapsed",
        )

        # Slider → change current image
        if (jump - 1) != rec_idx:
            set_current_by_index(jump - 1)
            st.rerun()

        # ... rest of your code unchanged ...
        rec_for_disp = rec
        if st.session_state.get("show_normalized"):
            im = normalize_image(rec["image"])
            rec_for_disp = dict(rec)
            rec_for_disp["image"] = im

        base_img, display_for_ui, disp_w, disp_h = create_image_display(
            rec_for_disp, scale
        )
        st.session_state["disp_w"] = disp_w
        mode = st.session_state.get("interaction_mode", "Draw box")

        M = rec.get("masks")

        if mode == "Draw mask":
            _handle_draw_mask_mode(
                rec=rec,
                display_for_ui=display_for_ui,
                disp_w=disp_w,
                disp_h=disp_h,
                key_ns=key_ns,
            )
        elif mode == "Draw box":
            _handle_draw_box_mode(
                rec=rec,
                display_for_ui=display_for_ui,
                disp_w=disp_w,
                disp_h=disp_h,
                key_ns=key_ns,
            )
        elif mode == "Remove mask":
            _handle_remove_mask_mode(
                base_img=base_img,
                disp_w=disp_w,
            )
        elif mode == "Assign class":
            _handle_assign_class_mode(
                base_img=base_img,
                disp_w=disp_w,
            )
