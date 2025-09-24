# panels/classify_cells.py
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image


from helpers.masks import polygon_to_mask, _resize_mask_nearest, composite_over
from helpers.boxes import is_unique_box, boxes_to_fabric_rects, draw_boxes_overlay

# panels/mask_editing.py
import numpy as np
import streamlit as st
from helpers.masks import (
    _resize_mask_nearest,
    toggle_at_point,
    stack_to_instances_binary_first,
    get_class_palette,
    composite_over_by_class,
)


from helpers.state_ops import ordered_keys, set_current_by_index, current
from helpers.masks import _resize_mask_nearest
from helpers import config as cfg  # CKPT_PATH, CFG_PATH
from contextlib import nullcontext

# panels/classify_cells.py
import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from helpers.state_ops import ordered_keys
from helpers.masks import stack_to_instances_binary_first


def _stem(name: str) -> str:
    return Path(name).stem


def _to_square_patch(rgb: np.ndarray, patch_size: int = 256) -> np.ndarray:
    """Center-fit keep-aspect into a square canvas."""
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    scale = patch_size / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = np.array(Image.fromarray(rgb).resize((new_w, new_h), Image.BILINEAR))
    canvas = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    y0 = (patch_size - new_h) // 2
    x0 = (patch_size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _build_classifier_dataset(patch_size: int = 256):
    """
    Walk current image records, build per-instance crops using the same
    instance numbering as mask downloads, store in session_state.
    """
    records, patches = [], {}

    for k in ordered_keys():
        rec = st.session_state.images[k]
        img = rec.get("image")
        m = rec.get("masks")
        if img is None or m is None or getattr(m, "size", 0) == 0:
            continue

        H, W = rec["H"], rec["W"]
        m = np.asarray(m)
        # filter to active masks (matches your download logic)
        active = rec.get("active", [True] * (m.shape[0] if m.ndim == 3 else 1))
        if m.ndim == 2:
            m = m[None, ...]
            active = [True]
        elif m.ndim == 3 and m.shape[-1] == 1:
            m = m[..., 0]

        if isinstance(active, list) and len(active) == m.shape[0]:
            keep = [i for i, a in enumerate(active) if a]
            m = m[keep] if keep else np.zeros((0, H, W), dtype=np.uint8)
        m = (m > 0).astype(np.uint8)
        if m.size == 0:
            continue

        # Instance map with SAME numbering as downloads
        inst = stack_to_instances_binary_first(m)  # (H,W) uint16
        if inst.max() == 0:
            continue

        base = _stem(rec["name"])
        for inst_id in range(1, int(inst.max()) + 1):
            mm = inst == inst_id
            if not mm.any():
                continue
            ys, xs = np.where(mm)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1

            crop_rgb = img[y0:y1, x0:x1]
            crop_mask = mm[y0:y1, x0:x1][..., None]
            masked_rgb = (crop_rgb * crop_mask).astype(np.uint8)
            patch_rgb = _to_square_patch(masked_rgb, patch_size=patch_size)

            fname = f"{base}_mask{inst_id}.png"
            bio = io.BytesIO()
            Image.fromarray(patch_rgb).save(bio, format="PNG")
            patches[fname] = bio.getvalue()

            records.append({"image": fname, "mask number": inst_id, "class": None})

    # Stash for main render (and possible download later)
    st.session_state["classifier_records_df"] = pd.DataFrame(records)
    st.session_state["classifier_patches"] = patches


def render_sidebar(*, key_ns: str = "side"):
    """
    Renders the sidebar controls for 'Create and Edit Masks'.
    """

    ok = ordered_keys()
    if not ok:
        st.warning("Upload an image in **Upload data** first.")
        # read from state so caller can no-op safely

    # ⬇️ get the current record (this was missing)
    cur = current()

    names = [st.session_state.images[k]["name"] for k in ok]
    curk = st.session_state.current_key
    cur_idx = ok.index(curk) if curk in ok else 0
    st.markdown(f"**Image {cur_idx+1}/{len(ok)}:** {names[cur_idx]}")

    c1, c2 = st.columns(2)
    if c1.button("◀ Prev", key=f"{key_ns}_prev", use_container_width=True):
        set_current_by_index(cur_idx - 1)
        st.rerun()
    if c2.button("Next ▶", key=f"{key_ns}_next", use_container_width=True):
        set_current_by_index(cur_idx + 1)
        st.rerun()

    st.toggle(
        "Show mask overlay",
        value=st.session_state[f"side_show_overlay"],
        key=f"{key_ns}_show_overlay",
    )

    # --- Class selection & creation (sidebar) ---
    st.markdown("### Classes")

    # keep a global list of labels
    labels = st.session_state.setdefault("all_classes", ["positive", "negative"])
    # current class to assign on click
    st.session_state.setdefault("side_current_class", labels[0])

    st.selectbox(
        "Current class",
        options=labels,
        index=(
            labels.index(st.session_state["side_current_class"])
            if st.session_state["side_current_class"] in labels
            else 0
        ),
        key="side_current_class",
    )

    new_label = st.text_input("Add new class", key="side_new_label")
    if st.button("Add", use_container_width=True, key="side_add_label") and new_label:
        if new_label not in labels:
            labels.append(new_label)
        st.session_state["side_current_class"] = new_label
        st.rerun()

    # One button: build dataset
    if st.button(
        "Create classifier dataset", use_container_width=True, key=f"{key_ns}_build"
    ):
        _build_classifier_dataset(patch_size=256)
        st.success("Classifier dataset created.")


def render_main(
    *,
    key_ns: str = "edit",
):
    """
    Render the Create/Edit Masks UI for the current image record.
    Required:
      cur          : image record dict (from st.session_state.images[...])
      mode         : one of ["Draw mask","Remove mask","Draw box","Remove box"]
    Optional:
      scale        : float (display scaling)
      key_ns       : widget key namespace to avoid collisions
    """

    cur = current()
    if cur is None:
        st.warning("Upload an image in **Upload data** first.")
    else:
        scale = 1.5
        H, W = cur["H"], cur["W"]
        disp_w, disp_h = int(W * scale), int(H * scale)

        # visualize once
        display_img = cur["image"]
    if (
        st.session_state.get("side_show_overlay", True)
        and cur.get("masks") is not None
        and getattr(cur["masks"], "size", 0) > 0
    ):
        # all class labels known so far (e.g., from your sidebar list)
        labels = st.session_state.setdefault("all_classes", ["positive", "negative"])
        palette = get_class_palette(labels)
        classes_map = cur.get("classes", {})  # {instance_id -> class}
        display_img = composite_over_by_class(
            cur["image"], cur["masks"], cur["active"], classes_map, palette, alpha=0.35
        )

        display_for_ui = np.array(
            Image.fromarray(display_img).resize((disp_w, disp_h), Image.BILINEAR)
        )

        # ensure helper state
        ss = st.session_state
        ss.setdefault("pred_canvas_nonce", 0)
        ss.setdefault("edit_canvas_nonce", 0)

        click = streamlit_image_coordinates(
            display_for_ui, key=f"{key_ns}_img_click", width=disp_w
        )

        if (
            click
            and cur.get("masks") is not None
            and getattr(cur["masks"], "size", 0) > 0
        ):
            x0 = int(round(int(click["x"]) / scale))
            y0 = int(round(int(click["y"]) / scale))
            if 0 <= x0 < W and 0 <= y0 < H and (x0, y0) != cur.get("last_click_xy"):

                # ---- build instance map (same ordering as downloads) ----
                m = np.asarray(cur["masks"])
                a = cur.get("active", [True] * (m.shape[0] if m.ndim == 3 else 1))
                if m.ndim == 2:
                    m = m[None, ...]
                    a = [True]
                elif m.ndim == 3 and m.shape[-1] == 1:
                    m = m[..., 0]
                if isinstance(a, list) and len(a) == m.shape[0]:
                    idx = [i for i, t in enumerate(a) if t]
                    m = m[idx] if idx else np.zeros((0, H, W), dtype=np.uint8)
                m = (m > 0).astype(np.uint8)
                if m.size == 0:
                    st.stop()

                inst = stack_to_instances_binary_first(m)  # (H,W) uint16
                inst_id = int(inst[y0, x0])

                # ---- assign class if you clicked on a mask ----
                if inst_id > 0:
                    cur.setdefault("classes", {})  # {instance_id:int -> class:str}
                    current_class = ss.get("side_current_class", None)
                    if current_class:
                        cur["classes"][inst_id] = current_class
                        cur["last_click_xy"] = (x0, y0)
                        st.rerun()
                else:
                    # clicked background: ignore (no delete)
                    cur["last_click_xy"] = (x0, y0)
        else:
            display_img = cur["image"]

    """Main content for the classifier panel."""
    df = st.session_state.get("classifier_records_df")
    if df is None or df.empty:
        st.info("No classifier dataset yet. Use the sidebar to create it.")
        return

    # Display the table exactly as requested
    st.subheader("Segments")
    if cur and cur.get("classes"):
        df = pd.DataFrame(
            [{"mask number": k, "class": v} for k, v in sorted(cur["classes"].items())]
        )
        st.dataframe(df, use_container_width=True)
