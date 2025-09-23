import io
from contextlib import nullcontext
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates
import tifffile as tiff
import hashlib


# --- import helpers ---
from helpers.image_io import load_image, load_masks_any
from helpers.masks import (
    polygon_to_mask,
    toggle_at_point,
    composite_over,
    stack_to_instances,
    _resize_mask_nearest,
)
from helpers.boxes import is_unique_box, boxes_to_fabric_rects, draw_boxes_overlay
from helpers.state_ops import (
    image_key,
    ensure_image,
    current,
    set_current_by_index,
    set_masks,
    add_drawn_mask,
    get_uploaded_images_list,
)

from helpers import config as cfg  # has model paths

st.set_page_config(page_title="Mask Toggle", layout="wide")


def _attach_masks_to_image(rec, new_masks):
    """Normalize/resize and attach masks to the given image record."""
    m = np.asarray(new_masks)
    # normalize shape to (N,H,W) without transpose guesses
    if m.ndim == 4 and m.shape[-1] in (1, 3):
        m = m[..., 0]
    if m.ndim == 3 and m.shape[-1] in (1, 3):
        m = m[..., 0][None, ...]
    if m.ndim == 2:
        m = m[None, ...]
    # resize to image size
    H, W = rec["H"], rec["W"]
    if m.shape[-2:] != (H, W):
        m = np.stack(
            [_resize_mask_nearest(mi.astype(np.uint8), H, W) for mi in m], axis=0
        )
    m = (m > 0).astype(np.uint8)

    # append instead of overwrite (match your Predict behavior)
    if rec.get("masks") is None or rec["masks"].size == 0:
        rec["masks"] = m
        rec["active"] = [True] * m.shape[0]
        rec["history"] = []
    else:
        rec["masks"] = np.concatenate([rec["masks"], m], axis=0)
        rec["active"].extend([True] * m.shape[0])
        rec["history"] = []


def _fingerprint_uploaded(uploaded_file) -> str:
    """
    Returns a stable SHA-256 digest of the uploaded file contents.
    Uses .getvalue() so we don't disturb the file pointer.
    """
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    else:
        # Fallback for generic file-like; read + rewind.
        data = uploaded_file.read()
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
    return hashlib.sha256(data).hexdigest()


# ============================================================
# ---------- Per-image state (single source of truth) --------
# ============================================================

st.session_state.setdefault("images", {})  # {key: record}
st.session_state.setdefault("image_order", [])  # [key,...]
st.session_state.setdefault("current_key", None)
st.session_state.setdefault("base_masks_by_img", {})  # {img_name: (N,H,W) uint8}
st.session_state.setdefault(
    "drawn_masks_by_img", {}
)  # {img_name: [ (H,W) uint8, ... ]}
st.session_state.setdefault(
    "drawn_boxes_by_img", {}
)  # {img_name: [(x0,y0,x1,y1), ...]}
st.session_state.setdefault("_img_files", [])  # list of UploadedFile
st.session_state.setdefault("active_img_idx", 0)  # which image is currently selected
st.session_state.setdefault(
    "mask_uploader_nonce", 0
)  # 🔑 remount file_uploader to clear it
st.session_state.setdefault("_imported_mask_fingerprints", set())  # idempotent imports
st.session_state.setdefault("extra_class_options", [])  # user-added class names
st.session_state.setdefault("current_class", "no class")  # dropdown selection


st.session_state.setdefault("pred_canvas_nonce", 0)
st.session_state.setdefault("edit_canvas_nonce", 0)

from pathlib import Path


def _image_stem(name: str) -> str:
    return Path(name).stem


def _mask_target_stem(mask_name: str) -> str:
    s = Path(mask_name).stem
    return s[:-5] if s.endswith("_mask") else s  # strip trailing "_mask"


# ============================================================
# ---------------------------- Sidebar -----------------------
# ============================================================

with st.sidebar:
    panel = st.radio(
        "Panels",
        ["Upload data", "Create and Edit Masks"],
        key="side_panel",
    )

    # -------- Create & Edit (combined) --------
    if panel == "Create and Edit Masks":

        names = [
            st.session_state.images[k]["name"] for k in st.session_state.image_order
        ]
        cur_idx = (
            st.session_state.image_order.index(st.session_state.current_key)
            if st.session_state.current_key in st.session_state.image_order
            else 0
        )
        st.markdown(f"**Image {cur_idx+1}/{len(names)}:** {names[cur_idx]}")
        nav_container = st.container()
        col1, col2 = nav_container.columns(2)

        if col1.button("◀ Prev", use_container_width=True):
            set_current_by_index(cur_idx - 1)
            st.rerun()
        if col2.button("Next ▶", use_container_width=True):
            set_current_by_index(cur_idx + 1)
            st.rerun()

        scale = st.slider(
            "Display scale",
            1.0,
            3.0,
            st.session_state.get("display_scale", 1.6),
            0.1,
            key="display_scale",  # shared with edit view
        )

        show_overlay = st.toggle("Show mask overlay", value=True, key="show_overlay")

        cur = current()
        if cur is None:
            st.info("Upload an image in **Upload data** first.")
            st.stop()

        H, W = cur["H"], cur["W"]

        # --- Predict / Boxes (left) ---
        st.markdown("**Create and edit masks**")
        mode = st.radio(
            "Select action to perform:",
            ["Draw box", "Remove box", "Draw mask", "Remove mask"],
            index=0,
            key="interaction_mode",
        )

        # Row of 3 equal-width buttons
        row = st.container()  # sidebar-friendly
        c1, c2, c3 = row.columns([1, 1, 1])  # equal widths

        if c1.button("Clear boxes", use_container_width=True):
            cur["boxes"] = []
            st.session_state["pred_canvas_nonce"] += 1
            st.rerun()
        if c2.button("Remove last box", use_container_width=True):
            if cur["boxes"]:
                cur["boxes"].pop()
                st.session_state["pred_canvas_nonce"] += 1
                st.rerun()

        if c3.button("Predict Masks", use_container_width=True):
            boxes = np.array(cur["boxes"], dtype=np.float32)
            if boxes.size == 0:
                st.info("No boxes drawn yet.")
                st.stop()
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            boxes = boxes[(w > 0) & (h > 0)]
            if boxes.size == 0:
                st.info("All boxes were empty.")
                st.stop()

            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
            ckpt_path = cfg.CKPT_PATH
            cfg_path = cfg.CFG_PATH

            sam = build_sam2(
                cfg_path, ckpt_path, device=device, apply_postprocessing=False
            )
            predictor = SAM2ImagePredictor(sam)
            amp = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device == "cuda"
                else nullcontext()
            )
            with torch.inference_mode():
                with amp:
                    predictor.set_image(cur["image"])

            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=True,
            )
            best_idx = scores.argmax(-1)
            row_idx = np.arange(scores.shape[0])
            masks_best = masks[row_idx, best_idx]  # expected (B,H,W)

            # Normalize new masks (no transpose guessing)
            new_masks = np.asarray(masks_best)
            if new_masks.ndim == 4 and new_masks.shape[-1] in (1, 3):
                new_masks = new_masks[..., 0]
            if new_masks.ndim == 3 and new_masks.shape[-1] in (1, 3):
                new_masks = new_masks[..., 0][None, ...]
            if new_masks.ndim == 2:
                new_masks = new_masks[None, ...]
            if new_masks.shape[-2:] != (H, W):
                new_masks = np.stack(
                    [
                        _resize_mask_nearest(mi.astype(np.uint8), H, W)
                        for mi in new_masks
                    ],
                    axis=0,
                )
            new_masks = (new_masks > 0).astype(np.uint8)

            # Append instead of overwrite
            if cur["masks"] is None or cur["masks"].size == 0:
                cur["masks"] = new_masks
                cur["active"] = [True] * new_masks.shape[0]
                cur["history"] = []
            else:
                cur["masks"] = np.concatenate([cur["masks"], new_masks], axis=0)
                cur["active"].extend([True] * new_masks.shape[0])

            # Clear boxes + reset canvas
            cur["boxes"] = []
            st.session_state["pred_canvas_nonce"] += 1

            st.success(
                f"Added {new_masks.shape[0]} masks (total: {cur['masks'].shape[0]}) for {cur['name']}."
            )
            st.rerun()

        # Row of 3 equal-width buttons
        row = st.container()  # sidebar-friendly
        c1, c2 = row.columns([1, 1])  # equal widths

        if c1.button("Clear masks", use_container_width=True):
            if cur["masks"] is not None:
                cur["history"].append(cur["active"].copy())
                cur["active"] = [False] * cur["masks"].shape[0]
                cur["last_click_xy"] = None

        if c2.button("Remove last mask", use_container_width=True):
            if cur["history"]:
                cur["active"] = cur["history"].pop()
                cur["last_click_xy"] = None

        def _stack_to_instances_binary_first(m: np.ndarray) -> np.ndarray:
            """
            Robust stack (N,H,W,[...]) -> instance labels (H,W) uint16.
            - Treats >0 as 1
            - Resolves overlaps by descending area (largest wins), matching your UI behavior.
            """
            m = np.asarray(m)
            if m.ndim == 4 and m.shape[-1] in (1, 3):
                m = m[..., 0]
            if m.ndim == 3 and m.shape[-1] in (1, 3):  # (H,W,1) sneaking in as 'stack'
                m = m[..., 0][None, ...]
            if m.ndim == 2:
                m = m[None, ...]
            bin_stack = (m > 0).astype(np.uint8)

            N, H, W = bin_stack.shape
            areas = bin_stack.reshape(N, -1).sum(axis=1)
            order = np.argsort(-areas)  # largest first

            inst = np.zeros((H, W), dtype=np.uint16)
            curr_id = 1
            for ch in order:
                mm = bin_stack[ch] > 0
                if mm.sum() == 0:
                    continue
                write_here = mm & (inst == 0)
                if write_here.any():
                    inst[write_here] = curr_id
                    curr_id += 1
            return inst

        def _zip_all_masks() -> bytes:
            """
            One TIF per uploaded image: <stem>_mask.tif with (H,W) uint16 instance IDs.
            Pulls from the per-image 'single source of truth' in st.session_state.images.
            """
            buf = io.BytesIO()
            with ZipFile(buf, "w") as zf:
                for img_up in st.session_state.get("_img_files", []):
                    name = img_up.name
                    rec = next(
                        (
                            r
                            for r in st.session_state.images.values()
                            if r["name"] == name
                        ),
                        None,
                    )
                    if rec is None:
                        continue

                    H, W = rec["H"], rec["W"]
                    m = rec.get("masks")

                    if m is None or getattr(m, "size", 0) == 0:
                        inst = np.zeros((H, W), dtype=np.uint16)
                    else:
                        # Option A: export only active masks (recommended)
                        active = rec.get("active", [True] * m.shape[0])
                        if (
                            isinstance(active, list)
                            and len(active) == getattr(m, "shape", (0,))[0]
                        ):
                            m = np.asarray(m)[[i for i, a in enumerate(active) if a]]
                        else:
                            m = np.asarray(m)

                        # Normalize to (N,H,W) binary 0/1, no resizing here (your images already define H,W)
                        if m.ndim == 2:
                            m = m[None, ...]
                        elif m.ndim == 3 and m.shape[-1] == 1:
                            m = m[..., 0]
                        m = (m > 0).astype(np.uint8)
                        if m.shape[-2:] != (H, W):
                            # Shouldn't happen if you attach via _attach_masks_to_image, but guard anyway
                            m = np.stack(
                                [
                                    _resize_mask_nearest(mi.astype(np.uint8), H, W)
                                    for mi in m
                                ],
                                axis=0,
                            )

                        inst = _stack_to_instances_binary_first(m)

                    stem = Path(name).stem
                    tif_bytes = io.BytesIO()
                    tiff.imwrite(
                        tif_bytes,
                        inst,
                        dtype=np.uint16,
                        photometric="minisblack",
                        compression="zlib",
                    )
                    zf.writestr(f"{stem}_mask.tif", tif_bytes.getvalue())

            buf.seek(0)
            return buf.getvalue()

        files = get_uploaded_images_list()
        st.download_button(
            "Download all masks (.zip)",
            data=_zip_all_masks() if files else b"",
            file_name="masks.zip",
            mime="application/zip",
            disabled=(len(files) == 0),
            use_container_width=True,
        )


# ============================================================
# --------------------------- Main area ----------------------
# ============================================================

# -------- Upload panel --------
if panel == "Upload data":

    # ----- uploaded images -----#
    imgs = st.file_uploader(
        "Images (upload first)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        key="u_imgs",
    )
    if imgs:
        # skip duplicates by filename
        existing_names = {
            st.session_state.images[k]["name"] for k in st.session_state.image_order
        }
        for up in imgs:
            if up.name in existing_names:
                # set current to the existing record
                for k in st.session_state.image_order:
                    if st.session_state.images[k]["name"] == up.name:
                        st.session_state.current_key = k
                        break
                continue
            # new image → register
            ensure_image(up)

        # picker for current
        names = [
            st.session_state.images[k]["name"] for k in st.session_state.image_order
        ]
        cur_key = st.session_state.current_key
        cur_idx = (
            st.session_state.image_order.index(cur_key)
            if cur_key in st.session_state.image_order
            else 0
        )
        sel = st.selectbox("Active image", names, index=cur_idx, key="active_img_name")
        set_current_by_index(names.index(sel))

    # ----- uploaded masks -----#
    # masks uploader (enabled only after images exist)
    uploader_key = f"u_masks_np_{st.session_state['mask_uploader_nonce']}"
    up_masks = st.file_uploader(
        "Import masks (.tif/.tiff/.npy/.npz) — images must be uploaded first",
        type=["tif", "tiff", "npy", "npz"],
        key=uploader_key,
        disabled=not st.session_state.image_order,
    )

    if st.session_state.image_order and up_masks is not None:
        fp = _fingerprint_uploaded(up_masks)
        if fp in st.session_state["_imported_mask_fingerprints"]:
            st.info(f"Mask file '{up_masks.name}' was already imported this session.")
        else:
            m = load_masks_any(up_masks)  # (N,H,W) uint8 0/1

            target_stem = _mask_target_stem(up_masks.name)  # strips trailing _mask
            target_key = None
            for k, rec in st.session_state.images.items():
                if _image_stem(rec["name"]) == target_stem:
                    target_key = k
                    break

            if target_key is None:
                st.error(
                    f"No image found for '{up_masks.name}'. Expected image with stem '{target_stem}'."
                )
            else:
                rec = st.session_state.images[target_key]
                _attach_masks_to_image(
                    rec, m
                )  # normalize+resize+append (dedupe-aware; see below)
                st.session_state.current_key = target_key
                st.session_state["_imported_mask_fingerprints"].add(fp)

                st.success(
                    f"Loaded {rec['masks'].shape[0]} total mask(s) into {rec['name']}"
                )

                # ✅ Clear the uploader by remounting it with a new key
                st.session_state["mask_uploader_nonce"] += 1
                st.rerun()

    # ---- Summary table: image–mask pairs ----
    st.subheader("Uploaded image–mask pairs")

    if st.session_state.image_order:
        # table header
        h1, h2, h3, h4 = st.columns([4, 2, 2, 2])
        h1.markdown("**Image**")
        h2.markdown("**Mask present**")
        h3.markdown("**Number of cells**")
        h4.markdown("**Remove**")

        for key in st.session_state.image_order:
            rec = st.session_state.images[key]
            masks = rec.get("masks")
            has_mask = bool(masks is not None and getattr(masks, "size", 0) > 0)
            n_cells = int(masks.shape[0]) if has_mask else 0

            c1, c2, c3, c4 = st.columns([4, 2, 2, 2])
            c1.write(rec["name"])
            c2.write("✅" if has_mask else "❌")
            c3.write(n_cells)
            if c4.button("Remove", key=f"remove_{key}"):
                del st.session_state.images[key]
                st.session_state.image_order.remove(key)
                if st.session_state.current_key == key:
                    st.session_state.current_key = (
                        st.session_state.image_order[0]
                        if st.session_state.image_order
                        else None
                    )
                st.rerun()
    else:
        st.info("No images uploaded yet.")


if panel == "Create and Edit Masks":

    disp_w, disp_h = int(W * scale), int(H * scale)

    # Build the visualized image once
    display_img = cur["image"]
    if show_overlay and cur["masks"] is not None and cur["masks"].size:
        display_img = composite_over(
            cur["image"], cur["masks"], cur["active"], alpha=0.35
        )

    display_for_ui = np.array(
        Image.fromarray(display_img).resize((disp_w, disp_h), Image.BILINEAR)
    )

    if mode == "Draw mask":
        # --- Draw mask mode (auto-close + append to mask stack) ---
        background = Image.fromarray(display_for_ui).convert("RGBA")
        canvas_state = cur["canvas"]  # holds closed_json + processed_count

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 255, 0.3)",
            stroke_width=1,
            stroke_color="white",
            background_color="white",
            background_image=background,
            update_streamlit=True,
            width=disp_w,
            height=disp_h,
            drawing_mode="freedraw",
            point_display_radius=3,
            initial_drawing=canvas_state.get("closed_json"),
            key=f"canvas_edit_{st.session_state['edit_canvas_nonce']}",
        )

        added_any = False
        if canvas_result.json_data is not None:
            data = canvas_result.json_data
            objs = data.get("objects", [])
            start_idx = canvas_state.get("processed_count", 0)
            new_objs = objs[start_idx:]

            changed = False
            for obj in new_objs:
                if obj.get("type") != "path":
                    continue

                # --- auto-close the path if needed ---
                p = obj.get("path", [])
                if p and p[-1][0] != "Z":
                    p.append(["Z"])
                    obj["path"] = p
                    changed = True

                # normalize styling (optional, keeps a consistent look)
                obj["fill"] = "rgba(0, 0, 255, 0.3)"
                obj["stroke"] = obj.get("stroke", "black")
                obj["strokeWidth"] = obj.get("strokeWidth", 3)

                # rasterize at display size -> resize back to original HxW
                mask_disp = polygon_to_mask(obj, disp_h, disp_w)
                mask_orig = _resize_mask_nearest(mask_disp, H, W)

                # append to current image's mask stack + set active=True
                add_drawn_mask(mask_orig)
                added_any = True

            # remember how many objects we already processed
            canvas_state["processed_count"] = len(objs)

            # keep a "closed" version so strokes persist between reruns
            if changed:
                canvas_state["closed_json"] = data

            if added_any:
                # wipe the canvas state and remount with a new key to avoid duplicates
                canvas_state["closed_json"] = None
                canvas_state["processed_count"] = 0
                st.session_state["edit_canvas_nonce"] += 1
                st.rerun()

        # ... handle new paths -> add_drawn_mask(...) + st.rerun() ...

    elif mode == "Draw box":
        # Rectangle canvas (ONLY widget)
        background = Image.fromarray(display_for_ui).convert("RGBA")

        initial_json = boxes_to_fabric_rects(cur["boxes"], scale=scale)
        num_initial = len(initial_json.get("objects", []))

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 255, 0.25)",
            stroke_width=2,
            stroke_color="white",
            background_color="white",
            background_image=background,
            update_streamlit=True,
            width=disp_w,
            height=disp_h,
            drawing_mode="rect",
            point_display_radius=3,
            initial_drawing=initial_json,
            key=f"canvas_pred_{st.session_state['pred_canvas_nonce']}",
        )

        # --- READ newly drawn rectangles and append to cur["boxes"] ---
        added_any = False
        if canvas_result.json_data is not None:
            objs = canvas_result.json_data.get("objects", [])
            # Only process rectangles drawn this run (after the seeded ones)
            for obj in objs[num_initial:]:
                if obj.get("type") != "rect":
                    continue

                left = float(obj.get("left", 0))
                top = float(obj.get("top", 0))
                width = float(obj.get("width", 0)) * float(obj.get("scaleX", 1.0))
                height = float(obj.get("height", 0)) * float(obj.get("scaleY", 1.0))

                x0 = int(round(left / scale))
                y0 = int(round(top / scale))
                x1 = int(round((left + width) / scale))
                y1 = int(round((top + height) / scale))

                # clamp + order
                x0 = max(0, min(W - 1, x0))
                x1 = max(0, min(W, x1))
                y0 = max(0, min(H - 1, y0))
                y1 = max(0, min(H, y1))
                if x1 < x0:
                    x0, x1 = x1, x0
                if y1 < y0:
                    y0, y1 = y1, y0

                box = (x0, y0, x1, y1)
                if is_unique_box(box, cur["boxes"]):
                    cur["boxes"].append(box)
                    added_any = True

        if added_any:
            st.rerun()

    elif mode == "Remove mask":
        # Clickable image (ONLY widget) – do NOT call st.image() again
        click = streamlit_image_coordinates(
            display_for_ui, key="img_click", width=disp_w
        )
        if click:
            x0 = int(round(int(click["x"]) / scale))
            y0 = int(round(int(click["y"]) / scale))
            if 0 <= x0 < W and 0 <= y0 < H and (x0, y0) != cur["last_click_xy"]:
                if cur["masks"] is not None and cur["masks"].size:
                    cur["history"].append(cur["active"].copy())
                    cur["active"] = toggle_at_point(cur["active"], cur["masks"], x0, y0)
                    cur["last_click_xy"] = (x0, y0)
                    st.rerun()

    else:  # mode == "Remove box"
        overlay_img = draw_boxes_overlay(
            display_img, cur["boxes"], alpha=0.25, outline_px=2
        )
        overlay_for_ui = np.array(
            Image.fromarray(overlay_img).resize((disp_w, disp_h), Image.BILINEAR)
        )
        click = streamlit_image_coordinates(
            overlay_for_ui, key="pred_click_remove", width=disp_w
        )
        if click:
            x0c = int(round(int(click["x"]) / scale))
            y0c = int(round(int(click["y"]) / scale))
            hits = [
                i
                for i, (bx0, by0, bx1, by1) in enumerate(cur["boxes"])
                if (bx0 <= x0c < bx1) and (by0 <= y0c < by1)
            ]
            if hits:
                cur["boxes"].pop(hits[-1])
                st.rerun()
