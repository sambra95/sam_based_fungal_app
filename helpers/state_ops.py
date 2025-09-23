from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
from .masks import _resize_mask_nearest


def image_key(uploaded_file) -> str:
    b = uploaded_file.getvalue()
    return f"{uploaded_file.name}:{len(b)}"


def ensure_image(uploaded_file):
    key = image_key(uploaded_file)
    if key not in st.session_state.images:
        img_np = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img_np, dtype=np.uint8)
        H, W = img_np.shape[:2]
        st.session_state.images[key] = {
            "name": uploaded_file.name,
            "image": img_np,
            "H": H,
            "W": W,
            "masks": None,
            "active": [],
            "history": [],
            "boxes": [],
            "last_click_xy": None,
            "canvas": {"closed_json": None, "processed_count": 0},
            "pred_canvas_init": None,
        }
        st.session_state.image_order.append(key)
    st.session_state.current_key = key


def current() -> dict | None:
    key = st.session_state.current_key
    return None if key is None else st.session_state.images.get(key)


def set_current_by_index(idx: int):
    order = st.session_state.image_order
    if not order:
        return
    st.session_state.current_key = order[idx % len(order)]


def set_masks(masks_u8: np.ndarray):
    cur = current()
    if cur is None:
        return
    m = (masks_u8 > 0).astype(np.uint8)
    cur["masks"] = m
    cur["active"] = [True] * m.shape[0]
    cur["history"] = []


def add_drawn_mask(mask_u8: np.ndarray):
    cur = current()
    if cur is None:
        return
    H, W = cur["H"], cur["W"]
    if mask_u8.shape[:2] != (H, W):
        mask_u8 = _resize_mask_nearest(mask_u8, H, W)
    mask_u8 = (mask_u8 > 0).astype(np.uint8)[None, ...]
    if cur["masks"] is None:
        cur["masks"] = mask_u8
        cur["active"] = [True]
    else:
        cur["masks"] = np.concatenate([cur["masks"], mask_u8], axis=0)
        cur["active"].append(True)


def get_uploaded_images_list():
    files = st.session_state.get("_img_files")
    if not files:
        files = st.session_state.get("u_imgs")
        if files:
            st.session_state["_img_files"] = files
    return files or []
