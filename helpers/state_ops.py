from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
from .mask_editing_functions import _resize_mask_nearest
import streamlit as st


def ensure_global_state() -> None:
    """Initialize all session-state keys used across panels."""
    ss = st.session_state

    # app-level state
    ss.setdefault("images", {})  # {order_key:int -> record:dict}
    ss.setdefault("name_to_key", {})  # {filename:str -> order_key:int}
    ss.setdefault("current_key", None)  # active order_key
    ss.setdefault("next_ord", 1)  # next order_key to assign
    ss.setdefault("_img_files", [])  # raw UploadedFile refs (optional)
    ss.setdefault("analysis_plots", [])

    # UI defaults / nonces
    ss.setdefault("pred_canvas_nonce", 0)
    ss.setdefault("edit_canvas_nonce", 0)
    ss.setdefault("mask_uploader_nonce", 0)
    ss.setdefault("image_uploader_nonce", 0)
    ss.setdefault("show_overlay", True)
    ss.setdefault("interaction_mode", "Draw box")
    ss.setdefault("side_panel", "Upload data")
    st.session_state.setdefault(f"side_interaction_mode", "Draw box")
    st.session_state.setdefault(f"side_show_overlay", True)

    # Ensure each image record has the expected fields
    # for rec in ss["images"].values():
    #     rec.setdefault("name", "")
    #     rec.setdefault("image", None)  # np.uint8 (H,W,3)
    #     rec.setdefault("H", 0)
    #     rec.setdefault("W", 0)
    #     rec.setdefault("masks", [])  # (N,H,W) uint8 or None
    #     rec.setdefault("boxes", [])  # list of (x0,y0,x1,y1)
    #     rec.setdefault("last_click_xy", None)
    #     rec.setdefault("canvas", {"closed_json": None, "processed_count": 0})


def stem(p: str) -> str:
    return Path(p).stem


def image_key(uploaded_file) -> str:
    b = uploaded_file.getvalue()
    return f"{uploaded_file.name}:{len(b)}"


def ensure_image(uploaded_file):
    name = uploaded_file.name
    m = st.session_state.name_to_key
    imgs = st.session_state.images

    # already have it → focus it
    if name in m:
        st.session_state.current_key = m[name]
        return

    # new record
    img_np = np.array(Image.open(uploaded_file).convert("RGB"), dtype=np.uint8)
    H, W = img_np.shape[:2]
    k = st.session_state.next_ord
    st.session_state.next_ord += 1

    imgs[k] = {
        "name": name,
        "image": img_np,
        "H": H,
        "W": W,
        "masks": np.zeros((0, H, W), dtype=np.uint8),
        "labels": [],
        "boxes": [],
        "last_click_xy": None,
        "canvas": {"closed_json": None, "processed_count": 0},
    }
    m[name] = k
    st.session_state.current_key = k


def ordered_keys():
    return sorted(st.session_state.images.keys())


def current():
    k = st.session_state.get("current_key")
    return st.session_state.images.get(k) if k is not None else None


def set_current_by_index(idx: int):
    ok = ordered_keys()
    if not ok:
        return
    st.session_state.current_key = ok[idx % len(ok)]


def set_masks(masks_u8: np.ndarray):
    cur = current()
    if cur is None:
        return
    m = (masks_u8 > 0).astype(np.uint8)
    cur["masks"].append(m)
    cur["labels"] = [True] * m.shape[0]


def add_drawn_mask(mask_u8: np.ndarray):
    cur = current()
    if cur is None:
        return
    H, W = cur["H"], cur["W"]
    if mask_u8.shape[:2] != (H, W):
        mask_u8 = _resize_mask_nearest(mask_u8, H, W)
    mask_u8 = (mask_u8 > 0).astype(np.uint8)[None, ...]
    cur["masks"] = cur["masks"].append(mask_u8)
    cur["labels"].append(None)


def delete_record(order_key: int):
    rec = st.session_state.images.pop(order_key, None)
    if rec:
        st.session_state.name_to_key.pop(rec["name"], None)
    ok = ordered_keys()
    st.session_state.current_key = ok[0] if ok else None
