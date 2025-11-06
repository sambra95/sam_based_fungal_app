from pathlib import Path
import streamlit as st


def ensure_global_state() -> None:
    """Initialize all session-state keys used across panels."""
    ss = st.session_state

    # app-level state
    ss.setdefault("images", {})  # {order_key:int -> record:dict}
    ss.setdefault("name_to_key", {})  # {filename:str -> order_key:int}
    ss.setdefault("current_key", None)  # active order_key
    ss.setdefault("next_ord", 1)  # next order_key to assign
    ss.setdefault("analysis_plots", [])
    ss.setdefault("cellpose_model_bytes", None)
    ss.setdefault("cellpose_model_name", None)
    ss.setdefault("densenet_ckpt_bytes", None)
    ss.setdefault("densenet_ckpt_name", None)
    ss.setdefault("side_new_label", "")
    ss.setdefault("show_overlay", True)
    st.session_state.setdefault("show_normalized", False)
    ss.setdefault("interaction_mode", "Remove mask")
    ss.setdefault("side_interaction_mode", "Draw box")

    # cellpose model training defaults
    ss.setdefault("cyto_to_train", "Cyto3")
    ss.setdefault("train_losses", [])
    ss.setdefault("test_losses", [])

    # cellpose inference
    ss.setdefault("cellpose_channels", [0, 0])
    ss.setdefault("cp_ch1", 0)
    ss.setdefault("cp_ch2", 0)
    ss.setdefault("cp_min_size", 0)
    ss.setdefault("cp_niter", 0)
    ss.setdefault("cp_flow_threshold", 0.3)
    ss.setdefault("cp_cellprob_threshold", 0.2)
    ss.setdefault("cp_diameter", 0.0)

    # cellpose training
    ss.setdefault("densenet_ckpt_bytes", None)
    ss.setdefault("dn_input_size", 64)
    ss.setdefault("dn_batch_size", 32)
    ss.setdefault("dn_max_epoch", 100)
    ss.setdefault("dn_val_split", 0.2)
    ss.setdefault("cp_compare_iou_png")

    # densenet
    ss.setdefault("densenet_model", None)

    # image dataset download options
    ss.setdefault("dl_normalize_download", False)

    # UI defaults / nonces
    ss.setdefault("pred_canvas_nonce", 0)
    ss.setdefault("edit_canvas_nonce", 0)
    ss.setdefault("mask_uploader_nonce", 0)
    ss.setdefault("image_uploader_nonce", 0)
    ss.setdefault("side_panel", "Upload data")

    # class defaults
    ss.setdefault("all_classes", ["No label"])
    ss.setdefault("side_current_class", ss["all_classes"][0])
    ss.setdefault("cp_grid_results_df", None)


def stem(p: str) -> str:
    return Path(p).stem


def ordered_keys():
    return sorted(st.session_state.images.keys())


def get_current_rec():
    k = st.session_state.get("current_key")
    return st.session_state.images.get(k) if k is not None else None


def set_current_by_index(idx: int):
    ok = ordered_keys()
    if not ok:
        return
    st.session_state.current_key = ok[idx % len(ok)]
