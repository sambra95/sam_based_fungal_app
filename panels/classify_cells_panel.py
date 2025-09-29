# panels/classify_cells.py
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
from tensorflow.keras.applications.densenet import preprocess_input

import os
import tempfile

# from helpers.densenet_functions import classify_rec_with_densenet_batched
from helpers.mask_editing_functions import (
    get_class_palette,
    composite_over_by_class,
)
import cv2
from helpers.state_ops import ordered_keys, set_current_by_index, current

from helpers.classifying_functions import (
    classes_map_from_labels,
    make_classifier_zip,
    extract_masked_cell_patch,
    _add_label_from_input,
)


def render_sidebar(*, key_ns: str = "side"):
    """
    Renders the sidebar controls for 'Create and Edit Masks'.
    """

    ok = ordered_keys()
    if not ok:
        st.warning("Upload an image in **Upload data** first.")
        # read from state so caller can no-op safely

    # ⬇️ get the current record (this was missing)
    rec = current()

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

    st.toggle("Show mask overlay", key="show_overlay")

    st.divider()

    # --- Class selection & creation (sidebar) ---
    st.markdown("### Assign classes to cell masks:")

    # keep a global list of labels
    labels = st.session_state.setdefault(
        "all_classes",
        [
            "Remove label",
        ],
    )
    # current class to assign on click
    st.session_state.setdefault("side_current_class", labels[0])

    st.text_input(
        "",
        key="side_new_label",
        placeholder="Type a new class here and press Enter",
        on_change=_add_label_from_input(labels, st.session_state["side_new_label"]),
    )

    st.selectbox(
        "Select class to assign by clicking",
        options=labels,
        index=(
            labels.index(st.session_state["side_current_class"])
            if st.session_state["side_current_class"] in labels
            else 0
        ),
        key="side_current_class",
    )

    # One button: build dataset + prepare labeled ZIP
    # ---- in your UI (single button) ----
    data = make_classifier_zip(patch_size=64)
    st.download_button(
        "Download classifier dataset (zip)",
        data=data or b"",
        file_name="classifier_dataset.zip",
        mime="application/zip",
        use_container_width=True,
        disabled=(data is None),
        help=(
            None
            if data is not None
            else "Assign at least one label to enable download."
        ),
    )

    # # defined elsewhere:
    # model = st.session_state["densenet_model"]

    # def prepare_numpy_for_model(patch: np.ndarray, target_size=(64, 64)) -> np.ndarray:
    #     """
    #     Input:  patch from extract_masked_cell_patch (H,W) or (H,W,C)
    #     Output: batch with shape (1, 64, 64, 3), float32, DenseNet-preprocessed
    #     """
    #     arr = np.asarray(patch)

    #     # Ensure 3 channels
    #     if arr.ndim == 2:
    #         arr = np.repeat(arr[..., None], 3, axis=2)
    #     elif arr.ndim == 3 and arr.shape[2] == 4:
    #         arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    #     elif arr.ndim == 3 and arr.shape[2] == 3:
    #         # Patches from OpenCV are likely BGR -> convert to RGB
    #         arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    #     else:
    #         raise ValueError(f"Unexpected patch shape: {arr.shape}")

    #     # Resize if needed
    #     if tuple(arr.shape[:2]) != tuple(target_size):
    #         interp = (
    #             cv2.INTER_AREA
    #             if max(arr.shape[:2]) > max(target_size)
    #             else cv2.INTER_LINEAR
    #         )
    #         arr = cv2.resize(arr, target_size, interpolation=interp)

    #     # Float32 + ImageNet normalization + batch dim
    #     arr = arr.astype(np.float32)
    #     arr = preprocess_input(arr)
    #     batch = np.expand_dims(arr, axis=0)
    #     return batch

    # def classify_rec_with_densenet(rec: dict, size: int = 64, batch_size: int = 128):
    #     M, img = rec["masks"], rec["image"]

    #     patches = []
    #     patches = [
    #         prepare_numpy_for_model(extract_masked_cell_patch(img, M[i], size=64))
    #         for i in range(M.shape[0])
    #     ]

    #     if not patches:
    #         rec["labels"] = []
    #         return rec

    #     # Predict in small batches to reduce memory pressure on GPU/Metal
    #     preds = [model.predict(p, verbose=0)[0] for p in patches]

    #     rec["labels"] = preds
    #     return rec

    # st.button(
    #     "Classify with Densenet121",
    #     on_click=lambda: (classify_rec_with_densenet(current(), size=64), st.rerun()),
    #     use_container_width=True,
    # )


def render_main(*, key_ns: str = "edit"):

    rec = current()
    if rec is None:
        st.warning("Upload an image in **Upload data** first.")
        return

    scale = 1.5
    H, W = rec["H"], rec["W"]
    disp_w, disp_h = int(W * scale), int(H * scale)

    display_img = rec["image"]
    M = rec.get("masks")
    has_instances = isinstance(M, np.ndarray) and M.ndim == 2 and M.any()
    if st.session_state.get("show_overlay", False) and has_instances:
        labels_global = st.session_state.setdefault(
            "all_classes", ["positive", "negative"]
        )
        palette = get_class_palette(labels_global)
        classes_map = classes_map_from_labels(
            M, rec.get("labels", {})
        )  # dict {id->class/None}
        display_img = composite_over_by_class(
            rec["image"], M, classes_map, palette, alpha=0.35
        )

    display_for_ui = np.array(
        Image.fromarray(display_img.astype(np.uint8)).resize(
            (disp_w, disp_h), Image.BILINEAR
        )
    )
    click = streamlit_image_coordinates(
        display_for_ui, key=f"{key_ns}_img_click", width=disp_w
    )

    if click and has_instances:
        x0 = int(round(int(click["x"]) / scale))
        y0 = int(round(int(click["y"]) / scale))
        if 0 <= x0 < W and 0 <= y0 < H and (x0, y0) != rec.get("last_click_xy"):
            iid = int(M[y0, x0])
            if iid > 0:
                cur_class = st.session_state.get("side_current_class")
                if cur_class is not None:
                    if cur_class == "Remove label":
                        rec.setdefault("labels", {}).pop(iid, None)
                    else:
                        rec.setdefault("labels", {})[iid] = cur_class
                    rec["last_click_xy"] = (x0, y0)
                    st.rerun()
            else:
                rec["last_click_xy"] = (x0, y0)

    # table of all masks with labels (default None)
    ids = np.unique(M) if isinstance(M, np.ndarray) and M.ndim == 2 else np.array([])
    ids = ids[ids != 0]
    labdict = rec.setdefault("labels", {})  # dict {id->class/None}
    rows = [{"instance_id": int(i), "label": labdict.get(int(i))} for i in ids]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["instance_id", "label"])
    st.dataframe(df, hide_index=True, use_container_width=True)
