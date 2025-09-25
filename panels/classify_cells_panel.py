# panels/classify_cells.py
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd

from helpers.mask_editing_functions import (
    get_class_palette,
    composite_over_by_class,
)

from helpers.state_ops import ordered_keys, set_current_by_index, current

from helpers.classifying_functions import classes_map_from_labels, make_classifier_zip


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

    st.toggle(
        "Show mask overlay",
        value=st.session_state[f"side_show_overlay"],
        key=f"{key_ns}_show_overlay",
    )

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

    new_label = st.text_input("Add new class", key="side_new_label")
    if st.button("Add", use_container_width=True, key="side_add_label") and new_label:
        if new_label not in labels:
            labels.append(new_label)
        st.session_state["side_current_class"] = new_label
        st.rerun()

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

    # One button: build dataset + prepare labeled ZIP
    # ---- in your UI (single button) ----
    data = make_classifier_zip(patch_size=256)
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


def render_main(*, key_ns: str = "edit"):
    rec = current()
    if rec is None:
        st.warning("Upload an image in **Upload data** first.")
        return

    scale = 1.5
    H, W = rec["H"], rec["W"]
    disp_w, disp_h = int(W * scale), int(H * scale)

    display_img = rec["image"]
    if (
        st.session_state.get("side_show_overlay", True)
        and isinstance(rec.get("masks"), np.ndarray)
        and rec["masks"].ndim == 3
        and rec["masks"].shape[0] > 0
    ):
        labels = st.session_state.setdefault("all_classes", ["positive", "negative"])
        palette = get_class_palette(labels)
        classes_map = classes_map_from_labels(rec["masks"], rec["labels"])
        display_img = composite_over_by_class(
            rec["image"], rec["masks"], classes_map, palette, alpha=0.35
        )

    display_for_ui = np.array(
        Image.fromarray(display_img).resize((disp_w, disp_h), Image.BILINEAR)
    )
    click = streamlit_image_coordinates(
        display_for_ui, key=f"{key_ns}_img_click", width=disp_w
    )

    if (
        click
        and isinstance(rec.get("masks"), np.ndarray)
        and rec["masks"].ndim == 3
        and rec["masks"].shape[0] > 0
    ):
        x0 = int(round(int(click["x"]) / scale))
        y0 = int(round(int(click["y"]) / scale))
        if 0 <= x0 < W and 0 <= y0 < H and (x0, y0) != rec.get("last_click_xy"):
            m = rec["masks"]
            if m.ndim == 2:
                m = m[None, ...]
            m = (m > 0).astype(np.uint8)

            hits = [i for i in range(m.shape[0]) if m[i, y0, x0] > 0]
            if hits:
                top = hits[-1]
                cur_class = st.session_state.get("side_current_class", None)
                if cur_class is not None:
                    if len(rec["labels"]) < m.shape[0]:
                        rec["labels"].extend([None] * (m.shape[0] - len(rec["labels"])))
                    rec["labels"][top] = (
                        None if cur_class == "Remove label" else cur_class
                    )
                    rec["last_click_xy"] = (x0, y0)
                    st.rerun()
            else:
                rec["last_click_xy"] = (x0, y0)

    # table of all masks with labels (default None)
    N = (
        rec["masks"].shape[0]
        if isinstance(rec.get("masks"), np.ndarray) and rec["masks"].ndim == 3
        else 0
    )
    rec.setdefault("labels", [])
    if len(rec["labels"]) < N:
        rec["labels"].extend([None] * (N - len(rec["labels"])))
    df = pd.DataFrame({"mask_index": list(range(N)), "label": rec["labels"][:N]})
    st.dataframe(df, hide_index=True, use_container_width=True)
