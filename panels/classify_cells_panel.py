# panels/classify_cells.py
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from helpers.mask_editing_functions import composite_over_by_class
from helpers.state_ops import ordered_keys, set_current_by_index, current
from helpers.classifying_functions import (
    classes_map_from_labels,
    _add_label_from_input,
    classify_cells_with_densenet,
    color_hex_for,
    palette_from_emojis,
    remove_class_everywhere,
    _rename_class_from_input,
    _row,
)

# ---------- Small utilities ----------


def _ensure_defaults():
    ss = st.session_state
    ss.setdefault("all_classes", ["Remove label"])
    ss.setdefault("side_current_class", ss["all_classes"][0])


def _image_display(rec, scale):
    """Return a resized UI image honoring the overlay toggle + dims."""
    H, W = rec["H"], rec["W"]
    disp_w, disp_h = int(W * scale), int(H * scale)

    M = rec.get("masks")
    has_instances = isinstance(M, np.ndarray) and M.ndim == 2 and M.any()

    if st.session_state.get("show_overlay", False) and has_instances:
        # classes present in this image
        classes_map = classes_map_from_labels(M, rec.get("labels", {}))
        present_classes = sorted(
            {c for c in classes_map.values() if c and c != "Remove label"}
        )
        # keep a stable session color for each present class
        _ = [color_hex_for(c) for c in present_classes]
        # palette from present classes; fall back to global list
        labels_global = st.session_state.setdefault("all_classes", ["Remove label"])
        palette = palette_from_emojis(present_classes or labels_global)
        base_img = composite_over_by_class(
            rec["image"], M, classes_map, palette, alpha=0.35
        )
    else:
        base_img = rec["image"]

    display_for_ui = np.array(
        Image.fromarray(base_img.astype(np.uint8)).resize(
            (disp_w, disp_h), Image.BILINEAR
        )
    )
    return display_for_ui, disp_w, disp_h


# ---------- Sidebar: navigation / overlay toggle (simple, full rerun) ----------


def nav_fragment(key_ns="side"):
    ok = ordered_keys()
    if not ok:
        st.warning("Upload an image in **Upload data** first.")
        return

    rec = current()
    names = [st.session_state.images[k]["name"] for k in ok]
    i = ok.index(st.session_state.current_key)
    st.markdown(f"**Image {i+1}/{len(ok)}:** {names[i]}")

    c1, c2 = st.columns(2)
    if c1.button("◀ Prev", key=f"{key_ns}_prev", use_container_width=True):
        set_current_by_index(i - 1)
        st.rerun()
    if c2.button("Next ▶", key=f"{key_ns}_next", use_container_width=True):
        set_current_by_index(i + 1)
        st.rerun()

    st.toggle("Show mask overlay", key="show_overlay")


# ---------- Sidebar: actions (heavy) ----------


@st.fragment
def classify_actions_fragment():
    rec = current()
    st.button(
        "Classify this image with DenseNet-121",
        use_container_width=True,
        on_click=lambda: _classify_one_and_refresh(rec),
    )

    st.button(
        "Batch classify all images with DenseNet-121",
        key="btn_batch_classify_cellpose",
        use_container_width=True,
        on_click=_batch_classify_and_refresh,
    )


def _classify_one_and_refresh(rec):
    if rec is not None:
        classify_cells_with_densenet(rec)
    st.rerun()


def _batch_classify_and_refresh():
    ok = ordered_keys()
    if not ok:
        return
    n = len(ok)
    pb = st.progress(0.0, text="Starting…")
    for i, k in enumerate(ok, 1):
        classify_cells_with_densenet(st.session_state.images.get(k))
        pb.progress(i / n, text=f"Classified {i}/{n}")
    pb.empty()
    st.rerun()


# ---------- Sidebar: choose current class + clear labels (light) ----------


@st.fragment
def class_selection_fragment():
    _ensure_defaults()

    # Promote any pending class BEFORE widgets are created
    ss = st.session_state
    if "pending_class" in ss:
        pc = ss.pop("pending_class")
        if pc not in ss["all_classes"]:
            ss["all_classes"].append(pc)
        ss["side_current_class"] = pc
    ss.setdefault("side_current_class", ss["all_classes"][0])

    rec = current()
    labels = ss.setdefault("all_classes", ["Remove label"])
    labdict = rec.get("labels", {}) if isinstance(rec.get("labels"), dict) else {}

    # Unlabel row
    _row(
        "Remove label", sum(1 for v in labdict.values() if v is None), key="use_unlabel"
    )

    # Actual classes
    for name in [c for c in labels if c != "Remove label"]:
        _row(name, sum(1 for v in labdict.values() if v == name), key=f"use_{name}")

    st.caption(f"Current click assign: **{ss.get('side_current_class','None')}**")

    if st.button(
        key="clear_labels_btn", use_container_width=True, label="Clear mask labels"
    ):
        rec["labels"] = {int(i): None for i in np.unique(rec["masks"]) if i != 0}
        st.rerun()


# ---------- Sidebar: manage classes (light) ----------


@st.fragment
def class_manage_fragment(key_ns="side"):
    ss = st.session_state
    labels = ss.setdefault("all_classes", ["Remove label"])

    st.markdown("### Manage classes")
    st.text_input(
        "",
        key="side_new_label",
        placeholder="Enter a new class here",
        on_change=_add_label_from_input(labels, ss.get("side_new_label", "")),
    )

    st.text_input(
        "",
        key="delete_new_label",
        placeholder="Delete class here.",
        on_change=remove_class_everywhere(ss.get("delete_new_label", "")),
    )

    editable = [c for c in ss.get("all_classes", []) if c != "Remove label"]
    if not editable:
        st.caption("No classes yet. Add a class above first.")
        return

    c1, c2 = st.columns([1, 2])
    with c1:
        st.selectbox("Class to relabel", options=editable, key=f"{key_ns}_rename_from")
    with c2:
        st.text_input(
            "New label",
            key=f"{key_ns}_rename_to",
            placeholder="Type the new class name and press Enter",
            on_change=_rename_class_from_input(
                f"{key_ns}_rename_from", f"{key_ns}_rename_to"
            ),
        )


# ---------- Main: display ----------


@st.fragment
def display_and_click_fragment(*, key_ns="edit", scale=1.5):
    _ensure_defaults()
    rec = current()
    if rec is None:
        st.warning("Upload an image in **Upload data** first.")
        return

    H, W = rec["H"], rec["W"]
    display_for_ui, disp_w, disp_h = _image_display(rec, scale)

    M = rec.get("masks")
    has_instances = isinstance(M, np.ndarray) and M.ndim == 2 and M.any()

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


# ---------- Rendering functions ----------


def render_sidebar(*, key_ns: str = "side"):
    ok = ordered_keys()
    if not ok:
        st.warning("Upload an image in **Upload data** first.")
        return

    nav_fragment(key_ns)  # full script rerun on change
    st.divider()
    st.markdown("### Classify cells")
    classify_actions_fragment()  # heavy work isolated
    class_selection_fragment()  # light toggles/rows
    class_manage_fragment(key_ns)  # add/delete/rename


def render_main(*, key_ns: str = "edit"):
    display_and_click_fragment(key_ns=key_ns, scale=1.5)
