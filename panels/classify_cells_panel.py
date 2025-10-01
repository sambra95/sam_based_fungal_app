# panels/classify_cells.py
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
from tensorflow.keras.applications.densenet import preprocess_input
import cv2

# from helpers.densenet_functions import classify_rec_with_densenet_batched
from helpers.mask_editing_functions import (
    get_class_palette,
    composite_over_by_class,
)

from helpers.state_ops import ordered_keys, set_current_by_index, current

from helpers.classifying_functions import (
    classes_map_from_labels,
    make_classifier_zip,
    extract_masked_cell_patch,
    _add_label_from_input,
    emoji_for,
    palette_from_emojis,
)


# panels/classify_cells.py (sidebar only)

import numpy as np
import streamlit as st
from PIL import Image
import cv2
from tensorflow.keras.applications.densenet import preprocess_input

from helpers.state_ops import ordered_keys, set_current_by_index, current
from helpers.classifying_functions import (
    extract_masked_cell_patch,
    _add_label_from_input,
)

# stable emoji-per-class (also used by overlay if you choose)
EMOJIS = ["🔴", "🟠", "🟡", "🟢", "🔵", "🟣", "🟤", "⚫", "⚪"]


def emoji_for(name: str) -> str:
    ss = st.session_state
    emap = ss.setdefault("class_emojis", {})
    if name and name not in emap:
        emap[name] = EMOJIS[abs(hash(name)) % len(EMOJIS)]
    return emap.get(name, "▫️")


def render_sidebar(*, key_ns: str = "side"):
    ok = ordered_keys()
    if not ok:
        st.warning("Upload an image in **Upload data** first.")
        return

    # ---- promote any class selected in previous run BEFORE widgets are created ----
    ss = st.session_state
    ss.setdefault("all_classes", ["Remove label"])
    if "pending_class" in ss:
        pc = ss.pop("pending_class")
        if pc not in ss["all_classes"]:
            ss["all_classes"].append(pc)
        ss["side_current_class"] = pc
    ss.setdefault("side_current_class", ss["all_classes"][0])

    # ---- navigation / overlay toggle ----
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
    st.divider()

    # ---- DenseNet-121 classify button ----
    st.markdown("### Classify with DenseNet-121")
    if st.button("Classify cells (DenseNet-121)", use_container_width=True):
        model = ss.get("densenet_model")
        if model is None:
            st.warning("Upload a DenseNet-121 classifier in the sidebar first.")
        else:
            M = rec.get("masks")
            if not isinstance(M, np.ndarray) or M.ndim != 2 or not np.any(M):
                st.info("No masks to classify.")
            else:
                all_classes = [
                    c for c in ss.get("all_classes", []) if c != "Remove label"
                ] or ["class0", "class1"]
                ids = [int(v) for v in np.unique(M) if v != 0]
                patches, keep_ids = [], []
                for iid in ids:
                    a = np.asarray(
                        extract_masked_cell_patch(rec["image"], M == iid, size=64)
                    )
                    if a.ndim == 2:
                        a = np.repeat(a[..., None], 3, axis=2)
                    elif a.ndim == 3 and a.shape[2] == 4:
                        a = cv2.cvtColor(a, cv2.COLOR_RGBA2RGB)
                    elif a.ndim == 3 and a.shape[2] == 3:
                        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
                    a = cv2.resize(a, (64, 64), interpolation=cv2.INTER_AREA)
                    patches.append(preprocess_input(a.astype(np.float32)))
                    keep_ids.append(iid)

                if not patches:
                    st.info("No valid patches extracted.")
                else:
                    X = np.stack(patches, axis=0)
                    preds = model.predict(X, verbose=0).argmax(axis=1)
                    for iid, cls_idx in zip(keep_ids, preds):
                        name = (
                            all_classes[int(cls_idx)]
                            if int(cls_idx) < len(all_classes)
                            else str(int(cls_idx))
                        )
                        rec.setdefault("labels", {})[int(iid)] = name
                        if (
                            name
                            and name != "Remove label"
                            and name not in ss["all_classes"]
                        ):
                            ss["all_classes"].append(name)
                        _ = emoji_for(name)  # ensure emoji assigned
                    st.session_state.images[st.session_state.current_key] = rec
                    st.success(f"Classified {len(keep_ids)} cells.")
                    st.rerun()

    st.divider()

    # ---- Simple class manager: add + quick pick with Select buttons ----
    st.markdown("### Assign classes to cell masks")
    labels = ss.setdefault("all_classes", ["Remove label"])

    st.text_input(
        "",
        key="side_new_label",
        placeholder="Type a new class here and press Enter",
        on_change=_add_label_from_input(labels, ss.get("side_new_label", "")),
    )

    st.caption(f"Current click assign: **{ss.get('side_current_class','None')}**")

    st.markdown("#### Assignable Classes")
    labdict = rec.get("labels", {}) if isinstance(rec.get("labels"), dict) else {}

    def _row(name: str, count: int, key: str):
        c1, c2, c3, c4 = st.columns([1, 5, 2, 3])
        c1.write("🧹" if name == "Remove label" else emoji_for(name))
        c2.write(f"**{name}**")
        c3.write(str(count))
        c4.button(
            "Select",
            key=key,
            use_container_width=True,
            on_click=lambda n=name: st.session_state.__setitem__("pending_class", n),
        )

    # Unlabel row
    _row(
        "Remove label", sum(1 for v in labdict.values() if v is None), key="use_unlabel"
    )

    # Actual classes
    for name in [c for c in labels if c != "Remove label"]:
        _row(name, sum(1 for v in labdict.values() if v == name), key=f"use_{name}")


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
        palette = palette_from_emojis(labels_global)  # ← now matches table
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

    # # table of all masks with labels (default None)
    # ids = np.unique(M) if isinstance(M, np.ndarray) and M.ndim == 2 else np.array([])
    # ids = ids[ids != 0]
    # labdict = rec.setdefault("labels", {})  # dict {id->class/None}
    # rows = [{"instance_id": int(i), "label": labdict.get(int(i))} for i in ids]
    # df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["instance_id", "label"])
    # st.dataframe(df, hide_index=True, use_container_width=True)
