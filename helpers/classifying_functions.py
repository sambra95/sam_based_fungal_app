# panels/classify_cells.py
import numpy as np
import streamlit as st
import numpy as np

from helpers.state_ops import ordered_keys, current
from helpers.densenet_functions import classify_cells_with_densenet

ss = st.session_state

PALETTE_HEX = [
    "#DC050C",  # 26
    "#5289C7",  # 12
    "#4EB265",  # 15
    "#F7F056",  # 18
    "#882E72",  # 9
    "#E8601C",  # 24
    "#D1BBD7",  # 3
    "#90C987",  # 16
    "#F1932D",  # 22
    "#CAE0AB",  # 17
    "#F6C141",  # 20
]
_DEFAULT_FALLBACK_HEX = "#777777"


def _hex_to_rgb01(hx: str) -> tuple[float, float, float]:
    return tuple(int(hx[i : i + 2], 16) / 255.0 for i in (1, 3, 5))


def color_hex_for(name: str) -> str:
    """
    Deterministic, unique color per class name using session-backed mapping.
    Reuses the palette without hashing collisions.
    """
    if not name or name == "Remove label":
        return _DEFAULT_FALLBACK_HEX
    cmap = st.session_state.setdefault("class_colors", {})
    if name not in cmap:
        # pick the first palette color not currently used; wrap if all are used
        used = set(cmap.values())
        choice = next((hx for hx in PALETTE_HEX if hx not in used), None)
        if choice is None:
            choice = PALETTE_HEX[len(cmap) % len(PALETTE_HEX)]
        cmap[name] = choice
    return cmap[name]


def _color_chip_md(hex_color: str, size: int = 14) -> str:
    return (
        f'<span style="display:inline-block;'
        f"width:{size}px;height:{size}px;margin-top:2px;"
        f"background:{hex_color};border:1px solid rgba(0,0,0,.15);"
        f'border-radius:3px;"></span>'
    )


def _rename_class_from_input(old_key: str, new_key: str):
    """Callback: read selected old class and typed new class from session_state and rename."""

    def _cb():
        ss = st.session_state
        old = ss.get(old_key)
        new = (ss.get(new_key, "") or "").strip()
        if not old or not new or old == new:
            return

        # Keep the selection stable across reruns:
        # set the selectbox's value to the *new* name so Streamlit won't
        # fall back to the first option when `old` disappears from options.
        ss[old_key] = new
        ss[new_key] = ""

        # Validate + apply (this may call st.rerun() internally)
        rename_class_everywhere(old, new)

    return _cb


def _all_image_records():
    """Yield all image records from session state safely."""
    ims = st.session_state.get("images", {}) or {}
    for k in ordered_keys():
        rec = ims.get(k)
        if isinstance(rec, dict):
            yield k, rec


def rename_class_everywhere(old_name: str, new_name: str):
    """
    Rename a class globally across the session. If new_name already exists,
    labels with old_name are reassigned to new_name and old_name is removed.
    This updates:
      - st.session_state['all_classes']
      - per-image rec['labels'] dicts
      - st.session_state['side_current_class'] if it was old_name
      - emoji map st.session_state['class_emojis']
    """
    ss = st.session_state
    if not old_name or old_name == "Remove label":
        st.warning("That class cannot be renamed.")
        return
    new_name = (new_name or "").strip()
    if not new_name or new_name == "Remove label":
        st.warning("Please choose a non-empty class name that isn't reserved.")
        return
    if new_name == old_name:
        st.info("No change needed.")
        return

    # Ensure class list exists
    all_classes = ss.setdefault("all_classes", ["Remove label"])

    # Track whether target already exists (merge)
    target_exists = new_name in all_classes

    # --- Update labels in every image record ---
    changed_labels = 0
    for _, rec in _all_image_records():
        lab = rec.get("labels")
        if isinstance(lab, dict):
            # Re-assign values from old_name -> new_name
            to_update = [iid for iid, cname in lab.items() if cname == old_name]
            for iid in to_update:
                lab[iid] = new_name
            changed_labels += len(to_update)

    # --- Update class list (merge or rename) ---
    if old_name in all_classes:
        all_classes = [c for c in all_classes if c != old_name]  # drop old
    if new_name not in all_classes:
        all_classes.append(new_name)
    # Keep "Remove label" first if you prefer; otherwise keep as-is
    # Re-assign back
    ss["all_classes"] = all_classes

    # --- Update current selection if needed ---
    if ss.get("side_current_class") == old_name:
        ss["side_current_class"] = new_name

    # --- Update emojis ---
    emap = ss.setdefault("class_emojis", {})
    if target_exists:
        # We’re merging into an existing class; keep the target’s emoji,
        # drop the old one if present.
        if old_name in emap:
            emap.pop(old_name, None)
    else:
        # True rename: carry over old emoji if present, otherwise assign fresh later
        if old_name in emap:
            emap[new_name] = emap.pop(old_name)
        else:
            # touch to ensure emoji assignment exists for the new name
            _ = color_hex_for(new_name)

    st.rerun()


def remove_class_everywhere(name: str):
    """
    Delete a class from the session. All masks with this class are unlabelled (set to None).
    Updates:
      - st.session_state['all_classes'] (removes the class)
      - per-image rec['labels'] values (name -> None)
      - st.session_state['side_current_class'] (fallback to "Remove label" if needed)
      - emoji map st.session_state['class_emojis'] (removes entry)
    """
    ss = st.session_state

    # Ensure class list exists and cant remove "Remove label"
    if name == "Remove label":
        return
    all_classes = ss.setdefault("all_classes", ["Remove label"])
    if name not in all_classes:
        return

    # Unlabel everywhere
    changed = 0
    for _, rec in _all_image_records():
        lab = rec.get("labels")
        if isinstance(lab, dict):
            to_update = [iid for iid, cname in lab.items() if cname == name]
            for iid in to_update:
                lab[iid] = None
            changed += len(to_update)

    # Remove from class list & emoji map
    ss["all_classes"] = [c for c in all_classes if c != name]
    ss.setdefault("class_emojis", {}).pop(name, None)

    # Fix current selection if needed
    if ss.get("side_current_class") == name:
        ss["side_current_class"] = "Remove label"
    st.rerun()


def palette_from_emojis(class_names):
    """
    Return {class_name: (r,g,b)} in 0..1 using the fixed palette.
    Includes '__unlabeled__' as white.
    """
    pal = {"__unlabeled__": (1.0, 1.0, 1.0)}
    for n in class_names:
        if not n or n == "Remove label":
            continue
        pal[n] = _hex_to_rgb01(color_hex_for(n))
    return pal


def classes_map_from_labels(masks, labels):
    inst = np.asarray(masks)
    if inst.ndim != 2 or inst.size == 0:
        return {}
    classes_map = {}
    for iid in np.unique(inst):
        if iid == 0:
            continue
        cls = labels.get(int(iid))
        classes_map[int(iid)] = cls if cls not in (None, "") else "Remove label"
    return classes_map


def _row(name: str, count: int, key: str, mode_ns: str = "side"):
    # icon | name | count | select |
    c1, c2, c3, c4 = st.columns([1, 5, 2, 3])
    if name == "Remove label":
        c1.write(" ")
    else:
        c1.markdown(_color_chip_md(color_hex_for(name)), unsafe_allow_html=True)
    c2.write(f"**{name}**")
    c3.write(str(count))

    def _select():
        # pick this class AND switch the main panel to Assign class mode
        st.session_state["pending_class"] = name
        st.session_state[f"interaction_mode"] = "Assign class"

    c4.button(
        "Select",
        key=f"{key}_select",
        use_container_width=True,
        on_click=_select,
    )


def _add_label_from_input(labels, new_label_ss):
    new_label = new_label_ss.strip()
    if not new_label:
        return
    # assumes `labels` is in scope; consider storing it in st.session_state if needed
    if new_label not in labels:
        labels.append(new_label)
    st.session_state["side_current_class"] = new_label
    st.session_state["side_new_label"] = ""


# -----------------------------------------------------#
# ------------- CLASSIFY SIDEBA ACTIONS -------------- #
# -----------------------------------------------------#


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
    """classify masks in the current image"""
    if rec is not None:
        classify_cells_with_densenet(rec)
    st.rerun()


def _batch_classify_and_refresh():
    """classify masks in the all images"""
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


# -----------------------------------------------------#
# ------------- RENDER CLASSIFY SIDEBAR -------------- #
# -----------------------------------------------------#


def class_selection_fragment():

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

    if st.button(
        key="clear_labels_btn", use_container_width=True, label="Clear mask labels"
    ):
        rec["labels"] = {int(i): None for i in np.unique(rec["masks"]) if i != 0}
        st.rerun()


@st.fragment
def class_manage_fragment(key_ns="side"):
    ss = st.session_state
    labels = ss.setdefault("all_classes", ["Remove label"])

    st.markdown("### Add and remove classes")
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
