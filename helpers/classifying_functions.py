# panels/classify_cells.py
import numpy as np
import streamlit as st

from helpers.state_ops import ordered_keys, get_current_rec
from helpers.densenet_functions import classify_cells_with_densenet

ss = st.session_state

class_colour_hexes = [
    "#DC050C",
    "#5289C7",
    "#4EB265",
    "#F7F056",
    "#882E72",
    "#E8601C",
    "#D1BBD7",
    "#90C987",
    "#F1932D",
    "#CAE0AB",
    "#F6C141",
]


def hex_to_rgb01(hx: str) -> tuple[float, float, float]:
    return tuple(int(hx[i : i + 2], 16) / 255.0 for i in (1, 3, 5))


def color_hex_for(name: str) -> str:
    """
    Deterministic, unique color per class name using session-backed mapping.
    """
    if name == "No label":
        return "#E5E5E5"
    cmap = st.session_state.setdefault("class_colors", {})
    if name not in cmap:
        # pick the first palette color not currently used; wrap if all are used
        used = set(cmap.values())
        choice = next((hx for hx in class_colour_hexes if hx not in used), None)
        if choice is None:
            choice = class_colour_hexes[len(cmap) % len(class_colour_hexes)]
        cmap[name] = choice
    return cmap[name]


def color_chip_md(hex_color: str, size: int = 14) -> str:
    return (
        f'<span style="display:inline-block;'
        f"width:{size}px;height:{size}px;margin-top:2px;"
        f"background:{hex_color};border:1px solid rgba(0,0,0,.15);"
        f'border-radius:3px;"></span>'
    )


def rename_class_from_input(old_key: str, new_key: str):
    ss = st.session_state
    old = ss.get(old_key)
    new = (ss.get(new_key, "") or "").strip()
    if not old or not new or old == new:
        return

    # Clear the input box
    ss[new_key] = ""

    # Stash the desired selectbox value for the next run
    ss[f"{old_key}__next_value"] = new

    # Apply the rename (this will st.rerun())
    rename_class_everywhere(old, new)


def yield_all_image_records():
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
    if not old_name or old_name == "No label":
        st.warning("That class cannot be renamed.")
        return
    new_name = (new_name or "").strip()
    if not new_name or new_name == "No label":
        st.warning("Please choose a non-empty class name that isn't reserved.")
        return
    if new_name == old_name:
        st.info("No change needed.")
        return

    # Ensure class list exists
    all_classes = ss.setdefault("all_classes", ["No label"])

    # Track whether target already exists (merge)
    target_exists = new_name in all_classes

    # --- Update labels in every image record ---
    changed_labels = 0
    for _, rec in yield_all_image_records():
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
    # Keep "No label" first if you prefer; otherwise keep as-is
    # Re-assign back
    ss["all_classes"] = all_classes

    # --- Update current selection if needed ---
    if ss.get("side_current_class") == old_name:
        ss["side_current_class"] = new_name

    # after emoji handling
    cmap = ss.setdefault("class_colors", {})
    if new_name in ss["all_classes"]:  # target_exists logic above already computed
        cmap.pop(old_name, None)  # merge: keep target color
    else:
        if old_name in cmap:
            cmap[new_name] = cmap.pop(old_name)
    st.rerun()


def remove_class_everywhere(name: str):
    """
    Delete a class from the session. All masks with this class are unlabelled (set to None).
    Updates:
      - st.session_state['all_classes'] (removes the class)
      - per-image rec['labels'] values (name -> None)
      - st.session_state['side_current_class'] (fallback to "No label" if needed)
      - emoji map st.session_state['class_emojis'] (removes entry)
    """
    ss = st.session_state

    ss.setdefault("class_emojis", {}).pop(name, None)
    ss.setdefault("class_colors", {}).pop(name, None)  # free color

    # Ensure class list exists and cant remove "No label"
    if name == "No label":
        return
    all_classes = ss.setdefault("all_classes", ["No label"])
    if name not in all_classes:
        return

    # Unlabel everywhere
    changed = 0
    for _, rec in yield_all_image_records():
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
        ss["side_current_class"] = "No label"
    st.rerun()


def create_colour_palette(class_names):
    """
    Return {class_name: (r,g,b)} in 0..1 using the fixed palette.
    Includes '__unlabeled__' as white.
    """
    pal = {"__unlabeled__": (1.0, 1.0, 1.0)}
    for n in class_names:
        if not n or n == "No label":
            continue
        pal[n] = hex_to_rgb01(color_hex_for(n))
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
        classes_map[int(iid)] = cls if cls not in (None, "") else "No label"
    return classes_map


def create_row(name: str, key: str, mode_ns: str = "side"):
    # icon | name | count | select |
    c1, c2, c3 = st.columns([1, 5, 5])
    c1.markdown(color_chip_md(color_hex_for(name)), unsafe_allow_html=True)
    c2.write(f"**{name}**")

    def _select():
        # pick this class and switch the main panel to Assign class mode
        st.session_state["pending_class"] = name
        st.session_state["interaction_mode"] = "Assign class"

    c3.button(
        "Assign label",
        key=f"{key}_select",
        use_container_width=True,
        on_click=_select,
        help="Click masks to label cells",
    )


def add_label_from_input(labels, new_label_ss):
    new_label = new_label_ss.strip()
    if not new_label:
        return
    # assumes `labels` is in scope; consider storing it in st.session_state if needed
    if new_label not in labels:
        labels.append(new_label)
    st.session_state["side_current_class"] = new_label
    st.session_state["side_new_label"] = ""


# -----------------------------------------------------#
# ------------- CLASSIFY SIDEBAR ACTIONS -------------- #
# -----------------------------------------------------#


@st.fragment
def classify_actions_fragment():
    rec = get_current_rec()

    col1, col2 = st.columns(2)
    col1.button(
        "Classify cells",
        use_container_width=True,
        on_click=lambda: classify_cells_with_densenet(rec),
        help="Classify all masks in this image with the loaded Densenet121 model.",
        disabled=st.session_state["densenet_model"] == None,
    )

    col2.button(
        "Batch classify cells",
        key="btn_batch_classify_cellpose",
        use_container_width=True,
        on_click=batch_classify_and_refresh,
        help="Batch classify all masks in all images with the loaded Densenet121 model.",
        disabled=st.session_state["densenet_model"] == None,
    )


def batch_classify_and_refresh():
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

    rec = get_current_rec()
    labels = ss.setdefault("all_classes", ["No label"])

    # Unlabel row
    create_row("No label", key="use_unlabel")

    # Actual classes
    for name in [c for c in labels if c != "No label"]:
        create_row(name, key=f"use_{name}")

    if st.button(
        key="clear_labels_btn", use_container_width=True, label="Clear mask labels"
    ):
        rec["labels"] = {int(i): None for i in np.unique(rec["masks"]) if i != 0}
        st.rerun()


def class_manage_fragment(key_ns="side"):
    ss = st.session_state
    labels = ss.setdefault("all_classes", ["No label"])

    # Apply any pending rename select value before creating widgets
    pending_key = f"{key_ns}_rename_from__next_value"
    pending_val = ss.pop(pending_key, None)
    if pending_val is not None:
        ss[f"{key_ns}_rename_from"] = pending_val

    st.markdown("### Add and remove classes")

    # --- Add new class ---
    st.text_input(
        "",
        key="side_new_label",
        placeholder="Enter a new class here",
        on_change=add_label_from_input(labels, ss.get("side_new_label", "")),
    )

    # --- Delete existing class (now as dropdown) ---
    editable = [c for c in labels if c != "No label"]
    if editable:
        del_class = st.selectbox(
            "Select class to delete",
            options=["(Select a class)"] + editable,
            key=f"{key_ns}_delete_label",
        )
        if del_class != "(Select a class)":
            remove_class_everywhere(del_class)
            st.success(f"Deleted class: {del_class}")
            st.rerun()
    else:
        st.caption("No classes yet. Add a class above first.")
        return

    # --- Rename class ---
    c1, c2 = st.columns([1, 2])
    with c1:
        st.selectbox("Class to relabel", options=editable, key=f"{key_ns}_rename_from")
    with c2:
        st.text_input(
            "New label",
            key=f"{key_ns}_rename_to",
            placeholder="Type the new class name and press Enter",
            on_change=lambda: rename_class_from_input(
                f"{key_ns}_rename_from", f"{key_ns}_rename_to"
            ),
        )
