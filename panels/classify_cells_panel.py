# panels/classify_cells.py
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image


# panels/mask_editing.py
import numpy as np
import streamlit as st
from helpers.mask_editing_functions import (
    stack_to_instances_binary_first,
    get_class_palette,
    composite_over_by_class,
)


from helpers.state_ops import ordered_keys, set_current_by_index, current
from helpers import config as cfg  # CKPT_PATH, CFG_PATH
from contextlib import nullcontext

# panels/classify_cells.py
import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from helpers.state_ops import ordered_keys
from helpers.mask_editing_functions import stack_to_instances_binary_first

# --- builder: fills session_state["classifier_records_df"] and ["classifier_patches"] ---
import io
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from PIL import Image


def classes_map_from_labels(masks, labels):
    m = np.asarray(masks)
    if m.ndim == 2:
        m = m[None, ...]
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    m = (m > 0).astype(np.uint8)
    inst = stack_to_instances_binary_first(m)  # (H,W) uint16
    classes_map = {}
    ids = np.unique(inst)
    ids = ids[ids != 0]
    for iid in ids:
        mm = inst == iid
        # pick mask index with max overlap for this instance
        ov = [(i, int((m[i] & mm).sum())) for i in range(m.shape[0])]
        owner = max(ov, key=lambda t: t[1])[0]
        cls = labels[owner] if owner < len(labels) else None
        if cls is not None:
            classes_map[int(iid)] = cls
    return classes_map


def _stem(name: str) -> str:
    return Path(name).stem


def _to_square_patch(rgb: np.ndarray, patch_size: int = 256) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    s = patch_size / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    resized = np.array(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
    canvas = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    y0, x0 = (patch_size - nh) // 2, (patch_size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def make_classifier_zip(patch_size: int = 256) -> bytes | None:
    rows = []
    buf = io.BytesIO()
    with ZipFile(buf, "w") as zf:
        for k in ordered_keys():
            rec = st.session_state.images[k]
            img, m = rec.get("image"), rec.get("masks")
            if (
                img is None
                or not isinstance(m, np.ndarray)
                or m.ndim != 3
                or m.shape[0] == 0
            ):
                continue
            m = (m > 0).astype(np.uint8)  # (N,H,W)
            N, H, W = m.shape
            labs = list(rec.get("labels", []))
            if len(labs) < N:
                labs.extend([None] * (N - len(labs)))

            inst = stack_to_instances_binary_first(m)  # (H,W) uint16
            ids = np.unique(inst)
            ids = ids[ids != 0]
            base = _stem(rec["name"])

            for iid in ids:
                mm = inst == int(iid)
                if not mm.any():
                    continue
                ys, xs = np.where(mm)
                y0, y1 = ys.min(), ys.max() + 1
                x0, x1 = xs.min(), xs.max() + 1

                # owner mask index by max overlap
                owner = int(np.argmax(m[:, mm].sum(axis=1)))
                cls = labs[owner]
                if cls == "Remove label":
                    cls = None
                if cls is None:  # only include labeled instances
                    continue

                crop_rgb = img[y0:y1, x0:x1]
                crop_mask = mm[y0:y1, x0:x1][..., None]
                patch_rgb = _to_square_patch(
                    (crop_rgb * crop_mask).astype(np.uint8), patch_size
                )

                fname = f"{base}_mask{int(iid)}.png"
                bio = io.BytesIO()
                Image.fromarray(patch_rgb).save(bio, "PNG")
                zf.writestr(f"images/{fname}", bio.getvalue())
                rows.append({"image": fname, "mask number": int(iid), "class": cls})

        if not rows:
            return None
        df = pd.DataFrame(rows)
        zf.writestr("labels.csv", df.to_csv(index=False).encode("utf-8"))

    buf.seek(0)
    return buf.getvalue()


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
