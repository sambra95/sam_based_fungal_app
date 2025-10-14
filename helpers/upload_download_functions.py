# helpers/image_io.py
from PIL import Image
import io
import numpy as np
import tifffile as tiff
import streamlit as st
from zipfile import ZipFile
from pathlib import Path
import streamlit as st

from helpers.state_ops import (
    ordered_keys,
    stem,
    set_current_by_index,
)

ss = st.session_state

# --------------------------------------
# ---------- UPLOAD FUNCTIONS ----------
# --------------------------------------


def _process_uploads(files, mask_suffix):
    if not files:
        return
    # load the images first
    mask_suffix_len = len(mask_suffix)
    imgs = [f for f in files if not stem(f.name).endswith(mask_suffix)]
    for f in imgs:
        create_new_record_with_image(f)
    ok = ordered_keys()
    if ok:
        set_current_by_index(len(ok) - 1)

    # then loads the masks (require prior image; match by stem without '_mask')
    masks = [f for f in files if stem(f.name).endswith(mask_suffix)]
    if masks and ss.images:
        stem_to_key = {stem(rec["name"]): k for k, rec in ss.images.items()}
        for f in masks:

            base = stem(f.name)[:-mask_suffix_len]
            k = stem_to_key.get(base)  # get the ID key
            if k is None:  # skips if no mask
                continue
            rec = ss.images[k]  # set the record
            rec["labels"] = {}  # reset the mask labels
            if f.name.endswith(".npy"):
                rec["masks"] = load_npy_mask(f, rec)
                rec["labels"] = {
                    int(i): None for i in np.unique(rec["masks"]) if i != 0
                }
            else:
                rec["masks"] = load_tif_mask(f, rec)
                rec["labels"] = {
                    int(i): None for i in np.unique(rec["masks"]) if i != 0
                }


def load_npy_mask(file, rec):
    """Read Cellpose *_seg.npy and return a (H,W) label matrix with 0 background, 1..N instances."""
    file = file.read()
    arr = np.load(io.BytesIO(file), allow_pickle=True).item()
    # Cellpose stores masks in dict under 'masks'
    mask = arr["masks"].astype(np.uint16)
    H, W = rec["H"], rec["W"]
    if mask.shape != (H, W):
        # resize if needed
        from PIL import Image

        mask = np.array(
            Image.fromarray(mask).resize((W, H), resample=Image.NEAREST),
            dtype=np.uint16,
        )
    return mask


def load_tif_mask(file, rec):
    """Read a label TIFF and return a (H,W) label matrix with 0 background, 1..N instances."""
    file = file.read()
    mask = tiff.imread(io.BytesIO(file)).astype(np.uint16)

    H, W = rec["H"], rec["W"]
    if mask.shape != (H, W):
        mask = np.array(
            Image.fromarray(mask).resize((W, H), resample=Image.NEAREST),
            dtype=np.uint16,
        )
    return mask


def create_new_record_with_image(uploaded_file):
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
        "masks": np.zeros((H, W), dtype=np.uint16),
        "labels": {},
        "boxes": [],
        "last_click_xy": None,
        "canvas": {"closed_json": None, "processed_count": 0},
    }
    m[name] = k
    st.session_state.current_key = k


def render_images_form():
    """display the uploaded images table"""
    ss, ok = st.session_state, sorted(st.session_state.images)

    def is_mask(m):
        return isinstance(m, np.ndarray) and m.ndim == 2 and m.any()

    rows = []
    for k in ok:
        rec, m = ss.images[k], ss.images[k].get("masks")
        has = is_mask(m)
        n = int(len(np.unique(m)) - 1) if has else 0
        nl = sum(v is not None for v in rec.get("labels", {}).values())
        rows.append(
            {
                "Image": rec.get("name", k),
                "Masks?": "✅" if has else "❌",
                "Number of Masks": n,
                "Labelled Masks": f"{nl}/{n}",
                "Remove": False,
            }
        )

    with st.form("images_form"):
        edited = st.data_editor(
            pd.DataFrame(rows, index=ok),
            hide_index=True,
            height=580,
            use_container_width=True,
            column_config={"Remove": st.column_config.CheckboxColumn()},
            disabled=["Image", "Masks Present", "Number of Masks", "Number of Labels"],
        )
        if st.form_submit_button("Apply", use_container_width=True):
            for k in edited.loc[edited["Remove"]].index:
                ss.images.pop(k, None)
            ks = sorted(ss.images)
            ss.current_key = ks[0] if ks else None
            st.rerun()


# --------------------------------------
# --------- DOWNLOAD FUNCTIONS ---------
# --------------------------------------


import io
import csv
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
import tifffile as tiff
import pandas as pd

from helpers.state_ops import ordered_keys
from helpers.classifying_functions import classes_map_from_labels, palette_from_emojis
from helpers.mask_editing_functions import composite_over_by_class
from helpers.densenet_functions import load_labeled_patches_from_session
from helpers.cell_metrics_functions import (
    _build_analysis_df,
    build_image_summary_df,
)


def _build_cell_metrics_zip(labels_selected):
    df = _build_analysis_df()
    if labels_selected:
        df = df[df["mask label"].isin(labels_selected)]
    items = []
    if not df.empty:
        items.append(("cell_analysis.csv", df.to_csv(index=False).encode("utf-8")))
    counts_df = build_image_summary_df()
    if not counts_df.empty:
        items.append(
            ("image_counts.csv", counts_df.to_csv(index=False).encode("utf-8"))
        )
    items += st.session_state.get("analysis_plots", [])
    return build_plots_zip(items) if items else b""


def _array_to_png_bytes(arr: np.ndarray) -> bytes:
    """Convert float/uint arrays to PNG bytes (3-channel)."""
    a = arr
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    elif a.ndim == 3 and a.shape[2] > 3:
        a = a[:, :, :3]

    if a.dtype.kind == "f":  # float -> assume 0..1 or 0..255
        a = np.clip(a, 0, 255)
        if a.max() <= 1.0:
            a = (a * 255.0).round()
    a = np.clip(a, 0, 255).astype(np.uint8)

    bio = io.BytesIO()
    Image.fromarray(a).save(bio, format="PNG")
    return bio.getvalue()


def build_densenet_artifacts_zip() -> bytes | None:
    """
    Build a ZIP containing DenseNet artifacts found in session_state:
    - densenet_model_bytes  (or densenet_model_path)
    - densenet_plot_losses_png
    - densenet_plot_confusion_png
    Returns ZIP bytes, or None if nothing to export.
    """
    # --- model bytes/name (optional) ---
    model_bytes = None
    model_name = None
    mbytes = st.session_state.get("densenet_model_bytes")
    mpath = st.session_state.get("densenet_model_path")
    if mbytes:
        model_bytes = mbytes
        model_name = Path(
            st.session_state.get("densenet_model_name", "densenet_model.bin")
        ).name
    elif mpath and Path(mpath).exists():
        model_bytes = Path(mpath).read_bytes()
        model_name = Path(mpath).name

    # --- plots (PNG only) ---
    def _get_png(key: str):
        b = st.session_state.get(key)
        return bytes(b) if isinstance(b, (bytes, bytearray)) and b else None

    losses_png = _get_png("densenet_plot_losses_png")
    cm_png = _get_png("densenet_plot_confusion_png")

    # Nothing to export?
    if model_bytes is None and losses_png is None and cm_png is None:
        return None

    # --- build the ZIP ---
    buf = io.BytesIO()
    with ZipFile(buf, mode="w", compression=ZIP_DEFLATED) as zf:
        if model_bytes is not None:
            zf.writestr(f"model/{model_name}", model_bytes)
        if losses_png is not None:
            zf.writestr("training_plots/losses.png", losses_png)
        if cm_png is not None:
            zf.writestr("evaluation/confusion_matrix.png", cm_png)

    return buf.getvalue()


def _counts_for_rec(rec) -> dict:
    """Return dict class_name -> count for one record, using rec['labels'] mapping."""
    labels = rec.get("labels", {}) or {}
    # Values in labels can be strings or ints or None; normalize to str
    by_class = {}
    for iid, cname in labels.items():
        cname = "unlabeled" if cname in (None, "", -1) else str(cname)
        by_class[cname] = by_class.get(cname, 0) + 1
    return by_class


def build_masks_images_zip(
    state_images, key_order, include_overlay: bool, include_counts: bool
) -> bytes:
    buf = io.BytesIO()
    with ZipFile(buf, mode="w", compression=ZIP_DEFLATED) as zf:
        # Class color palette (only if overlays requested)
        palette = (
            palette_from_emojis(
                st.session_state.setdefault("all_classes", ["Remove label"])
            )
            if include_overlay
            else None
        )

        for k in key_order:
            rec = state_images[k]
            name = Path(rec.get("name", f"{k}")).stem

            # Write mask as 16-bit TIFF (instance labels)
            inst = rec.get("masks")
            if isinstance(inst, np.ndarray) and inst.ndim == 2 and inst.size:
                tbuf = io.BytesIO()
                tiff.imwrite(tbuf, inst.astype(np.uint16))
                zf.writestr(f"masks/{name}.tif", tbuf.getvalue())

            img = np.asarray(rec["image"], dtype=np.uint8)
            ibuf = io.BytesIO()
            tiff.imwrite(ibuf, img, photometric="rgb")
            zf.writestr(f"images/{name}.tif", ibuf.getvalue())

            # Optional overlay (colored masks) and counts text
            if include_overlay:
                classes_map = classes_map_from_labels(
                    rec.get("masks"), rec.get("labels", {})
                )
                overlay = composite_over_by_class(
                    rec["image"], rec.get("masks"), classes_map, palette, alpha=0.35
                )
                if include_counts:
                    draw = ImageDraw.Draw(Image.fromarray(overlay))
                    counts = _counts_for_rec(rec)
                    lines = [f"{cls}: {cnt}" for cls, cnt in sorted(counts.items())]
                    if lines:
                        # simple text box
                        txt = "\n".join(lines)
                        pil = Image.fromarray(overlay)
                        d = ImageDraw.Draw(pil)
                        # background rectangle
                        w, h = d.multiline_textbbox((0, 0), txt)[2:]
                        pad = 6
                        d.rectangle(
                            [0, 0, w + 2 * pad, h + 2 * pad], fill=(0, 0, 0, 128)
                        )
                        d.multiline_text((pad, pad), txt, fill=(255, 255, 255))
                        overlay = np.array(pil)

                overlay = np.asarray(overlay, dtype=np.uint8)
                obuf = io.BytesIO()
                tiff.imwrite(obuf, overlay, photometric="rgb", compression="deflate")
                zf.writestr(f"overlays/{name}.tif", obuf.getvalue())

    return buf.getvalue()


def build_counts_csv(state_images, key_order) -> str:
    # Determine union of class names
    all_classes = set()
    per_image = []
    for k in key_order:
        rec = state_images[k]
        counts = _counts_for_rec(rec)
        all_classes |= set(counts.keys())
        per_image.append((rec.get("name", k), counts))
    cols = ["image", "total"] + sorted(all_classes)
    # Build CSV
    sio = io.StringIO()
    writer = csv.writer(sio)
    writer.writerow(cols)
    for name, counts in per_image:
        total = sum(counts.values()) if counts else 0
        row = [name, total] + [counts.get(cls, 0) for cls in sorted(all_classes)]
        writer.writerow(row)
    return sio.getvalue()


def build_patchset_zip_from_session(patch_size: int = 64) -> bytes | None:
    X, y, classes = load_labeled_patches_from_session(patch_size=patch_size)
    if X.shape[0] == 0:
        return None

    buf = io.BytesIO()
    rows = []
    with ZipFile(buf, mode="w", compression=ZIP_DEFLATED) as zf:
        for i in range(X.shape[0]):
            fname = f"patch_{i:06d}.png"
            label_idx = int(y[i])
            label_name = (
                classes[label_idx]
                if 0 <= label_idx < len(classes)
                else f"class{label_idx}"
            )
            zf.writestr(f"patches/{fname}", _array_to_png_bytes(X[i]))
            rows.append(
                {"filename": fname, "label_idx": label_idx, "label": label_name}
            )

        df = pd.DataFrame(rows, columns=["filename", "label_idx", "label"])
        zf.writestr("labels.csv", df.to_csv(index=False).encode("utf-8"))

    return buf.getvalue()


def build_model_artifacts_zip(prefix: str) -> bytes | None:
    """
    Gather model + training artifacts from session_state:
    - f"{prefix}_model_path" or f"{prefix}_model_bytes"
    - PNG plots: `{prefix}_plot_losses_png` / fallback `cp_losses_png`,
                `{prefix}_plot_iou_png`    / fallback `cp_compare_iou_png`
    - hparams CSV (optional): f"{prefix}_hparams_csv_path" or f"{prefix}_hparams_csv_bytes"
    - grid search results (optional): `{prefix}_grid_results_df` or `cp_grid_results_df` -> CSV
    """
    model_bytes = None
    model_name = None

    # model
    mpath = st.session_state.get(f"{prefix}_model_path")
    mbytes = st.session_state.get(f"{prefix}_model_bytes")
    if mbytes:
        model_bytes = mbytes
        model_name = Path(
            st.session_state.get(f"{prefix}_model_name", f"{prefix}_model.bin")
        ).name
    elif mpath and Path(mpath).exists():
        model_bytes = Path(mpath).read_bytes()
        model_name = Path(mpath).name

    # hyperparameters CSV (optional)
    hcsv = None
    hpath = st.session_state.get(f"{prefix}_hparams_csv_path")
    hbytes = st.session_state.get(f"{prefix}_hparams_csv_bytes")
    if hbytes:
        hcsv = ("hyperparameters.csv", hbytes)
    elif hpath and Path(hpath).exists():
        hcsv = (Path(hpath).name, Path(hpath).read_bytes())

    # plots (PNG only)
    def _get_png(key):
        data = st.session_state.get(key)
        return bytes(data) if isinstance(data, (bytes, bytearray)) and data else None

    loss_png = _get_png(f"{prefix}_plot_losses_png") or _get_png("cp_losses_png")
    iou_png = _get_png(f"{prefix}_plot_iou_png") or _get_png("cp_compare_iou_png")

    # grid search results DF -> CSV (optional)
    grid_df = st.session_state.get(f"{prefix}_grid_results_df") or st.session_state.get(
        "cp_grid_results_df"
    )
    grid_csv_bytes = (
        grid_df.to_csv(index=False).encode("utf-8")
        if isinstance(grid_df, pd.DataFrame) and not grid_df.empty
        else None
    )

    # nothing to export?
    if (
        model_bytes is None
        and hcsv is None
        and loss_png is None
        and iou_png is None
        and grid_csv_bytes is None
    ):
        return None

    # zip it
    buf = io.BytesIO()
    with ZipFile(buf, mode="w", compression=ZIP_DEFLATED) as zf:
        if model_bytes is not None:
            zf.writestr(f"model/{model_name}", model_bytes)
        if loss_png is not None:
            zf.writestr("training_plots/losses.png", loss_png)
        if iou_png is not None:
            zf.writestr("training_plots/iou_comparison.png", iou_png)
        if hcsv is not None:
            zf.writestr(f"training/{hcsv[0]}", hcsv[1])
        if grid_csv_bytes is not None:
            zf.writestr("tuning/gridsearch_results.csv", grid_csv_bytes)

    return buf.getvalue()


def build_plots_zip(plot_paths_or_bytes) -> bytes:
    """
    Accepts either list of file paths or list of (name, bytes).
    """
    if not plot_paths_or_bytes:
        return b""
    buf = io.BytesIO()
    with ZipFile(buf, mode="w", compression=ZIP_DEFLATED) as zf:
        for i, item in enumerate(plot_paths_or_bytes):
            if isinstance(item, (str, Path)) and Path(item).exists():
                p = Path(item)
                zf.writestr(p.name, p.read_bytes())
            elif isinstance(item, tuple) and len(item) == 2:
                zf.writestr(str(item[0]), item[1])
            else:
                # skip unknown
                pass
    return buf.getvalue()
