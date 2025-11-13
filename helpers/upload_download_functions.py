# helpers/image_io.py
from PIL import Image
import io
import numpy as np
import tifffile as tiff
import streamlit as st
from zipfile import ZipFile
from pathlib import Path
from zipfile import ZIP_DEFLATED
import streamlit as st
from PIL import ImageDraw
import pandas as pd
from PIL import UnidentifiedImageError
from helpers.classifying_functions import classes_map_from_labels, create_colour_palette
from helpers.mask_editing_functions import create_image_mask_overlay
from helpers.densenet_functions import resize_with_aspect_ratio


from helpers.state_ops import (
    ordered_keys,
    stem,
    set_current_by_index,
)

from helpers.cellpose_functions import normalize_image

ss = st.session_state

# --------------------------------------
# ---------- UPLOAD FUNCTIONS ----------
# --------------------------------------


def process_uploads(files, mask_suffix):
    if not files:
        return []
    skipped = []

    mask_suffix_len = len(mask_suffix)
    imgs = [(f) for f in files if not stem(f.name).endswith(mask_suffix)]
    for f in imgs:
        try:
            create_new_record_with_image(f)
        except (UnidentifiedImageError, Exception):
            skipped.append(f.name)

    ok = ordered_keys()
    if ok:
        set_current_by_index(len(ok) - 1)

    masks = [f for f in files if stem(f.name).endswith(mask_suffix)]
    if masks and ss.images:
        stem_to_key = {stem(rec["name"]): k for k, rec in ss.images.items()}
        for f in masks:
            base = stem(f.name)[:-mask_suffix_len]
            k = stem_to_key.get(base)
            if k is None:
                skipped.append(f.name)
                continue
            rec = ss.images[k]
            rec["labels"] = {}
            try:
                if f.name.endswith(".npy"):
                    rec["masks"] = load_npy_mask(f, rec)
                else:
                    rec["masks"] = load_tif_mask(f, rec)
                rec["labels"] = {
                    int(i): None for i in np.unique(rec["masks"]) if i != 0
                }
            except Exception:
                skipped.append(f.name)
                continue

    return skipped


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
    if name in m:
        st.session_state.current_key = m[name]
        return

    try:
        img_np = np.array(Image.open(uploaded_file).convert("RGB"), dtype=np.uint8)
        img_np = resize_with_aspect_ratio(
            img_np, patch_size=512
        )  # images are always resized to 512x512
    except (UnidentifiedImageError, Exception):
        raise

    H, W = img_np.shape[:2]
    k = st.session_state.next_ord
    st.session_state.next_ord += 1
    imgs[k] = {
        "name": name,
        "id": k,
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
        return isinstance(m, np.ndarray) and m.any()

    rows = []
    for k in ok:
        rec, m = ss.images[k], ss.images[k].get("masks")
        has = is_mask(m)
        n = int(len(np.unique(m)) - 1) if has else 0
        nl = sum(v is not None for v in rec.get("labels", {}).values())
        rows.append(
            {
                "Image": rec.get("name", k),
                "Mask Present": "✅" if has else "❌",
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
        if st.form_submit_button("Remove selected images", use_container_width=True):
            for k in edited.loc[edited["Remove"]].index:
                ss.images.pop(k, None)
            ks = sorted(ss.images)
            ss.current_key = ks[0] if ks else None
            st.rerun()


# --------------------------------------
# --------- DOWNLOAD FUNCTIONS ---------
# --------------------------------------


def counts_for_rec(rec) -> dict:
    """Return dict class_name -> count for one record, using rec['labels'] mapping."""
    labels = rec.get("labels", {}) or {}
    # Values in labels can be strings or ints or None; normalize to str
    by_class = {}
    for iid, cname in labels.items():
        cname = "No label" if cname in (None, "", -1) else str(cname)
        by_class[cname] = by_class.get(cname, 0) + 1
    return by_class


def build_masks_images_zip(
    state_images, key_order, include_overlay: bool, include_counts: bool
) -> bytes:
    buf = io.BytesIO()
    with ZipFile(buf, mode="w", compression=ZIP_DEFLATED) as zf:
        # Class color palette (only if overlays requested)
        palette = (
            create_colour_palette(
                st.session_state.setdefault("all_classes", ["No label"])
            )
            if include_overlay
            else None
        )

        # --- prep columns & rows for summary.csv
        class_cols = [c for c in st.session_state["all_classes"]]
        summary_rows = []

        # iterate through records
        for k in key_order:
            rec = state_images[k]
            name = Path(rec.get("name", f"{k}")).stem
            counts = counts_for_rec(rec)

            # write mask
            mask = rec.get("masks")
            tbuf = io.BytesIO()
            tiff.imwrite(tbuf, mask.astype(np.uint16))
            mask_suffix = ss["mask_suffix"]
            zf.writestr(f"masks/{name}{mask_suffix}.tif", tbuf.getvalue())

            # write iamge
            img = np.asarray(rec["image"])

            # optionally normalize image
            if st.session_state["dl_normalize_download"]:
                img = normalize_image(img)

            # optional overlay (colored masks) for image
            if include_overlay:
                classes_map = classes_map_from_labels(
                    rec.get("masks"), rec.get("labels", {})
                )
                img = create_image_mask_overlay(
                    img, rec.get("masks"), classes_map, palette, alpha=0.35
                )

            # optionally annotate image with class counts
            if include_counts:
                lines = [f"{cls}: {cnt}" for cls, cnt in sorted(counts.items())]
                if lines:
                    txt = "\n".join(lines)
                    pil = Image.fromarray(img)
                    d = ImageDraw.Draw(pil)
                    # measure text
                    w, h = d.multiline_textbbox((0, 0), txt)[2:]
                    img_w, img_h = pil.size
                    # create new image with extra space for text
                    new_img = Image.new("RGB", (img_w + w + 10, img_h), color="white")
                    new_img.paste(pil, (0, 0))
                    # draw text on right side
                    d = ImageDraw.Draw(new_img)
                    d.multiline_text((img_w + 5, 5), txt, fill=(0, 0, 0))
                    img = np.array(new_img)

            # capture a row for the CSV
            row = {"image": name}
            row.update({c: int(counts.get(c, 0)) for c in class_cols})
            row["total"] = int(sum(counts.values()))
            summary_rows.append(row)

            # write processed image to zip file
            ibuf = io.BytesIO()
            tiff.imwrite(ibuf, img, photometric="rgb", compression="deflate")
            zf.writestr(f"images/{name}.tif", ibuf.getvalue())

        # --- write summary.csv into the zip (image + per-class + total)
        df = pd.DataFrame(summary_rows, columns=["image"] + class_cols + ["total"])
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        zf.writestr("summary.csv", csv_buf.getvalue())

    return buf.getvalue()
