# panels/downloads.py
import io
import csv
import hashlib
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


def render_main():
    st.markdown("## Downloads")

    images = st.session_state.get("images", {})
    ok = ordered_keys() if images else []

    # ---------------------------
    # Helpers
    # ---------------------------
    def _arr_hash(a: np.ndarray) -> str:
        if a is None or not isinstance(a, np.ndarray):
            return "none"
        return hashlib.blake2b(a.view(np.uint8), digest_size=16).hexdigest()

    def _to_u8(img: np.ndarray) -> np.ndarray:
        """Robustly convert microscopy arrays (uint8/uint16/float) to uint8 RGB for PNG export."""
        a = img
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        if a.dtype == np.float32 or a.dtype == np.float64:
            a = np.clip(a, 0, 1)
            a = (a * 255).astype(np.uint8)
        elif a.dtype == np.uint16:
            a = (a / 257.0).astype(np.uint8)  # 16-bit -> 8-bit
        elif a.dtype != np.uint8:
            a = a.astype(np.uint8)
        return a

    def _counts_for_rec(rec) -> dict:
        """Return dict class_name -> count for one record, using rec['labels'] mapping."""
        labels = rec.get("labels", {}) or {}
        # Values in labels can be strings or ints or None; normalize to str
        by_class = {}
        for iid, cname in labels.items():
            cname = "unlabeled" if cname in (None, "", -1) else str(cname)
            by_class[cname] = by_class.get(cname, 0) + 1
        return by_class

    # ---------------------------
    # Cached builders
    # ---------------------------
    @st.cache_data(show_spinner=True)
    def build_masks_images_zip(
        state_images, key_order, include_overlay: bool, include_counts: bool
    ) -> bytes:
        buf = io.BytesIO()
        with ZipFile(buf, mode="w", compression=ZIP_DEFLATED) as zf:
            # Write a README
            zf.writestr(
                "README.txt",
                "Export produced by the Downloads panel.\n"
                "images/*.png are 8-bit previews; masks/*.tif are 16-bit instance labels.\n",
            )

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

                # Write image preview (PNG)
                img_u8 = _to_u8(rec["image"])
                ibuf = io.BytesIO()
                Image.fromarray(img_u8).save(ibuf, format="PNG")
                zf.writestr(f"images/{name}.png", ibuf.getvalue())

                # Optional overlay (colored masks) and counts text
                if include_overlay:
                    classes_map = classes_map_from_labels(
                        rec.get("masks"), rec.get("labels", {})
                    )
                    overlay = composite_over_by_class(
                        img_u8, rec.get("masks"), classes_map, palette, alpha=0.35
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

                    obuf = io.BytesIO()
                    Image.fromarray(overlay).save(obuf, format="PNG")
                    zf.writestr(f"overlays/{name}.png", obuf.getvalue())

        return buf.getvalue()

    @st.cache_data(show_spinner=True)
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

    @st.cache_data(show_spinner=True)
    def build_patchset_zip(patches) -> bytes:
        """
        Expect st.session_state['patches'] as a list of dicts:
        [{'image': np.ndarray, 'label': 'hypha'/'yeast'/..., 'name': 'patch_00001'}, ...]
        """
        buf = io.BytesIO()
        with ZipFile(buf, mode="w", compression=ZIP_DEFLATED) as zf:
            rows = []
            for i, p in enumerate(patches):
                name = p.get("name", f"patch_{i:05d}")
                img_u8 = _to_u8(p["image"])
                ibuf = io.BytesIO()
                Image.fromarray(img_u8).save(ibuf, format="PNG")
                zf.writestr(f"patches/{name}.png", ibuf.getvalue())
                rows.append([f"{name}.png", p.get("label", "unlabeled")])

            # labels.csv
            sio = io.StringIO()
            writer = csv.writer(sio)
            writer.writerow(["patch", "label"])
            writer.writerows(rows)
            zf.writestr("labels.csv", sio.getvalue())
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
            return (
                bytes(data) if isinstance(data, (bytes, bytearray)) and data else None
            )

        loss_png = _get_png(f"{prefix}_plot_losses_png") or _get_png("cp_losses_png")
        iou_png = _get_png(f"{prefix}_plot_iou_png") or _get_png("cp_compare_iou_png")

        # grid search results DF -> CSV (optional)
        grid_df = st.session_state.get(
            f"{prefix}_grid_results_df"
        ) or st.session_state.get("cp_grid_results_df")
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

    @st.cache_data(show_spinner=True)
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

    # ---------------------------
    # UI Sections
    # ---------------------------

    # Cellpose artifacts
    with st.expander("Cellpose: model + training artifacts", expanded=False):
        if st.button(
            "Build Cellpose ZIP", use_container_width=True, key="build_cellpose_zip"
        ):
            st.session_state["cellpose_zip_ready"] = build_model_artifacts_zip(
                "cellpose"
            )

        data = st.session_state.get("cellpose_zip_ready")
        st.download_button(
            "Download Cellpose artifacts (.zip)",
            data=data or b"",
            file_name="cellpose_artifacts.zip",
            mime="application/zip",
            disabled=not bool(data),
            use_container_width=True,
            key="dl_cellpose_zip",
            help=None if data else "No Cellpose artifacts found in session_state.",
        )

    # DenseNet artifacts

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

    with st.expander("DenseNet: model + training artifacts", expanded=False):
        if st.button(
            "Build DenseNet ZIP", use_container_width=True, key="build_densenet_zip"
        ):
            st.session_state["densenet_zip_ready"] = build_densenet_artifacts_zip()

        data = st.session_state.get("densenet_zip_ready")
        st.download_button(
            "Download DenseNet artifacts (.zip)",
            data=data or b"",
            file_name="densenet_artifacts.zip",
            mime="application/zip",
            disabled=not bool(data),
            use_container_width=True,
            key="dl_densenet_zip",
            help=None if data else "No DenseNet artifacts found in session_state.",
        )

    # Masks & images (with overlay options)
    with st.expander("Masks & images", expanded=True):
        c1, c2 = st.columns(2)
        include_overlay = c1.checkbox(
            "Include colored mask overlays", value=True, key="dl_include_overlay"
        )
        include_counts = c2.checkbox(
            "Overlay per-image class counts", value=False, key="dl_include_counts"
        )

        if not ok:
            st.info("No images in memory. Upload data first.")
        else:
            if st.button(
                "Build Masks & Images ZIP",
                use_container_width=True,
                key="build_masks_images_zip",
            ):
                st.session_state["masks_images_zip_ready"] = build_masks_images_zip(
                    images, ok, include_overlay, include_counts
                )

            data = st.session_state.get("masks_images_zip_ready")
            st.download_button(
                "Download masks & images (.zip)",
                data=data or b"",
                file_name="masks_and_images.zip",
                mime="application/zip",
                disabled=not bool(data),
                use_container_width=True,
                key="dl_masks_images_zip",
            )

    # Cell patch set with CSV labels
    with st.expander("Cell patch set + labels.csv", expanded=False):
        patches = st.session_state.get(
            "patches"
        )  # expected list of dicts with 'image' and 'label'
        if not patches:
            st.info("No patches found in session_state['patches'].")
            disabled = True
        else:
            disabled = False
            if st.button(
                "Build Patch Set ZIP", use_container_width=True, key="build_patch_zip"
            ):
                st.session_state["patch_zip_ready"] = build_patchset_zip(patches)

        data = st.session_state.get("patch_zip_ready")
        st.download_button(
            "Download patch set (.zip)",
            data=data or b"",
            file_name="cell_patches.zip",
            mime="application/zip",
            disabled=disabled or not bool(data),
            use_container_width=True,
            key="dl_patch_zip",
        )

    # Morphology plots (expects files or (name,bytes) in session_state['morphology_plots'])
    with st.expander("Morphology plots", expanded=False):
        plots = st.session_state.get("morphology_plots", [])
        if not plots:
            st.info("No plots found in session_state['morphology_plots'].")
            disabled = True
        else:
            disabled = False
            if st.button(
                "Build Morphology Plots ZIP",
                use_container_width=True,
                key="build_morph_zip",
            ):
                st.session_state["morph_zip_ready"] = build_plots_zip(plots)

        data = st.session_state.get("morph_zip_ready")
        st.download_button(
            "Download morphology plots (.zip)",
            data=data or b"",
            file_name="morphology_plots.zip",
            mime="application/zip",
            disabled=disabled or not bool(data),
            use_container_width=True,
            key="dl_morph_zip",
        )

    # Image counts (classified) as CSV
    with st.expander("Image counts (classified) CSV", expanded=False):
        if not ok:
            st.info("No images in memory. Upload data first.")
            csv_data = None
        else:
            if st.button(
                "Build Counts CSV", use_container_width=True, key="build_counts_csv"
            ):
                st.session_state["counts_csv_ready"] = build_counts_csv(images, ok)
            csv_data = st.session_state.get("counts_csv_ready")

        st.download_button(
            "Download counts (.csv)",
            data=csv_data or "",
            file_name="image_counts.csv",
            mime="text/csv",
            disabled=not bool(csv_data),
            use_container_width=True,
            key="dl_counts_csv",
        )
