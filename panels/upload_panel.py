import streamlit as st
from helpers.state_ops import (
    ordered_keys,
)
from helpers.upload_download_functions import (
    _process_uploads,
)
import os, tempfile, hashlib
import numpy as np


def render_main():
    ss = st.session_state

    # ---------- Layout: 2 columns ----------
    col1, col2 = st.columns([2, 2])

    # ---------- LEFT: uploads & summary ----------
    with col1:
        # ---- single uploader: images & masks ----
        st.subheader("Upload images & masks here")

        # 👇 allow user to specify mask suffix (default "_masks")
        mask_suffix = st.text_input(
            "Mask file suffix (must match filenames before extension)",
            value=ss.get("mask_suffix", "_masks"),
            key="mask_suffix_input",
        )
        ss["mask_suffix"] = mask_suffix.strip() or "_masks"

        up_key = f"u_all_np_{ss.get('uploader_nonce', 0)}"
        files = st.file_uploader(
            f"Upload images (.png/.jpg/.tif) and masks with '{ss['mask_suffix']}' suffix (.tif/npy)",
            type=["tif", "tiff", "npy", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key=up_key,
        )

        if files:
            _process_uploads(files, rec, mask_suffix)
            ss["uploader_nonce"] = ss.get("uploader_nonce", 0) + 1
            st.rerun()

        st.divider()

        # ---- Cellpose model (custom weights) ----
        st.subheader("Upload Cellpose model here")
        cellpose_file = st.file_uploader(
            "Upload Cellpose model file",
            # Cellpose custom models are commonly .npy; allow pt/pth just in case
            type=["npy", "pt", "pth"],
            key="upload_cellpose_model",
        )
        if cellpose_file is not None:
            ss["cellpose_model_bytes"] = cellpose_file.read()
            ss["cellpose_model_name"] = cellpose_file.name
            st.success(f"Loaded Cellpose model: {cellpose_file.name}")

        st.divider()

        # ---- DenseNet-121 classifier ----
        st.subheader("Upload Densenet121 classifier here")
        densenet_file = st.file_uploader(
            "Upload DenseNet-121 checkpoint",
            type=[
                "keras",
            ],  # Keras formats only; torch files won't work with load_model
            key="upload_densenet_ckpt",
        )
        if densenet_file is not None:
            from tensorflow.keras.models import load_model, Model

            data = densenet_file.read()
            ext = os.path.splitext(densenet_file.name)[1].lower() or ".keras"
            h = hashlib.sha1(data).hexdigest()[:12]
            path = os.path.join(tempfile.gettempdir(), f"densenet_{h}{ext}")
            if not os.path.exists(path):
                with open(path, "wb") as f:
                    f.write(data)

            model = load_model(path, compile=False, safe_mode=False)
            if isinstance(model.outputs, (list, tuple)) and len(model.outputs) > 1:
                model = Model(model.inputs, model.outputs[0])  # single-output wrapper

            ss["densenet_model"] = model
            ss["densenet_model_path"] = path
            ss["densenet_ckpt_name"] = densenet_file.name
            st.success(f"Loaded DenseNet-121 classifier: {densenet_file.name}")

        # ---- Clear buttons ----
        col_a, col_b = st.columns(2)
        if col_a.button("Clear Cellpose model", use_container_width=True):
            ss["cellpose_model_bytes"] = None
            ss["cellpose_model_name"] = None
            ss["train_losses"] = []
            ss["test_losses"] = []
        if col_b.button("Clear DenseNet-121", use_container_width=True):
            ss["densenet_ckpt_bytes"] = None
            ss["densenet_ckpt_name"] = None

        st.divider()

    # ---------- RIGHT: model uploads (persist in session_state) ----------
    with col2:

        # ---- Status panel ----
        st.subheader("**Current model files**")
        st.write("Cellpose model:", ss.get("cellpose_model_name") or "—")
        st.write("DenseNet-121 model:", ss.get("densenet_ckpt_name") or "—")

        st.divider()

        # ---- Summary table: image–mask pairs ----

        num_images = len(st.session_state["images"].keys())

        st.subheader(f"Uploaded images and masks ({num_images})")
        ok = ordered_keys()
        if not ok:
            st.info("No images uploaded yet.")
        else:
            h1, h2, h3, h4, h5 = st.columns([4, 2, 2, 2, 2])
            h1.markdown("**Image**")
            h2.markdown("**Mask present**")
            h3.markdown("**Number of cells**")
            h4.markdown("**Labelled Masks**")
            h5.markdown("**Remove**")
            for k in ok:
                rec = ss.images[k]  # sets the row record
                masks = rec.get("masks")
                n_labels = len([v for v in rec["labels"].values() if v != None])
                # n_labels is just number of values of not None in dictionary
                has_mask = (
                    isinstance(masks, np.ndarray) and masks.ndim == 2 and masks.any()
                )  # check for a mask with the right format
                n_cells = (
                    int(len(np.unique(masks)) - 1) if has_mask else 0
                )  # n_cells = number of non 0 integers
                c1, c2, c3, c4, c5 = st.columns([4, 2, 2, 2, 2])
                # write out the table
                c1.write(rec["name"])
                c2.write("✅" if has_mask else "❌")
                c3.write(str(n_cells))
                c4.write(f"{n_labels}/{n_cells}")
                if c5.button("Remove", key=f"remove_{k}"):
                    del ss.images[k]
                    ok2 = ordered_keys()
                    ss.current_key = ok2[0] if ok2 else None
                    st.rerun()
