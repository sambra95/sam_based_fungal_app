import streamlit as st
from helpers.state_ops import (
    ordered_keys,
)
from helpers.upload_download_functions import _process_uploads, render_images_form
import os, tempfile, hashlib
import numpy as np


def render_main():
    ss = st.session_state

    # ---------- Layout: 2 columns ----------
    col1, col2 = st.columns([2, 2])

    # ---------- LEFT: uploads & summary ----------
    with col1:
        # ---- single uploader: images & masks ----

        with st.container(border=True):
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
                _process_uploads(files, mask_suffix)
                ss["uploader_nonce"] = ss.get("uploader_nonce", 0) + 1
                st.rerun()

        with st.container(border=True):
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

        # ---- DenseNet-121 classifier ----
        with st.container(border=True):
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
                    model = Model(
                        model.inputs, model.outputs[0]
                    )  # single-output wrapper

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

    # ---------- RIGHT: model uploads (persist in session_state) ----------
    with col2:
        with st.container(border=True):
            # ---- Status panel ----
            st.subheader("**Current model files**")

            # Get theme primary color
            primary = st.get_option("theme.primaryColor")

            # Grab session state (short alias)
            ss = st.session_state

            # Prepare model info HTML
            cellpose_model = ss.get("cellpose_model_name") or "—"
            densenet_model = ss.get("densenet_ckpt_name") or "—"

            st.markdown(
                f"""
                <div style="
                    background: {primary}14;             /* ~8% opacity of theme color */
                    border: 1px solid {primary}59;       /* ~35% opacity */
                    border-radius: 10px;
                    padding: 14px 16px;
                    margin-top: 8px;
                    line-height: 1.6;
                ">
                    <p><b>Cellpose model:</b> {cellpose_model}</p>
                    <p><b>DenseNet-121 model:</b> {densenet_model}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.divider()

            # ---- Summary table: image–mask pairs ----
            num_images = len(st.session_state["images"].keys())
            st.subheader(f"Uploaded images and masks ({num_images})")

            ok = ordered_keys()
            if not ok:
                st.info("No images uploaded yet.")
            else:
                render_images_form()
