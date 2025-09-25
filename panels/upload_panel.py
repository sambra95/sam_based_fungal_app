import streamlit as st
from helpers.state_ops import ensure_image, ordered_keys, stem, set_current_by_index
from helpers.upload_download_functions import load_tif_masks_for_rec
from helpers.mask_editing_functions import append_masks_to_rec


def render_main():

    # ----- uploaded images -----#
    image_uploader_key = f"u_images_np_{st.session_state['image_uploader_nonce']}"
    st.subheader("Upload images here")
    imgs = st.file_uploader(
        "Images must be uploaded before masks",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        key=image_uploader_key,
    )
    if imgs:
        for up in imgs:
            ensure_image(up)
        ok = ordered_keys()
        curk = st.session_state.current_key
        set_current_by_index(curk)

    # ----- uploaded masks -----#
    st.subheader("Upload masks here")

    # unique key per run (optional, if you already maintain a nonce)
    mask_uploader_key = f"u_masks_np_{st.session_state['mask_uploader_nonce']}"

    up_masks_list = st.file_uploader(
        "Import masks",
        type=["tif", "tiff"],
        key=mask_uploader_key,
        disabled=not st.session_state.images,
        accept_multiple_files=True,
    )

    if up_masks_list and st.session_state.images:

        # map image stems
        stem_to_key = {
            stem(rec["name"]): k for k, rec in st.session_state.images.items()
        }

        for up_masks in up_masks_list:
            s = stem(up_masks.name)
            target_stem = s[:-5] if s.endswith("_mask") else s
            k = stem_to_key.get(target_stem)
            if k is not None:  # skips mask if no corresponding image file found
                rec = st.session_state.images[k]
                m = load_tif_masks_for_rec(up_masks, rec)
                append_masks_to_rec(rec, m)  # normalize/resize + append/replace
                st.session_state.current_key = k

        # rotate key once so selected files don't re-attach on rerun
        st.session_state["mask_uploader_nonce"] += 1
        st.rerun()

    # ---- Summary table: image–mask pairs ----

    st.subheader("Uploaded image–mask pairs")

    ok = ordered_keys()  # or list(st.session_state.images.keys()) if you don't sort
    if not ok:
        st.info("No images uploaded yet.")
    else:
        # header
        h1, h2, h3, h4, h5 = st.columns([4, 2, 2, 2, 2])
        h1.markdown("**Image**")
        h2.markdown("**Mask present**")
        h3.markdown("**Number of cells**")
        h4.markdown("**Labelled Masks**")
        h5.markdown("**Remove**")

        # rows
        for k in ok:

            rec = st.session_state.images[k]
            masks = rec.get("masks")
            number_labels = len([x for x in rec.get("labels") if x != None])
            has_mask = bool(masks is not None and getattr(masks, "size", 0) > 0)
            n_cells = int(masks.shape[0]) if has_mask else 0

            c1, c2, c3, c4, c5 = st.columns([4, 2, 2, 2, 2])
            c1.write(rec["name"])
            c2.write("✅" if has_mask else "❌")
            c3.write(n_cells)
            c4.write(f"{number_labels}/{n_cells}")
            if c5.button("Remove", key=f"remove_{k}"):
                # delete and fix current selection
                del st.session_state.images[k]
                ok2 = ordered_keys()
                st.session_state.current_key = ok2[0] if ok2 else None
                st.rerun()
