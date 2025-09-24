import streamlit as st
from helpers.state_ops import ensure_image, ordered_keys, set_current_by_index, stem
from helpers.image_io import load_masks_any
from helpers.masks import _attach_masks_to_image


def render():
    image_uploader_key = f"u_images_np_{st.session_state['image_uploader_nonce']}"
    # ----- uploaded images -----#
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
        names = [st.session_state.images[k]["name"] for k in ok]
        curk = st.session_state.current_key
        cur_idx = ok.index(curk) if curk in ok else 0
        sel = st.selectbox("Active image", names, index=cur_idx)
        set_current_by_index(names.index(sel))

    # ----- uploaded masks -----#
    st.subheader("Upload masks here")

    # unique key per run (optional, if you already maintain a nonce)
    uploader_key = f"u_masks_np_{st.session_state['mask_uploader_nonce']}"

    up_masks_list = st.file_uploader(
        "Import masks",
        type=["tif", "tiff", "npy", "npz"],
        key=uploader_key,
        disabled=not st.session_state.images,
        accept_multiple_files=True,  # 👈 allow multiple
    )

    if up_masks_list and st.session_state.images:
        # map image stems -> record keys once

        stem_to_key = {
            stem(rec["name"]): k for k, rec in st.session_state.images.items()
        }

        added, skipped = 0, []
        for up_masks in up_masks_list:
            s = stem(up_masks.name)
            target_stem = s[:-5] if s.endswith("_mask") else s
            k = stem_to_key.get(target_stem)
            if k is None:
                skipped.append(up_masks.name)
                continue

            m = load_masks_any(up_masks)  # (N,H,W) uint8 0/1
            rec = st.session_state.images[k]
            _attach_masks_to_image(rec, m)  # normalize/resize + append/replace
            st.session_state.current_key = k
            added += m.shape[0] if m.ndim == 3 else 1

        if added:
            st.success(f"Attached {added} mask(s).")
        if skipped:
            st.warning("No matching image for: " + ", ".join(skipped))

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
        h1, h2, h3, h4 = st.columns([4, 2, 2, 2])
        h1.markdown("**Image**")
        h2.markdown("**Mask present**")
        h3.markdown("**Number of cells**")
        h4.markdown("**Remove**")

        # rows
        for k in ok:
            rec = st.session_state.images[k]
            masks = rec.get("masks")
            has_mask = bool(masks is not None and getattr(masks, "size", 0) > 0)
            n_cells = int(masks.shape[0]) if has_mask else 0

            c1, c2, c3, c4 = st.columns([4, 2, 2, 2])
            c1.write(rec["name"])
            c2.write("✅" if has_mask else "❌")
            c3.write(n_cells)
            if c4.button("Remove", key=f"remove_{k}"):
                # delete and fix current selection
                del st.session_state.images[k]
                ok2 = ordered_keys()
                st.session_state.current_key = ok2[0] if ok2 else None
                st.rerun()
