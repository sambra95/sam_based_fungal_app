# panels/train_densenet.py
import numpy as np
import pandas as pd
import streamlit as st

# from PIL import Image  # (only if your helpers rely on PIL types somewhere)
import itertools
from cellpose import models, metrics, core
import torch
import io as IO

# ---- app helpers ----
from helpers.state_ops import ordered_keys
from helpers.densenet_functions import (
    load_labeled_patches_from_session,
    fine_tune_densenet,
    evaluate_fine_tinued_densenet,
    download_densenet_training_record,
)
from helpers.cellpose_functions import (
    finetune_cellpose_from_records,
    _plot_losses,
    compare_models_mean_iou_plot,
    download_cellpose_training_record,
)


ss = st.session_state


# ========== DenseNet: options (light) + dataset summary (light-ish) + training (heavy) ==========


def _densenet_options(key_ns="train_densenet"):
    """Light controls - lives outside fragments so changing options refreshes summary."""
    st.header("Fine-tune a DenseNet classifier")

    if not ordered_keys():
        st.info("Upload data and add labels in the other panels first.")
        return False

    # show information about the training set
    densenet_summary_fragment()

    c1, c2, c4 = st.columns(3)
    ss.setdefault("dn_input_size", 64)
    ss.setdefault("dn_batch_size", 32)
    ss.setdefault("dn_max_epoch", 100)
    ss.setdefault("dn_val_split", 0.2)

    ss["dn_input_size"] = c1.selectbox(
        "Input size",
        options=[64, 96, 128],
        index=[64, 96, 128].index(ss["dn_input_size"]),
    )
    ss["dn_batch_size"] = c2.selectbox(
        "Batch size",
        options=[8, 16, 32, 64],
        index=[8, 16, 32, 64].index(ss["dn_batch_size"]),
    )

    ss["dn_max_epoch"] = c4.number_input(
        "Max epochs",
        min_value=1,
        max_value=500,
        value=int(ss["dn_max_epoch"]),
        step=5,
        key="max_epoch_densenet_ui",
    )
    return True


@st.fragment
def densenet_summary_fragment():
    """Loads patches and shows a simple class frequency table (reruns when the page reruns)."""
    input_size = int(ss.get("dn_input_size", 64))

    # Load patches only for summary; heavy-ish but isolated here
    X, y, classes = load_labeled_patches_from_session(patch_size=input_size)

    # Count occurrences per class (ensure all classes present)
    counts = np.bincount(y, minlength=len(classes))
    freq_df = pd.DataFrame({"Class": list(classes), "Count": counts.astype(int)})

    st.info(
        f"Training set: {int(counts.sum())} labelled cells across {len(classes)} classes."
    )
    # --- Pretty, form-like card with rounded edges ---
    st.dataframe(
        freq_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Class": st.column_config.TextColumn("Class"),
            "Count": st.column_config.NumberColumn("Count", format="%d"),
        },
    )


@st.fragment
def densenet_train_fragment():
    """Runs the full DenseNet training pipeline when the button is clicked."""
    go = st.button("Fine tune Densenet121", use_container_width=True, type="primary")
    if not go:
        return

    # Read hyperparameter options from session
    input_size = int(ss.get("dn_input_size", 64))
    batch_size = int(ss.get("dn_batch_size", 32))
    epochs = int(ss.get("dn_max_epoch", 100))
    val_split = 0.2

    # fine tune the densenet model
    history, val_gen, classes = fine_tune_densenet(
        input_size=input_size, batch_size=batch_size, epochs=epochs, val_split=val_split
    )

    # evaluate the fine tuned densenet model on validation dataset
    evaluate_fine_tinued_densenet(history=history, val_gen=val_gen, classes=classes)


def render_densenet_train_panel(key_ns: str = "train_densenet"):
    if not _densenet_options(key_ns):
        return
    densenet_train_fragment()


def show_densenet_training_plots(height: int = 600):
    """Render saved DenseNet training plots from session state (if available)."""
    k1, k2 = "densenet_plot_losses_png", "densenet_plot_confusion_png"
    with st.container(border=True, height=height):
        if (k1 not in st.session_state) and (k2 not in st.session_state):
            st.empty()
            return
        st.header("DenseNet Training Summary")

        # button to download fine-tuned model, training data and training stats
        download_densenet_training_record()

        if k1 in st.session_state:
            st.image(
                st.session_state[k1],
                use_container_width=True,
            )
        else:
            st.empty()
        if k2 in st.session_state:
            st.image(
                st.session_state[k2],
                use_container_width=True,
            )
        else:
            st.empty()


# ========== Cellpose: options + training ==========


def _cellpose_options(key_ns="train_cellpose"):
    st.header("Fine-tune a Cellpose segmenter")

    if not ordered_keys():
        st.info("Upload data and label masks first.")
        return False

    # --- show dataset stats ---
    def is_mask(m):
        return isinstance(m, np.ndarray) and m.ndim == 2 and m.any()

    n_images, n_masks = len(ordered_keys()), 0
    for k in ordered_keys():
        rec = st.session_state["images"][k]
        m = rec["masks"]
        has = is_mask(m)
        n = int(len(np.unique(m)) - 1) if has else 0
        n_masks += n
    st.info(f"Training set: {n_masks} cell masks across {n_images} images.")

    # --- show training options ---
    c1, c2, c3 = st.columns(3)

    # Defaults
    ss.setdefault("cp_base_model", "cyto2")
    ss.setdefault("cp_max_epoch", 100)
    ss.setdefault("cp_lr", 0.1)
    ss.setdefault("cp_wd", 5e-4)
    ss.setdefault("cp_nimg", 32)

    ss["cp_base_model"] = c1.selectbox(
        "Base model",
        options=["cyto", "cyto2", "cyto3", "nuclei", "scratch"],
        index=["cyto2", "cyto", "cyto3", "nuclei", "scratch"].index(
            ss["cp_base_model"]
        ),  # this line sets the
    )
    ss["cp_max_epoch"] = c2.number_input(
        "Max epochs", 1, 1000, int(ss["cp_max_epoch"]), step=10
    )
    ss["cp_lr"] = c3.number_input(
        "Learning rate",
        min_value=1e-8,
        max_value=1.0,
        value=float(ss["cp_lr"]),
        format="%.5f",
    )
    ss["cp_wd"] = c1.number_input(
        "Weight decay",
        min_value=0.0,
        max_value=1.0,
        value=float(ss["cp_wd"]),
        step=1e-6,
        format="%.8f",  # more decimals prevents snapping to 0
        key="cp_wd_input",
    )

    ss["cp_nimg"] = c2.selectbox(
        "Batch size",
        options=[8, 16, 32, 64],
        index=[8, 16, 32, 64].index(ss["cp_nimg"]),
        key="cellpose_batch_size",
    )

    # --- Hyperparameter tuning toggle + grid inputs ---

    col1, col2 = st.columns([1, 1])
    with col1:
        ss.setdefault("cp_do_gridsearch", False)
        ss["cp_do_gridsearch"] = st.checkbox(
            "Optimise hyperparameters",
            value=bool(ss["cp_do_gridsearch"]),
        )

    with col2:
        with st.popover(
            "Hyperparameter search options",
            use_container_width=True,
            help="An exhaustive combinatorial gridsearch will be performed with these values if the 'Optimise hyperparameters' option is selected.",
        ):
            st.caption("Provide comma-separated lists. Leave blank to use defaults.")

            # Defaults for the grid
            ss.setdefault("cp_grid_cellprob", "0.2, 0.0, -0.2")
            ss.setdefault("cp_grid_flow", "0.3, 0.4, 0.6")
            ss.setdefault("cp_grid_niter", "1000")
            ss.setdefault("cp_grid_min_size", "0, 100")

            ss["cp_grid_cellprob"] = st.text_input(
                "cellprob_threshold values",
                ss["cp_grid_cellprob"],
                help="Examples: 0.2, 0.0, -0.2",
            )
            ss["cp_grid_flow"] = st.text_input(
                "flow_threshold values",
                ss["cp_grid_flow"],
                help="Examples: 0.3, 0.4, 0.6",
            )
            ss["cp_grid_niter"] = st.text_input(
                "niter values", ss["cp_grid_niter"], help="Examples: 500, 1000"
            )
            ss["cp_grid_min_size"] = st.text_input(
                "min_size values",
                ss["cp_grid_min_size"],
                help="Examples: 0, 100, 200",
            )

    return True


@st.fragment
def cellpose_train_fragment():
    go = st.button("Fine-tune Cellpose", use_container_width=True, type="primary")
    if not go:
        return

    recs = {k: st.session_state["images"][k] for k in ordered_keys()}
    base_model = ss.get("cp_base_model", "cyto2")
    epochs = int(ss.get("cp_max_epoch", 100))
    lr = float(ss.get("cp_lr", 5e-5))
    wd = float(ss.get("cp_wd", 0.1))
    nimg = int(ss.get("cp_nimg", 32))
    channels = ss.get("cellpose_channels", [0, 0])

    train_losses, test_losses, model_name = finetune_cellpose_from_records(
        recs,
        base_model=base_model,
        epochs=epochs,
        learning_rate=lr,
        weight_decay=wd,
        nimg_per_epoch=nimg,
        channels=channels,
    )

    st.success(f"Fine-tuning complete ✅ (model: {model_name})")

    st.session_state["train_losses"] = train_losses
    st.session_state["test_losses"] = test_losses

    _plot_losses(train_losses, test_losses)

    # ----- Prepare a (possibly subsampled) evaluation set -----
    masks = [rec["masks"] for rec in recs.values()]
    images = [rec["image"] for rec in recs.values()]
    N = len(images)
    sample_n = min(50, N)
    if N > sample_n:
        rng = np.random.default_rng()
        idx = rng.choice(N, size=sample_n, replace=False)
        images = [images[i] for i in idx]
        masks = [masks[i] for i in idx]

    # ----- OPTIONAL: Hyperparameter grid search -----
    if ss.get("cp_do_gridsearch", False):
        st.subheader("Hyperparameter tuning (grid search)")

        # Parse user grid text inputs into lists
        def _parse_float_list(s, default):
            s = (s or "").strip()
            if not s:
                return default
            try:
                return [float(x.strip()) for x in s.split(",")]
            except Exception:
                return default

        def _parse_int_list(s, default):
            s = (s or "").strip()
            if not s:
                return default
            try:
                return [int(float(x.strip())) for x in s.split(",")]
            except Exception:
                return default

        grid_cellprob = _parse_float_list(
            ss.get("cp_grid_cellprob", "0.2, 0.0, -0.2"), [0.2, 0.0, -0.2]
        )
        grid_flow = _parse_float_list(
            ss.get("cp_grid_flow", "0.3, 0.4, 0.6"), [0.3, 0.4, 0.6]
        )
        grid_niter = _parse_int_list(ss.get("cp_grid_niter", "1000"), [1000])
        grid_min_size = _parse_int_list(ss.get("cp_grid_min_size", "0, 100"), [0, 100])

        # Build combinations
        combos = list(
            itertools.product(grid_cellprob, grid_flow, grid_niter, grid_min_size)
        )
        total = len(combos)
        if total == 0:
            st.warning("No valid grid combinations. Skipping tuning.")
        else:
            # Get channels from session (fallback 0,0)
            ch1 = int(st.session_state.get("cp_ch1", 0))
            ch2 = int(st.session_state.get("cp_ch2", 0))
            channels = [ch1, ch2]

            # Load the in-memory fine-tuned model from session state
            # Expecting bytes saved somewhere like "cellpose_model_bytes"
            use_gpu = core.use_gpu()
            eval_model = models.CellposeModel(
                gpu=use_gpu,
                model_type=base_model if base_model != "scratch" else "cyto2",
            )
            ft_bytes = st.session_state.get("cellpose_model_bytes")
            if ft_bytes:
                try:
                    sd = torch.load(IO.BytesIO(ft_bytes), map_location="cpu")
                    eval_model.net.load_state_dict(sd)
                except Exception as e:
                    st.error(f"Failed to load fine-tuned weights from session: {e}")

            results = []
            pb = st.progress(0.0, text="Starting grid search…")
            for i, (cellprob, flowthresh, niter, min_size) in enumerate(combos, 1):
                pb.progress(
                    i / total,
                    text=f"Evaluating {i}/{total} (cp={cellprob}, flow={flowthresh}, niter={niter}, min={min_size})",
                )
                try:
                    masks_pred, flows, styles = eval_model.eval(
                        list(images),
                        channels=list(channels),
                        diameter=None,
                        cellprob_threshold=cellprob,
                        flow_threshold=flowthresh,
                        niter=niter,
                        min_size=min_size,
                    )
                    ap = metrics.average_precision(masks, masks_pred)[
                        0
                    ]  # AP per-image matrix
                    score = float(np.nanmean(ap[:, 0]))  # mean AP at IoU=0.5
                except Exception as e:
                    st.warning(
                        f"Evaluation error for cp={cellprob}, flow={flowthresh}, niter={niter}, min={min_size}: {e}"
                    )
                    score = float("nan")

                results.append(
                    {
                        "cellprob": cellprob,
                        "flow_threshold": flowthresh,
                        "niter": niter,
                        "min_size": min_size,
                        "ap_iou_0.5": score,
                    }
                )

            pb.empty()

            # Store + show results (no CSV write)
            df = pd.DataFrame(results).sort_values(
                by="ap_iou_0.5", ascending=False, na_position="last"
            )
            st.session_state["cp_grid_results_df"] = df
            st.dataframe(df, use_container_width=True)

            # Pick best and push into session state used by the mask editing panel
            if not df.empty and np.isfinite(df["ap_iou_0.5"].iloc[0]):
                best = df.iloc[0]
                ss["cp_cellprob_threshold"] = float(best["cellprob"])
                ss["cp_flow_threshold"] = float(best["flow_threshold"])
                ss["cp_min_size"] = int(best["min_size"])
                ss["cp_niter"] = int(
                    best["niter"]
                )  # not used in segment_rec_with_cellpose, but we keep it
                st.success(
                    f"Best hyperparameters set: cellprob={best['cellprob']}, "
                    f"flow={best['flow_threshold']}, min_size={int(best['min_size'])}, niter={int(best['niter'])}"
                )
            else:
                st.info("No valid result to set best hyperparameters.")

    # ----- Plot model comparison afterwards (unchanged) -----
    compare_models_mean_iou_plot(
        images,
        masks,
        base_model_name=base_model if base_model != "scratch" else "cyto2",
    )


def show_cellpose_training_plots(height: int = 600):
    """Render saved Cellpose plots from session state (if available)."""
    k1, k2 = "cp_losses_png", "cp_compare_iou_png"
    with st.container(border=True, height=height):
        if (k1 not in st.session_state) and (k2 not in st.session_state):
            st.empty()
            return
        st.header("Cellpose Training Summary")

        # button for downloading fine-tuned model, training data and training stats in a zip file
        download_cellpose_training_record()

        if k1 in st.session_state:
            st.image(
                st.session_state[k1],
                use_container_width=True,
            )
        else:
            st.info("No fine-tuning data to show.")
        if k2 in st.session_state:
            st.image(
                st.session_state[k2],
                use_container_width=True,
            )
        else:
            st.info("No fine-tuning data to show.")


def render_cellpose_train_panel(key_ns="train_cellpose"):
    if not _cellpose_options(key_ns):
        return
    cellpose_train_fragment()  # heavy; runs only on button click
