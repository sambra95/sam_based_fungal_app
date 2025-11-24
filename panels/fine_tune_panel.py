# panels/train_densenet.py
import numpy as np
import pandas as pd
import streamlit as st

import itertools
from cellpose import models, metrics, core
import torch
import io as IO

from helpers.state_ops import ordered_keys
from helpers.densenet_functions import (
    load_labeled_patches,
    finetune_densenet,
    evaluate_fine_tuned_densenet,
    plot_loss_curve,
    build_densenet_zip_bytes,
)
from helpers.cellpose_functions import (
    finetune_cellpose,
    compute_prediction_ious,
    plot_iou_comparison,
    plot_pred_vs_true_counts,
    get_cellpose_model,
    build_cellpose_zip_bytes,
)


ss = st.session_state


# ========== DenseNet: options (light) + dataset summary (light-ish) + training (heavy) ==========


@st.fragment
def render_densenet_options(key_ns="train_densenet"):
    """Light controls - lives outside fragments so changing options refreshes summary."""
    st.header("Fine-tune a DenseNet classifier")

    if not ordered_keys():
        st.info("Upload data and add labels in the other panels first.")
        return False

    # show information about the training set
    render_densenet_summary_fragment()

    c1, c2, c4 = st.columns(3)
    # options for densenet training with defaults already set
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
def render_densenet_summary_fragment():
    """Loads patches and shows a simple class frequency table (reruns when the page reruns)."""
    input_size = int(ss.get("dn_input_size"))

    # Load patches only for summary; heavy-ish but isolated here
    X, y, classes = load_labeled_patches(patch_size=input_size)

    # Count occurrences per class (ensure all classes present)
    counts = np.bincount(y, minlength=len(classes))
    freq_df = pd.DataFrame({"Class": list(classes), "Count": counts.astype(int)})

    st.info(
        f"Training set: {int(counts.sum())} labelled cells across {len(classes)} classes."
    )
    # Pretty, form-like card with rounded edges
    st.dataframe(
        freq_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Class": st.column_config.TextColumn("Class"),
            "Count": st.column_config.NumberColumn("Count", format="%d"),
        },
    )


def densenet_can_train(
    patch_size: int, min_classes: int = 2, min_instances: int = 2
) -> bool:
    """
    Return True if there are at least `min_classes` classes
    with >= `min_instances` examples each.
    """
    _, y, classes = load_labeled_patches(patch_size=patch_size)

    # counts[i] = number of samples for class i
    counts = np.bincount(y, minlength=len(classes))

    # how many classes have at least `min_instances` examples?
    n_ok = int((counts >= min_instances).sum())
    return n_ok >= min_classes


def render_densenet_train_fragment():
    """Runs the full DenseNet training pipeline when the button is clicked."""

    # Read hyperparameter options from session
    input_size = int(ss.get("dn_input_size"))
    batch_size = int(ss.get("dn_batch_size"))
    epochs = int(ss.get("dn_max_epoch"))
    val_split = 0.2

    # --- check if we have enough data to train ---
    can_train = densenet_can_train(patch_size=input_size)

    if not can_train:
        st.warning(
            "Need at least 2 classes with ≥ 2 labelled cells each "
            "before fine-tuning DenseNet."
        )

    # Disable button if we don't have enough data
    go = st.button(
        "Fine tune Densenet121",
        use_container_width=True,
        type="primary",
        disabled=not can_train,
    )

    # Don't proceed if button not clicked or training is not allowed
    if (not go) or (not can_train):
        return

    # fine tune the densenet model
    history, val_gen, classes = finetune_densenet(
        input_size=input_size, batch_size=batch_size, epochs=epochs, val_split=val_split
    )

    # evaluate the fine tuned densenet model on validation dataset
    evaluate_fine_tuned_densenet(history=history, val_gen=val_gen, classes=classes)

    ss["dn_zip_bytes"] = build_densenet_zip_bytes(input_size)


def show_densenet_training_plots():
    """Render saved DenseNet training plots from session state (if available)."""

    # check for uploaded data
    k1, k2 = "densenet_training_metrics", "densenet_training_metrics"
    if (k1 not in st.session_state) and (k2 not in st.session_state):
        st.info("No fine-tuning data available. Tune a model first.")
        st.empty()
        return

    else:

        st.header("DenseNet Training Summary")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                st.session_state["densenet_training_losses"],
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                st.session_state["densenet_training_metrics"],
                use_container_width=True,
            )

        st.plotly_chart(
            st.session_state["densenet_confusion_matrix"],
            use_container_width=True,
        )

        st.download_button(
            "Download fine-tuned DenseNet model, dataset and training metrics",
            data=ss["dn_zip_bytes"],
            file_name="densenet_training.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary",
        )


# ========== Cellpose: options + training ==========


def render_cellpose_options(key_ns="train_cellpose"):
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
    ss.setdefault("cp_learning_rate", 0.1)
    ss.setdefault("cp_weight_decay", 1e-4)
    ss.setdefault("cp_batch_size", 32)

    ss["cp_base_model"] = c1.selectbox(
        "Base model",
        options=["cyto", "cyto2", "cyto3", "nuclei", "scratch"],
        index=["cyto", "cyto2", "cyto3", "nuclei", "scratch"].index(
            ss["cp_base_model"]
        ),  # this line sets the
    )
    ss["cp_max_epoch"] = c2.number_input(
        "Max epochs", 1, 1000, int(ss["cp_max_epoch"]), step=10
    )
    ss["cp_learning_rate"] = c3.number_input(
        "Learning rate",
        min_value=1e-8,
        max_value=10.0,
        value=float(ss["cp_learning_rate"]),
        format="%.5f",
    )
    ss["cp_weight_decay"] = c1.number_input(
        "Weight decay",
        min_value=0.0,
        max_value=1.0,
        value=float(ss["cp_weight_decay"]),
        step=1e-8,
        format="%.8f",  # more decimals prevents snapping to 0
        key="cp_weight_decay_input",
    )

    ss["cp_batch_size"] = c2.selectbox(
        "Batch size",
        options=[8, 16, 32, 64],
        index=[8, 16, 32, 64].index(ss["cp_batch_size"]),
        key="cellpose_batch_size",
    )

    ss["cp_training_ch1"] = c3.number_input(
        "Channel 1",
        min_value=0,
        max_value=4,
        value=int(ss["cp_training_ch1"]),
        step=1,
        key="cp_training_ch1_input",
        help="Set to 0 if single-channel images.",
    )

    ss["cp_training_ch2"] = c1.number_input(
        "Channel 2",
        min_value=0,
        max_value=4,
        value=int(ss["cp_training_ch2"]),
        step=1,
        key="cp_training_ch2_input",
        help="Set to 0 if single-channel images.",
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


def render_cellpose_train_fragment():
    """Runs the full Cellpose fine-tuning pipeline when the button is clicked."""

    # start fine-tuning when button clicked
    go = st.button("Fine-tune Cellpose", use_container_width=True, type="primary")
    if not go:
        return

    # gather training data
    recs = {k: st.session_state["images"][k] for k in ordered_keys()}
    base_model = ss.get("cp_base_model")
    epochs = int(ss.get("cp_max_epoch"))
    lr = float(ss.get("cp_learning_rate"))
    wd = float(ss.get("cp_weight_decay"))
    nimg = int(ss.get("cp_batch_size"))
    channels = [ss.get("cp_training_ch1"), ss.get("cp_training_ch2")]

    # fine-tune the cellpose model
    with st.spinner("Fine-tuning Cellpose…"):
        train_losses, test_losses, model_name = finetune_cellpose(
            recs,
            base_model=base_model,
            epochs=epochs,
            learning_rate=lr,
            weight_decay=wd,
            nimg_per_epoch=nimg,
            channels=channels,
        )

        # store training losses in session state for plotting
        st.session_state["train_losses"] = train_losses
        st.session_state["test_losses"] = test_losses

        st.session_state["cellpose_training_losses"] = plot_loss_curve(
            train_losses, test_losses
        )

    # prepare a evaluation set
    masks = [rec["masks"] for rec in recs.values()]
    images = [rec["image"] for rec in recs.values()]

    # subsample to max 50 images for speed
    N = len(images)
    sample_n = min(50, N)
    if N > sample_n:
        rng = np.random.default_rng()
        idx = rng.choice(N, size=sample_n, replace=False)
        images = [images[i] for i in idx]
        masks = [masks[i] for i in idx]

    # OPTIONAL: hyperparameter grid search
    if ss.get("cp_do_gridsearch"):
        st.subheader("Hyperparameter tuning (grid search)")

        # parse user grid text inputs into lists
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

        # build grid lists
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
            # Get channels from session
            ch1 = int(st.session_state.get("cp_ch1"))
            ch2 = int(st.session_state.get("cp_ch2"))
            channels = [ch1, ch2]

            # Load the in-memory fine-tuned model from session state
            # Expecting bytes saved somewhere like "cellpose_model_bytes"
            use_gpu = core.use_gpu()
            eval_model = models.CellposeModel(
                gpu=use_gpu,
                model_type=base_model if base_model != "scratch" else "cyto2",
            )
            ft_bytes = st.session_state.get("cellpose_model_bytes")

            sd = torch.load(IO.BytesIO(ft_bytes), map_location="cpu")
            eval_model.net.load_state_dict(sd)

            results = []
            pb = st.progress(0.0, text="Starting grid search…")

            # iterate over all hyperparameter combinations and store performance metrics
            for i, (cellprob, flowthresh, niter, min_size) in enumerate(combos, 1):
                pb.progress(
                    i / total,
                    text=f"Evaluating {i}/{total} (cp={cellprob}, flow={flowthresh}, niter={niter}, min={min_size})",
                )
                masks_pred, flows, styles = eval_model.eval(
                    images,
                    channels=channels,
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

            # store grid results dataframe in session state for later display
            df = pd.DataFrame(results).sort_values(
                by="ap_iou_0.5", ascending=False, na_position="last"
            )
            st.session_state["cp_grid_results_df"] = df

            st.success(f"Fine-tuning complete ✅ (model: {model_name})")

            # Pick best hyperparameters and push into session state used by the mask editing panel
            if not df.empty and np.isfinite(df["ap_iou_0.5"].iloc[0]):
                best = df.iloc[0]
                ss["cp_cellprob_threshold"] = float(best["cellprob"])
                ss["cp_flow_threshold"] = float(best["flow_threshold"])
                ss["cp_min_size"] = int(best["min_size"])
                ss["cp_niter"] = int(best["niter"])
                st.success(
                    f"Best hyperparameters set: cellprob={best['cellprob']}, "
                    f"flow={best['flow_threshold']}, min_size={int(best['min_size'])}, niter={int(best['niter'])}"
                )

            else:
                st.info("No valid result to set best hyperparameters.")

    #  plot model comparison
    with st.spinner("Validating model…"):
        use_gpu = core.use_gpu()
        base_model = models.CellposeModel(gpu=use_gpu, model_type=ss["cp_base_model"])
        base_preds, _, _ = base_model.eval(images, channels=channels)

        # Fine-tuned model from session BYTES
        tuned_model = get_cellpose_model()
        tuned_preds, _, _ = tuned_model.eval(images, channels=channels)

        # compare and plot ious pre and post training
        base_ious = compute_prediction_ious(
            images=images, masks=masks, model=base_model, channels=channels
        )
        tuned_ious = compute_prediction_ious(
            images=images, masks=masks, model=tuned_model, channels=channels
        )
        ss["cellpose_iou_comparison"] = plot_iou_comparison(base_ious, tuned_ious)

        # compare and plot true vs predicted counts pre and post training
        gt_counts = [int(np.count_nonzero(np.unique(mask))) for mask in masks]
        base_counts = [int(np.count_nonzero(np.unique(pred))) for pred in base_preds]
        tuned_counts = [int(np.count_nonzero(np.unique(pred))) for pred in tuned_preds]

        ss["cellpose_original_counts_comparison"] = plot_pred_vs_true_counts(
            gt_counts, base_counts, title="Base Model Predictions"
        )
        ss["cellpose_tuned_counts_comparison"] = plot_pred_vs_true_counts(
            gt_counts, tuned_counts, title="Tuned Model Predictions"
        )

        ss["cp_zip_bytes"] = build_cellpose_zip_bytes()


def show_cellpose_training_plots():
    """Render saved Cellpose plots from session state (if available)."""

    # check for uploaded data
    if "cellpose_training_losses" not in st.session_state:
        st.header("Cellpose Training Summary")
        st.info("No fine-tuning data to show.")
        return

    else:

        st.header("Cellpose Training Summary")

        col1, col2 = st.columns(2)
        with col1:

            # plot training losses
            st.plotly_chart(
                st.session_state["cellpose_training_losses"],
                use_container_width=True,
            )

            # plot original vs predicted counts
            st.plotly_chart(
                st.session_state["cellpose_original_counts_comparison"],
                use_container_width=True,
            )

        with col2:

            # plot iou comparison
            st.plotly_chart(
                st.session_state["cellpose_iou_comparison"],
                use_container_width=True,
            )

            # plot tuned vs predicted counts
            st.plotly_chart(
                st.session_state["cellpose_tuned_counts_comparison"],
                use_container_width=True,
            )

        # display grid search results if applicable
        if ss.get("cp_do_gridsearch"):
            st.dataframe(
                st.session_state["cp_grid_results_df"],
                hide_index=True,
                use_container_width=True,
            )

        else:
            st.info("No hyperparameter tuning performed.")

        # button for downloading fine-tuned model, training data and training stats in a zip file
        st.download_button(
            "Download Cellpose model, dataset and training metrics (ZIP)",
            data=ss["cp_zip_bytes"],
            file_name="cellpose_training.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary",
        )
