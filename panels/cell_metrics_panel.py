# panels/cell_metrics_panel.py
import streamlit as st

from helpers.upload_download_functions import ordered_keys
from helpers.cell_metrics_functions import (
    _build_analysis_df,
    _violin,
    _bar,
)


def render_sidebar():

    if not ordered_keys():
        return False

    st.select_slider(
        "Plot type",
        options=["Violin", "Bar"],
        value=st.session_state.get("analysis_plot_type", "Violin"),
        key="analysis_plot_type",
    )

    df = _build_analysis_df()
    if df.empty:
        st.info("No masks found.")
        return

    # Labels multiselect (single instance)
    label_options = sorted(
        df["mask label"].unique(), key=lambda x: (x != "Unlabelled", str(x))
    )
    default_labels = st.session_state.get("analysis_labels", label_options)
    default_labels = [
        label for label in default_labels if label in label_options
    ] or label_options
    st.multiselect(
        "Include labels",
        options=label_options,
        default=default_labels,
        key="analysis_labels",
    )

    # Metrics multiselect (names must match df columns)
    metric_options = [
        col for col in df.columns if col not in ["image", "mask #", "mask label"]
    ]
    default_metrics = st.session_state.get("analysis_metrics", metric_options)
    default_metrics = [
        m for m in default_metrics if m in metric_options
    ] or metric_options
    st.multiselect(
        "Include metrics",
        options=metric_options,
        default=default_metrics,
        key="analysis_metrics",
    )


def render_main():

    if not ordered_keys():
        st.info("Upload data and label masks first.")
        return False

    df = _build_analysis_df()

    df_filt = df.copy()
    df_filt["mask label"] = (
        df_filt["mask label"].replace("Remove label", None).fillna("Unlabelled")
    )

    # ... after building df and df_filt ...
    labels_to_plot = st.session_state.get("analysis_labels", None)
    metrics = st.session_state.get("analysis_metrics") or [
        "mask area",
        "mask perimeter",
    ]

    if labels_to_plot:
        df_filt = df_filt[df_filt["mask label"].isin(labels_to_plot)]

    if df_filt.empty:
        st.info("No data for the selected labels.")
        return

    ptype = st.session_state.get("analysis_plot_type", "Violin")
    plots = []
    for col in metrics:
        fname, img = (_violin if ptype == "Violin" else _bar)(df_filt, col)
        plots.append((fname, img))
    st.session_state["analysis_plots"] = plots
