import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from helpers.state_ops import ordered_keys
from skimage.measure import regionprops
from zipfile import ZipFile
from pathlib import Path
from zipfile import ZIP_DEFLATED
from helpers.classifying_functions import color_hex_for


def _hex_for_plot_label(label: str) -> str:
    """
    Map plotting labels to the same hex used for masks.
    'Unlabelled' and 'No label' both use the reserved 'No label' color.
    """
    if label in (None, "", "Unlabelled", "No label"):
        return color_hex_for("No label")
    return color_hex_for(label)


def plot_violin(df: pd.DataFrame, value_col: str):
    df["label"] = df["mask label"].replace("No label", None).fillna("Unlabelled")
    order = sorted(df["label"].unique(), key=lambda x: (x != "Unlabelled", str(x)))

    # use mask colors
    color_map = {lab: _hex_for_plot_label(lab) for lab in order}

    show_points = bool(st.session_state.get("overlay_datapoints", False))
    fig = go.Figure()

    for lab in order:
        idx = df["label"] == lab
        vals = df.loc[idx, value_col]
        x_vals = [lab] * len(vals)

        # violin body with the mask-matched color
        fig.add_trace(
            go.Violin(
                x=x_vals,
                y=vals,
                name=str(lab),
                legendgroup=str(lab),
                box_visible=True,
                meanline_visible=True,
                line_color="black",
                fillcolor=color_map[lab],
                opacity=0.85,
                points=False,
                hoverinfo="skip",
                showlegend=False,
            )
        )

        if show_points and len(vals) > 0:
            imgs = df.loc[idx, "image"].astype(str).to_numpy()
            masks = df.loc[idx, "mask #"].astype(str).to_numpy()
            texts = [f"{im}_patch{mk}" for im, mk in zip(imgs, masks)]

            fig.add_trace(
                go.Violin(
                    x=x_vals,
                    y=vals,
                    name=f"{lab}_pts",
                    legendgroup=str(lab),
                    box_visible=False,
                    meanline_visible=False,
                    line_color="rgba(0,0,0,0)",
                    fillcolor="rgba(0,0,0,0)",
                    opacity=1.0,
                    points="all",
                    pointpos=0,
                    jitter=0.25,
                    marker=dict(
                        size=4,
                        opacity=0.8,
                        color="black",
                        line=dict(width=0.3, color="black"),
                    ),
                    text=texts,  # per-point labels
                    hovertemplate="Image: %{text}<extra></extra>",
                    hoveron="points",
                    showlegend=False,
                )
            )

    fig.update_layout(
        violinmode="overlay",
        xaxis_title="Label",
        yaxis_title=value_col.replace("_", " ").title(),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
        height=500,
        showlegend=False,
    )
    fig.update_xaxes(showline=True, linecolor="black", gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showline=True, linecolor="black", gridcolor="rgba(0,0,0,0.1)")

    return f"{value_col.replace(' ', '_')}", fig


def plot_bar(df: pd.DataFrame, value_col: str):
    df["label"] = df["mask label"].replace("No label", None).fillna("Unlabelled")
    order = sorted(df["label"].unique(), key=lambda x: (x != "Unlabelled", str(x)))
    title_y = value_col.replace("_", " ").title()

    # numeric x positions and colors
    xpos = np.arange(len(order), dtype=float)
    colors = [_hex_for_plot_label(lab) for lab in order]

    # bar means + SD
    g = df.groupby("label")[value_col]
    means = g.mean().reindex(order).to_numpy()
    sds = g.std().reindex(order).fillna(0).to_numpy()

    bar = go.Bar(
        x=xpos,
        y=means,
        marker_color=colors,
        marker_line=dict(color="black", width=1),
        error_y=dict(type="data", array=sds, visible=True),
        opacity=0.9,
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>" + title_y + ": %{y:.2f}<extra></extra>",
    )

    traces = add_data_points_to_plot(bar, order, df, value_col, xpos)

    fig = go.Figure(traces, layout=dict(barcornerradius=10))
    fig.update_layout(
        xaxis=dict(
            tickvals=xpos,
            ticktext=order,
            showline=True,
            linecolor="black",
            gridcolor="rgba(0,0,0,0.1)",
        ),
        yaxis=dict(
            title=title_y, showline=True, linecolor="black", gridcolor="rgba(0,0,0,0.1)"
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
        bargap=0.3,
        height=500,
        showlegend=False,
    )

    return f"bar_{value_col.replace(' ', '_')}.png", fig


def add_data_points_to_plot(plot, order, sub, value_col, xpos):
    # jittered points per category (optional)
    traces = [plot]
    for i, lab in enumerate(order):
        idx = sub["label"] == lab
        ys = sub.loc[idx, value_col].to_numpy()
        if ys.size == 0:
            continue

        imgs = sub.loc[idx, "image"].astype(str).to_numpy()
        masks = sub.loc[idx, "mask #"].astype(str).to_numpy()
        texts = [f"{im}_patch{mk}" for im, mk in zip(imgs, masks)]

        rng = np.random.default_rng(42 + i)  # deterministic jitter per label
        xj = np.full(ys.shape, xpos[i], dtype=float) + rng.uniform(-0.20, 0.20, ys.size)

        traces.append(
            go.Scatter(
                x=xj,
                y=ys,
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=6,
                    opacity=0.75,
                    color="black",
                    line=dict(width=0.3, color="black"),
                ),
                text=texts,
                hovertemplate="Image: %{text}<extra></extra>",
            )
        )

    return traces


def build_analysis_df():
    rows = []
    for k in ordered_keys():
        rec = st.session_state.images[k]
        inst = rec.get("masks")
        if not isinstance(inst, np.ndarray) or inst.ndim != 2 or not inst.any():
            continue

        labdict = rec.get("labels", {})  # dict {instance_id -> class/None}
        for prop in regionprops(inst):  # prop.label is the instance id
            iid = int(prop.label)
            cls = labdict.get(iid)
            rows.append(
                {
                    "image": rec["name"],
                    "mask #": iid,
                    "mask label": ("Unlabelled" if cls in (None, "No label") else cls),
                    "mask area": float(prop.area),
                    "mask perimeter": float(
                        prop.perimeter
                    ),  # or perimeter_crofton if you prefer
                    # add any other metrics here, using `prop`
                }
            )
    return pd.DataFrame(rows)


def build_image_summary_df():
    rows = []
    all_classes = set()

    for k in ordered_keys():
        rec = st.session_state.images[k]
        inst = rec.get("masks")
        if not isinstance(inst, np.ndarray) or inst.ndim != 2:
            continue

        ids = np.unique(inst)
        ids = ids[ids != 0]
        total = len(ids)

        labdict = rec.get("labels", {})
        counts = {}
        unlabelled = 0

        for iid in ids:
            cls = labdict.get(int(iid))
            if cls is None or cls == "No label":
                unlabelled += 1
            else:
                counts[cls] = counts.get(cls, 0) + 1
                all_classes.add(cls)

        rows.append(
            {
                "image": rec["name"],
                "total cells": total,
                "unlabelled": unlabelled,
                **counts,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).fillna(0)
    # make sure integer counts
    for col in df.columns:
        if col != "image":
            df[col] = df[col].astype(int)

    # ensure all class columns appear
    for cls in sorted(all_classes):
        if cls not in df.columns:
            df[cls] = 0

    return df


# --- FUNCTIONS FOR DOWNLOADING CLASS CHARACTERISTICS PLOTS


def build_cell_metrics_zip(labels_selected):
    df = build_analysis_df()
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
