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


def hex_for_plot_label(label: str) -> str:
    """
    Map plotting labels to the same hex used for masks.
    'Unlabelled' and 'No label' both use the reserved 'No label' color.
    """
    if label in (None, "", "Unlabelled", "No label"):
        return color_hex_for("No label")
    return color_hex_for(label)


def plot_violin(df: pd.DataFrame, value_col: str):
    """
    Create a violin plot of `value_col` grouped by mask label.
    Shows data points if `overlay_datapoints` is set in session state."""
    df["label"] = df["mask label"].replace("No label", None).fillna("Unlabelled")
    order = sorted(df["label"].unique(), key=lambda x: (x != "Unlabelled", str(x)))

    # use mask colors
    color_map = {lab: hex_for_plot_label(lab) for lab in order}

    show_points = bool(st.session_state.get("overlay_datapoints", False))
    fig = go.Figure()

    # violin traces per label
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

        # optional data points overlaid
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

    # final layout
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
    """
    Create a bar plot of mean `value_col` grouped by mask label,
    with error bars showing standard deviation. Shows data points if `overlay_datapoints` is set
    in session state."""

    df["label"] = df["mask label"].replace("No label", None).fillna("Unlabelled")
    order = sorted(df["label"].unique(), key=lambda x: (x != "Unlabelled", str(x)))
    title_y = value_col.replace("_", " ").title()

    # numeric x positions and colors
    xpos = np.arange(len(order), dtype=float)
    colors = [hex_for_plot_label(lab) for lab in order]

    # bar means + SD
    g = df.groupby("label")[value_col]
    means = g.mean().reindex(order).to_numpy()
    sds = g.std().reindex(order).fillna(0).to_numpy()

    # main bar trace
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

    # add jittered data points if requested
    traces = add_data_points_to_plot(bar, order, df, value_col, xpos)

    # final layout
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
    """
    If `overlay_datapoints` is set in session state, add a scatter trace
    with jittered points for each category to the given plotly bar plot."""

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

        # scatter trace for data points
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


def safe_div(num, den):
    """Return num / den, but np.nan if den is 0 or not finite."""
    if den == 0 or not np.isfinite(den):
        return float("nan")
    return float(num / den)


def mask_shape_metrics(prop):
    """
    Compute a set of shape metrics for a single skimage regionprops object.

    Parameters
    ----------
    prop : skimage.measure._regionprops.RegionProperties
        Region properties for a single labeled instance.
    max_edge_to_edge : float or None
        Optional precomputed longest internal chord length.
        If provided, will be used for a normalized metric.

    Returns
    -------
    dict
        Keys are metric names (strings), values are floats (or np.nan).
    """

    area = float(prop.area)
    perimeter = float(prop.perimeter)

    # Basic axis lengths
    major = float(getattr(prop, "major_axis_length", 0.0))
    minor = float(getattr(prop, "minor_axis_length", 0.0))

    # Classic circularity / roundness measure:
    #   circularity = 4*pi*area / perimeter^2
    # = 1 for a perfect circle, < 1 for other shapes.
    circularity = safe_div(4.0 * np.pi * area, perimeter**2)

    # Alternative "roundness" using major axis:
    #   roundness = 4*area / (pi * major_axis_length^2)
    # again 1 for a perfect circle if major_axis_length is the diameter.
    if major > 0:
        roundness = safe_div(4.0 * area, np.pi * major**2)
    else:
        roundness = float("nan")

    # Aspect ratio (elongated shapes >> 1)
    aspect_ratio = safe_div(major, minor) if minor > 0 else float("nan")

    # Elongation in [0, 1):
    #   0 -> circle-like, ->1 very elongated
    elongation = (
        safe_div(major - minor, major + minor) if (major + minor) > 0 else float("nan")
    )

    # Solidity = area / convex_area (1 for convex shapes)
    solidity = float(getattr(prop, "solidity", float("nan")))

    # Extent = area / bounding_box_area
    extent = float(getattr(prop, "extent", float("nan")))

    # Eccentricity of the ellipse that has the same second-moments
    eccentricity = float(getattr(prop, "eccentricity", float("nan")))

    # Compactness (inverse of circularity, higher = less compact)
    compactness = safe_div(perimeter**2, 4.0 * np.pi * area)

    # Bounding box aspect ratio
    minr, minc, maxr, maxc = prop.bbox
    bbox_h = float(maxr - minr)
    bbox_w = float(maxc - minc)
    bbox_aspect = (
        safe_div(max(bbox_h, bbox_w), min(bbox_h, bbox_w))
        if min(bbox_h, bbox_w) > 0
        else float("nan")
    )

    return {
        "area": area,
        "perimeter": perimeter,
        "major axis length": major,
        "minor axis length": minor,
        "circularity": circularity,
        "roundness": roundness,
        "aspect ratio": aspect_ratio,
        "elongation": elongation,
        "solidity": solidity,
        "extent": extent,
        "eccentricity": eccentricity,
        "compactness": compactness,
        "bbox aspect ratio": bbox_aspect,
    }


@st.cache_data(show_spinner="Building analysis DataFrame...")
def build_analysis_df(records):
    """
    Build a DataFrame with per-mask metrics for all images in session state.

    Columns (existing):
        image, mask #, mask label, mask area, mask perimeter, max edge-to-edge

    New columns (from mask_shape_metrics):
        circularity, roundness, aspect ratio, elongation,
        solidity, extent, eccentricity, compactness,
        bbox aspect ratio, chord / major axis
    """

    rows = []
    # iterate through the image records
    for k in ordered_keys():
        rec = st.session_state.images[k]
        inst = rec.get("masks")
        # skip invalid masks
        if not inst.any():
            continue

        labdict = rec.get("labels", {})  # dict {instance_id -> class/None}
        for prop in regionprops(inst):  # prop.label is the instance id
            iid = int(prop.label)
            cls = labdict.get(iid)

            # compute shape metrics
            shape_metrics = mask_shape_metrics(prop)

            row = {
                "image": rec["name"],
                "mask #": iid,
                "mask label": ("Unlabelled" if cls in (None, "No label") else cls),
            }

            # merge in the shape metrics
            row.update(shape_metrics)

            rows.append(row)

    return pd.DataFrame(rows)


# --- FUNCTIONS FOR DOWNLOADING CLASS CHARACTERISTICS PLOTS


def build_cell_metrics_zip(labels_selected):
    """
    Build a ZIP archive (bytes) containing cell metrics CSV and plots
    for the selected labels (list of str). If empty, include all labels.
    """
    df = build_analysis_df(st.session_state["images"])
    if labels_selected:
        df = df[df["mask label"].isin(labels_selected)]
    items = []
    if not df.empty:
        items.append(("cell_analysis.csv", df.to_csv(index=False).encode("utf-8")))

    items += st.session_state.get("analysis_plots", [])
    return build_plots_zip(items) if items else b""


def build_plots_zip(plot_paths_or_bytes) -> bytes:
    """
    Given a list of file paths (str or Path) or tuples of (filename, bytes),
    build a ZIP archive (bytes) containing those files.
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


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle


def show_shape_metric_reference():
    st.subheader("Shape Descriptor Reference")

    st.markdown(
        """
        Below is a quick reference for the shape metrics computed from each labeled region.
        Use this as a guide when interpreting the measurements for your segmented cells.

        **Notation:**  
        - \(A\): area (number of pixels in the object)  
        - \(P\): perimeter (length of the object's boundary)  
        - \(a, b\): semi-major and semi-minor axes of the best-fit ellipse  

        All quantities are reported in pixel units.
        """
    )

    # --- Textual definitions -------------------------------------------------
    metrics = [
        {
            "Name": "area (A)",
            "What it describes": "Size of the object in pixels.",
            "How it is calculated": "Number of pixels inside the masked region.",
        },
        {
            "Name": "perimeter (P)",
            "What it describes": "Length of the object's boundary.",
            "How it is calculated": "Length of the outer contour of the masked region.",
        },
        {
            "Name": "major axis length",
            "What it describes": (
                "Longest axis of the best-fit ellipse. "
                "Larger values (relative to object size) indicate a more elongated object."
            ),
            "How it is calculated": (
                "Length of the major axis of the ellipse with the same second moments as the region."
            ),
        },
        {
            "Name": "minor axis length",
            "What it describes": "Shortest axis of the best-fit ellipse.",
            "How it is calculated": (
                "Length of the minor axis of the ellipse with the same second moments as the region."
            ),
        },
        {
            "Name": "circularity",
            "What it describes": (
                "How close the shape is to a perfect circle. "
                "Circularity = 1 for a perfect circle; irregular or elongated shapes have values < 1."
            ),
            "How it is calculated": "4 · π · A / P²",
        },
        {
            "Name": "roundness",
            "What it describes": (
                "Circle-likeness based on the major axis. "
                "Equals 1 for a perfect circle (if the major axis corresponds to the diameter). "
                "Lower values indicate elongation."
            ),
            "How it is calculated": "4 · A / (π · major_axis_length²)",
        },
        {
            "Name": "aspect ratio",
            "What it describes": (
                "Ratio of major to minor axis length. "
                "Values ≥ 1; higher values indicate more elongation of the best-fit ellipse."
            ),
            "How it is calculated": "major_axis_length / minor_axis_length",
        },
        {
            "Name": "elongation",
            "What it describes": (
                "Normalized elongation in the range [0, 1). "
                "Values near 0 indicate circle-like shapes; values approaching 1 indicate strong elongation."
            ),
            "How it is calculated": (
                "(major_axis_length − minor_axis_length) / "
                "(major_axis_length + minor_axis_length)"
            ),
        },
        {
            "Name": "solidity",
            "What it describes": (
                "How filled the object is relative to its convex hull. "
                "A value of 1 indicates a perfectly convex shape; lower values indicate concavities or "
                "irregular boundaries."
            ),
            "How it is calculated": "area / convex_area",
        },
        {
            "Name": "extent",
            "What it describes": (
                "Fraction of the bounding box area occupied by the object. "
                "Values near 1 indicate that the object nearly fills its bounding box."
            ),
            "How it is calculated": "area / bounding_box_area",
        },
        {
            "Name": "eccentricity",
            "What it describes": (
                "Eccentricity of the ellipse with matching second moments. "
                "0 corresponds to a perfect circle; values approaching 1 correspond to highly elongated shapes."
            ),
            "How it is calculated": "√(1 − (b² / a²)), using semi-major axis a and semi-minor axis b.",
        },
        {
            "Name": "compactness",
            "What it describes": (
                "Inverse of circularity; a measure of boundary irregularity. "
                "Equal to 1 for a perfect circle; > 1 for less compact or more jagged shapes."
            ),
            "How it is calculated": "P² / (4 · π · A)",
        },
        {
            "Name": "bbox aspect ratio",
            "What it describes": (
                "Elongation of the axis-aligned bounding box. "
                "Values ≥ 1; higher values indicate a more elongated bounding region."
            ),
            "How it is calculated": "max(bbox_height, bbox_width) / min(bbox_height, bbox_width)",
        },
    ]

    # --- Helper functions for illustrations ----------------------------------

    def plot_circularity_compactness():
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.set_aspect("equal")

        # Circle – high circularity, compact
        circle = Circle((0.5, 0.5), 0.25, fill=False)
        ax.add_patch(circle)
        ax.text(
            0.5,
            0.15,
            "Circle\n(circularity ≈ 1,\ncompactness ≈ 1)",
            ha="center",
            va="top",
            fontsize=8,
        )

        # Irregular blob – lower circularity, less compact
        blob = Polygon(
            [
                [1.1, 0.75],
                [1.4, 0.6],
                [1.35, 0.4],
                [1.2, 0.3],
                [1.0, 0.35],
                [0.95, 0.55],
            ],
            closed=True,
            fill=False,
        )
        ax.add_patch(blob)
        ax.text(
            1.25,
            0.15,
            "Irregular\n(circularity < 1,\ncompactness > 1)",
            ha="center",
            va="top",
            fontsize=8,
        )

        ax.set_xlim(0, 1.7)
        ax.set_ylim(0, 1.0)
        ax.axis("off")
        return fig

    def plot_aspect_elong_ecc():
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.set_aspect("equal")

        # Nearly circular ellipse
        ell1 = Ellipse((0.5, 0.5), 0.4, 0.35, fill=False)
        ax.add_patch(ell1)
        ax.text(
            0.5,
            0.15,
            "Almost circle\n(aspect ratio ≈ 1,\nlow elongation,\nlow eccentricity)",
            ha="center",
            va="top",
            fontsize=8,
        )

        # Elongated ellipse
        ell2 = Ellipse((1.3, 0.5), 0.7, 0.2, fill=False)
        ax.add_patch(ell2)
        ax.text(
            1.3,
            0.15,
            "Elongated\n(aspect ratio ≫ 1,\nhigher elongation,\nhigher eccentricity)",
            ha="center",
            va="top",
            fontsize=8,
        )

        ax.set_xlim(0, 1.8)
        ax.set_ylim(0, 1.0)
        ax.axis("off")
        return fig

    def plot_solidity():
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.set_aspect("equal")

        # Convex shape
        convex = Polygon(
            [[0.2, 0.2], [0.6, 0.25], [0.7, 0.6], [0.3, 0.8]],
            closed=True,
            fill=False,
        )
        ax.add_patch(convex)
        ax.text(
            0.4,
            0.1,
            "Convex\n(solidity ≈ 1)",
            ha="center",
            va="top",
            fontsize=8,
        )

        # Shape with indentation (concavity)
        concave = Polygon(
            [
                [1.0, 0.2],
                [1.4, 0.25],
                [1.45, 0.5],
                [1.2, 0.45],
                [1.35, 0.8],
                [1.0, 0.75],
            ],
            closed=True,
            fill=False,
        )
        ax.add_patch(concave)
        ax.text(
            1.25,
            0.1,
            "Concave\n(solidity < 1)",
            ha="center",
            va="top",
            fontsize=8,
        )

        ax.set_xlim(0, 1.8)
        ax.set_ylim(0, 1.0)
        ax.axis("off")
        return fig

    def plot_extent_bbox_aspect():
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.set_aspect("equal")

        # Compact object filling box (high extent)
        bbox1 = Rectangle((0.1, 0.1), 0.6, 0.6, fill=False, linestyle="dotted")
        ax.add_patch(bbox1)
        obj1 = Rectangle((0.15, 0.15), 0.5, 0.5, fill=False)
        ax.add_patch(obj1)
        ax.text(
            0.4,
            0.05,
            "High extent\n(object almost fills box)",
            ha="center",
            va="top",
            fontsize=8,
        )

        # Thin object inside tall box (low extent, high bbox aspect ratio)
        bbox2 = Rectangle((1.0, 0.1), 0.4, 0.8, fill=False, linestyle="dotted")
        ax.add_patch(bbox2)
        obj2 = Rectangle((1.05, 0.45), 0.3, 0.1, fill=False)
        ax.add_patch(obj2)
        ax.text(
            1.2,
            0.05,
            "Low extent,\nbox aspect ratio ≫ 1",
            ha="center",
            va="top",
            fontsize=8,
        )

        ax.set_xlim(0, 1.8)
        ax.set_ylim(0, 1.0)
        ax.axis("off")
        return fig

    # --- Per-metric expanders ("popovers") -----------------------------------

    for m in metrics:
        with st.expander(m["Name"]):
            st.markdown(
                f"**What it describes:** {m['What it describes']}\n\n"
                f"**How it is calculated:** {m['How it is calculated']}"
            )

            # Attach the appropriate illustration(s) where relevant
            if m["Name"] in ["circularity", "compactness"]:
                fig = plot_circularity_compactness()
                st.pyplot(fig)
                st.caption(
                    "For the same area A, shapes with longer perimeters P have lower circularity "
                    "and higher compactness."
                )

            if m["Name"] in ["aspect ratio", "elongation", "eccentricity"]:
                fig = plot_aspect_elong_ecc()
                st.pyplot(fig)
                st.caption(
                    "As the ellipse becomes more stretched (larger major_axis_length / minor_axis_length), "
                    "aspect ratio, elongation, and eccentricity all increase."
                )

            if m["Name"] == "solidity":
                fig = plot_solidity()
                st.pyplot(fig)
                st.caption(
                    "Solidity = area / convex_area. A convex shape matches its convex hull (solidity ≈ 1). "
                    "Indentations or holes reduce the area relative to the hull, lowering solidity."
                )

            if m["Name"] in ["extent", "bbox aspect ratio"]:
                fig = plot_extent_bbox_aspect()
                st.pyplot(fig)
                st.caption(
                    "Extent measures how much of the bounding box area the object occupies. "
                    "The bounding-box aspect ratio compares its longer side to its shorter side; "
                    "thin, elongated boxes have large aspect ratios."
                )
