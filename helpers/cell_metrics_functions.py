import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import io

from pathlib import Path
from helpers.state_ops import ordered_keys
from skimage.measure import regionprops, label


def _stem(n):
    return Path(n).stem


def _perimeter_px(mm: np.ndarray) -> int:
    mb = mm.astype(bool)
    interior = (
        mb
        & np.roll(mb, 1, 0)
        & np.roll(mb, -1, 0)
        & np.roll(mb, 1, 1)
        & np.roll(mb, -1, 1)
    )
    edge = mb & ~interior
    return int(edge.sum())


def _violin(df: pd.DataFrame, value_col: str):
    import seaborn as sns, matplotlib.pyplot as plt

    sub = df.copy()
    sub["label"] = sub["mask label"].replace("Remove label", None).fillna("Unlabelled")
    order = sorted(sub["label"].unique(), key=lambda x: (x != "Unlabelled", str(x)))
    pal = sns.color_palette("Set2", n_colors=len(order))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    sns.violinplot(
        data=sub,
        x="label",
        y=value_col,
        order=order,
        inner="box",
        cut=0,
        palette=pal,
        linewidth=1.0,
        ax=ax,
    )
    for art in ax.collections:
        try:
            art.set_edgecolor("black")
            art.set_linewidth(0.8)
        except Exception:
            pass
    sns.swarmplot(
        data=sub,
        x="label",
        y=value_col,
        order=order,
        size=3,
        alpha=0.75,
        linewidth=0.3,
        edgecolor="black",
        color="k",
        ax=ax,
    )
    ax.set_xlabel("Label")
    ax.set_ylabel(value_col.replace("_", " ").title())
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    sns.despine(ax=ax)

    # show + return PNG bytes
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"violin_{value_col.replace(' ', '_')}.png", buf.getvalue()


def _bar(df: pd.DataFrame, value_col: str):
    import seaborn as sns, matplotlib.pyplot as plt

    sub = df.copy()
    sub["label"] = sub["mask label"].replace("Remove label", None).fillna("Unlabelled")
    order = sorted(sub["label"].unique(), key=lambda x: (x != "Unlabelled", str(x)))
    pal = sns.color_palette("Set2", n_colors=len(order))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    sns.barplot(
        data=sub, x="label", y=value_col, order=order, palette=pal, errorbar="sd", ax=ax
    )
    for p in ax.patches:
        p.set_edgecolor("black")
        p.set_linewidth(0.8)
    sns.stripplot(
        data=sub,
        x="label",
        y=value_col,
        order=order,
        color="k",
        alpha=0.6,
        size=3,
        jitter=0.25,
        linewidth=0.3,
        edgecolor="black",
        ax=ax,
    )
    ax.set_xlabel("Label")
    ax.set_ylabel(value_col.replace("_", " ").title())
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    sns.despine(ax=ax)

    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"bar_{value_col.replace(' ', '_')}.png", buf.getvalue()


def _build_analysis_df() -> pd.DataFrame:
    rows = []
    for k in ordered_keys():
        rec = st.session_state.images[k]
        masks = rec.get("masks")

        labs = list(rec.get("labels", []))

        base = _stem(rec["name"])
        N = masks.shape[0]
        for i in range(N):
            area = masks[i] > 0
            prop = regionprops(label(masks[i]))[0]
            rows.append(
                {
                    "image": base,
                    "mask #": i,  # use i+1 if you prefer 1-based
                    "mask label": ("Unlabelled" if labs[i] == None else labs[i]),
                    "mask area": int(area.sum()),
                    "mask perimeter": prop.perimeter,
                    "mask eccentricity": prop.eccentricity,
                    "mask solidity": prop.eccentricity,
                }
            )
    return pd.DataFrame(rows)


def build_image_summary_df() -> pd.DataFrame:
    rows = []
    for k in ordered_keys():
        rec = st.session_state.images[k]
        masks = rec.get("masks")
        labs = list(rec.get("labels", []))
        base = _stem(rec["name"])

        if masks is None:
            continue

        group_counts = {}
        total = 0
        N = masks.shape[0]

        for i in range(N):
            m = masks[i]
            if not np.any(m):
                continue

            # Determine group label
            raw_lab = labs[i] if i < len(labs) else None
            group = raw_lab if (raw_lab is not None and raw_lab != "") else "Unlabelled"

            # Count how many connected components are inside this mask
            n_components = len(regionprops(label(m)))
            total += n_components
            group_counts[group] = group_counts.get(group, 0) + n_components

        # Build row: total + group counts
        row = {"image": base, "total cells": total}
        row.update(group_counts)
        rows.append(row)

    # Normalize to DataFrame
    df = pd.DataFrame(rows).fillna(0).set_index("image")
    # Ensure integer columns
    df = df.astype(int)
    return df.reset_index()
