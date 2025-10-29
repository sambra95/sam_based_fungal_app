import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
from helpers.state_ops import ordered_keys
from skimage.measure import regionprops


def _violin(df: pd.DataFrame, value_col: str):

    sub = df.copy()
    sub["label"] = sub["mask label"].replace("No label", None).fillna("Unlabelled")
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
    sub["label"] = sub["mask label"].replace("No label", None).fillna("Unlabelled")
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


def _build_analysis_df():
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
