# panels/train_densenet.py
import random

import numpy as np
import streamlit as st
import cv2
import matplotlib.pyplot as plt
import io as IO
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import io
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime
import pandas as pd

from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score


# ---- bring in existing app helpers ----
from helpers.state_ops import ordered_keys
from helpers.cellpose_functions import _save_fig_to_session, normalize_image

ss = st.session_state

# -------------------------------
#  Preprocessing and loader functions
# -------------------------------


def generate_cell_patch(image: np.ndarray, mask: np.ndarray, patch_size: int = 64):
    """takes an image and boolean mask input and a normalized square patch image from the mask"""

    im, m = np.asarray(image), np.asarray(mask, bool)

    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop, mc = im[y0:y1, x0:x1], m[y0:y1, x0:x1]
    crop = (crop * mc[..., None] if crop.ndim == 3 else crop * mc).astype(im.dtype)

    # checks to make sure crop is the correct format
    if crop.ndim == 2:
        crop = np.stack([crop] * 3, axis=-1)
    elif crop.ndim == 3 and crop.shape[2] == 4:
        crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
    elif crop.ndim == 3 and crop.shape[2] == 3:
        # assume BGR if from cv; convert to RGB for consistency in augmentation
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    else:
        crop = crop[..., :3]

    crop = resize_with_aspect_ratio(crop, patch_size=patch_size)
    return crop.astype(np.float32)


def resize_with_aspect_ratio(img: np.ndarray, patch_size=64) -> np.ndarray:
    """resizes input image to a square of with 'patch_size' height while maintaining the aspect ratio"""
    th, tw = patch_size, patch_size
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((th, tw, img.shape[2] if img.ndim == 3 else 1), dtype=img.dtype)
    scale = min(th / h, tw / w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )
    # center pad
    if img.ndim == 2:
        canvas = np.zeros((th, tw), dtype=img.dtype)
        y0, x0 = (th - nh) // 2, (tw - nw) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
        return canvas
    else:
        c = img.shape[2]
        canvas = np.zeros((th, tw, c), dtype=img.dtype)
        y0, x0 = (th - nh) // 2, (tw - nw) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw, :] = resized
        return canvas


def generate_patches_with_ids(rec, patch_size=64):
    """returns list of cell patches and patch ids from input record"""

    M = rec.get("masks")

    # extract the individual masks
    ids = [int(v) for v in np.unique(M) if v != 0]
    patches, keep_ids = [], []
    for iid in ids:
        patches.append(
            generate_cell_patch(
                image=rec["image"], mask=M == iid, patch_size=patch_size
            )
        )
        keep_ids.append(iid)

    return patches, keep_ids


# -------------------------------
#  Model inference on patch
# -------------------------------


def classify_cells_with_densenet(rec: dict) -> None:
    """Classify segmented cell masks in `rec` using a DenseNet-121 model.
    Mutates `rec` and session_state, then triggers a rerun on success.
    """

    model = ss.get("densenet_model")
    M = rec.get("masks")

    # exit if no masks for classification
    if not isinstance(M, np.ndarray) or M.ndim != 2 or not np.any(M):
        return

    # extract noramlized cell patches. keep ids to that class can be added to the correct mask
    patches, keep_ids = generate_patches_with_ids(rec)

    # normalize the patches
    patches = [normalize_image(patch) for patch in patches]

    # classify the patches
    X = np.stack(patches, axis=0)
    preds = model.predict(X, verbose=0).argmax(axis=1)

    # add class predictions to the record
    all_classes = [c for c in ss.get("all_classes", []) if c != "No label"] or [
        "class0",
        "class1",
    ]
    labels = rec.setdefault("labels", {})
    for iid, cls_idx in zip(keep_ids, preds):
        idx = int(cls_idx)
        name = all_classes[idx] if idx < len(all_classes) else str(idx)
        labels[int(iid)] = name
        if name and name != "No label" and name not in ss.get("all_classes", []):
            ss.setdefault("all_classes", []).append(name)

    ss.images[ss.current_key] = rec


# -------------------------------
#  Densenet121 training: Augementation
# -------------------------------


def geo_rotate(img, angle: int, keep_resolution: bool = True):
    """
    Rotate by `angle` degrees. If keep_resolution=True, expand canvas to fit the whole image.
    Supports HxW or HxWxC (uint8 or float).
    """
    h, w = img.shape[:2]
    cX, cY = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    if keep_resolution:
        # compute new bounds
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        nW = int(h * sin + w * cos)
        nH = int(h * cos + w * sin)

        # adjust translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        borderMode = cv2.BORDER_REFLECT_101
        rotated = cv2.warpAffine(
            img,
            M,
            (nW, nH),
            flags=cv2.INTER_LINEAR,
            borderMode=borderMode,
        )
    else:
        rotated = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    return resize_with_aspect_ratio(rotated)


def kernel_blur(img, method: str, ksize=(3, 3), gaussian_sigma: float = 1.0):
    """
    Minimal shim matching your previous calls: only 'gaussian' is used.
    """
    if method != "gaussian":
        raise ValueError("Only 'gaussian' blur is supported in this shim.")
    kx, ky = ksize
    kx = kx if kx % 2 == 1 else kx + 1
    ky = ky if ky % 2 == 1 else ky + 1
    return cv2.GaussianBlur(img, (kx, ky), sigmaX=gaussian_sigma)


def photo_bc(img, alpha: float = 1.0, beta: float = 0.0):
    """
    Brightness/contrast adjust:
      out = alpha * img + beta
    Works for uint8 or float; returns same dtype as input.
    """
    if img.dtype == np.uint8:
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    # float path
    out = img.astype(np.float32) * float(alpha) + float(beta)
    if out.dtype != img.dtype:
        out = out.astype(img.dtype)
    return out


def random_flip(img):
    flip_type = random.choice(["horizontal", "vertical"])
    return np.fliplr(img) if flip_type == "horizontal" else np.flipud(img)


def rotate_image(img):
    angle = random.randint(-100, 100)
    return geo_rotate(img, angle=angle, keep_resolution=True)


def adjust_brightness_contrast(img):
    alpha = random.uniform(0.8, 1.5)
    beta = random.randint(15, 35)
    return photo_bc(img, alpha=alpha, beta=beta)


def blur_gaussian(img):
    ksize = random.choice([1, 3, 5])
    return kernel_blur(img, "gaussian", ksize=(ksize, ksize), gaussian_sigma=1)


def no_change(img):
    return img


augmentations = [
    random_flip,
    rotate_image,
    adjust_brightness_contrast,
    blur_gaussian,
    no_change,
]


def random_augmentation_pipeline(image_np, num_transforms=3):
    selected = random.sample(augmentations, num_transforms)
    out = image_np
    for t in selected:
        out = t(out)
    return out


# -------------------------------
#  Densenet training: DenseNet121 Model
# -------------------------------


def build_densenet(input_shape=(64, 64, 3), num_classes=2):
    # Fresh DenseNet (no ImageNet normalization dependency)
    base = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False
    x_in = layers.Input(shape=input_shape)
    x = base(x_in, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=x_in, outputs=x)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -------------------------------
#  Densenet121 training: image loading
# -------------------------------


def load_labeled_patches_from_session(patch_size: int = 64):
    """
    Build X, y from all loaded images with labels.
    """
    ims = st.session_state.get("images", {}) or {}
    all_classes = [
        c for c in st.session_state.get("all_classes", []) if c != "No label"
    ]
    if not all_classes:
        all_classes = ["class0", "class1"]
    name_to_idx = {c: i for i, c in enumerate(all_classes)}

    X, y = [], []
    for k in ordered_keys():
        rec = ims.get(k) or {}
        img, M, labs = rec.get("image"), rec.get("masks"), rec.get("labels", {})
        if img is None or not isinstance(M, np.ndarray) or M.ndim != 2 or not np.any(M):
            continue
        ids = [int(v) for v in np.unique(M) if v != 0]
        for iid in ids:
            cname = labs.get(int(iid))
            if not cname or cname == "No label":
                continue
            patch = generate_cell_patch(
                image=img, mask=(M == iid), patch_size=patch_size
            )

            X.append(patch)
            y.append(name_to_idx[cname])

    if not X:
        return (
            np.zeros((0, patch_size, patch_size, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            all_classes,
        )

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)
    return X, y, all_classes


class AugSequence(Sequence):
    """class for loading image/label pairs into densenet for training."""

    def __init__(
        self, X, y, batch_size=32, num_transforms=3, shuffle=True, target_size=None
    ):
        self.X = X
        self.y = y
        self.bs = batch_size
        self.nt = num_transforms
        self.shuffle = shuffle
        if target_size is None:
            self.target_size = (X.shape[1], X.shape[2])  # (H, W)
        else:
            self.target_size = tuple(target_size)
        self.indices = np.arange(len(X))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return max(1, int(np.ceil(len(self.X) / self.bs)))

    @property
    def num_batches(self):
        return len(self)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        sl = slice(idx * self.bs, (idx + 1) * self.bs)
        inds = self.indices[sl]
        Xb = self.X[inds]
        yb = self.y[inds]

        out = []
        for img in Xb:
            img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            aug = random_augmentation_pipeline(img_uint8, num_transforms=self.nt)
            aug = normalize_image(aug)
            out.append(aug.astype(np.float32))

        Xo = np.stack(out, axis=0)
        return Xo, yb


def fine_tune_densenet(input_size, batch_size, epochs, val_split):
    # Load data
    X, y, classes = load_labeled_patches_from_session(patch_size=input_size)
    if X.shape[0] < 2 or len(np.unique(y)) < 2:
        st.warning("Need at least 2 samples and 2 classes. Add more labeled cells.")
        return

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, stratify=y, random_state=42
    )

    # Data generators
    train_gen = AugSequence(
        X_train,
        y_train,
        batch_size=batch_size,
        num_transforms=3,
        shuffle=True,
        target_size=(input_size, input_size),
    )
    val_gen = AugSequence(
        X_val,
        y_val,
        batch_size=batch_size,
        num_transforms=0,
        shuffle=False,
        target_size=(input_size, input_size),
    )

    # Class weights
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(classes)), y=y
    )
    class_weights_dict = {i: float(w) for i, w in enumerate(class_weights)}

    # Build model
    model = build_densenet(
        input_shape=(input_size, input_size, 3),
        num_classes=len(classes),
    )

    # Train
    es = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )
    with st.spinner("Training DenseNet…"):
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=[es],
            class_weight=class_weights_dict,
            verbose=0,
        )

    # Persist the model in session
    ss["densenet_ckpt_name"] = "densenet_finetuned"
    ss["densenet_model"] = model

    return history, val_gen, classes


from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def evaluate_fine_tinued_densenet(history, val_gen, classes):
    # 1) collect full val set
    Xv, yv = [], []
    for i in range(len(val_gen)):
        xb, yb = val_gen[i]
        Xv.append(xb)
        yv.append(yb)
    Xv = np.concatenate(Xv, axis=0)
    yv = np.concatenate(yv, axis=0)

    # 2) ensure yv are class indices (not one-hot/probabilities)
    if yv.ndim == 2:  # e.g., shape (N, C)
        y_true = np.argmax(yv, axis=1)
    else:
        y_true = yv.astype(int)

    # 3) predictions as indices
    y_probs = st.session_state["densenet_model"].predict(Xv, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    st.info(
        "Model stored in session. You can use it immediately from the **Classify cells** panel."
    )

    # 4) metrics (macro often more informative; weighted also shown)
    acc = accuracy_score(y_true, y_pred)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics_dict = {
        "Accuracy": acc,
        "Precision (weighted)": prec_w,
        "F1 (weighted)": f1_w,
        "Precision (macro)": prec_m,
        "F1 (macro)": f1_m,
    }

    # 5) plots
    fig = _plot_densenet_losses(
        history.history.get("loss", []),
        history.history.get("val_loss", []),
        metrics={
            "Accuracy": acc,
            "Precision": prec_w,
            "F1": f1_w,
        },  # keep your bar chart concise
    )

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    _ = _plot_confusion_matrix(cm, classes, normalize=False)

    # (optional) quick text report in the app for sanity-checking
    st.code(
        classification_report(
            y_true, y_pred, target_names=list(classes), zero_division=0
        )
    )


# -------------------------------
#  Functions visualizing and downloading training metrics
# -------------------------------


def _plot_confusion_matrix(
    cm: np.ndarray, class_names: list[str], *, normalize: bool = True
):
    """Nice confusion matrix:
    - optional row-normalization to percentages
    - count + % annotated in each cell
    - readable axes & colors
    """
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cmn = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
            cmn = np.nan_to_num(cmn)
    else:
        cmn = cm

    fig, ax = plt.subplots(
        figsize=(max(5, 0.8 * len(class_names)), max(4, 0.8 * len(class_names)))
    )
    ax.imshow(cmn, interpolation="nearest", cmap="Blues")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    # annotate cells with count (+ percent if normalized)
    thresh = cmn.max() * 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            if normalize:
                txt = f"{count}\n{cmn[i,j]*100:.1f}%"
            else:
                txt = f"{count}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if cmn[i, j] > thresh else "black",
                fontsize=9,
            )

    ax.grid(False)
    fig.tight_layout()

    _save_fig_to_session(fig, key_prefix=f"densenet_plot_confusion", dpi=300)

    return fig


def _plot_densenet_losses(train_losses, test_losses, metrics=None):
    """
    Plot training/validation loss curves and, if provided,
    a bar chart of evaluation metrics.
    metrics should be a dict like {"accuracy": 0.625, "precision": 0.828, "F1": 0.702}.
    """
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # --- left subplot: loss curves ---
    ax1.plot(epochs, train_losses, label="train")
    if test_losses is not None and len(test_losses) == len(train_losses):
        test_epochs = [e for e, v in zip(epochs, test_losses) if v != 0]
        test_vals = [v for v in test_losses if v != 0]
        ax1.plot(test_epochs, test_vals, label="val")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("DenseNet training/validation loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- right subplot: metrics bar chart ---
    if metrics is not None:
        labels = list(metrics.keys())
        values = list(metrics.values())
        ax2.bar(
            labels,
            values,
            color=["tab:blue", "tab:orange", "tab:green"],
            alpha=0.8,
            edgecolor="black",
        )
        ax2.set_ylim(0, 1.0)
        ax2.set_title("Validation metrics")
        for i, v in enumerate(values):
            ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    else:
        ax2.axis("off")

    # 🔸 Save PNG to session_state
    _save_fig_to_session(fig, key_prefix=f"densenet_plot_losses", dpi=300)

    return fig


def _array_to_png_bytes(arr: np.ndarray) -> bytes:
    """Convert float/uint arrays to PNG bytes (3-channel)."""
    a = arr
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    elif a.ndim == 3 and a.shape[2] > 3:
        a = a[:, :, :3]

    if a.dtype.kind == "f":  # float -> assume 0..1 or 0..255
        a = np.clip(a, 0, 255)
        if a.max() <= 1.0:
            a = (a * 255.0).round()
    a = np.clip(a, 0, 255).astype(np.uint8)

    bio = io.BytesIO()
    Image.fromarray(a).save(bio, format="PNG")
    return bio.getvalue()


def build_patchset_zip_from_session(patch_size: int = 64) -> bytes | None:
    X, y, classes = load_labeled_patches_from_session(patch_size=patch_size)
    if X.shape[0] == 0:
        return None

    buf, rows = IO.BytesIO(), []
    ok = ordered_keys()
    parent_names = [st.session_state["images"][k]["name"].rsplit(".", 1)[0] for k in ok]
    patch_per_img = int(np.ceil(X.shape[0] / len(ok))) if ok else X.shape[0]

    with ZipFile(buf, "w", ZIP_DEFLATED) as zf:
        for i in range(X.shape[0]):
            parent = parent_names[min(i // patch_per_img, len(parent_names) - 1)]
            fname = f"{parent}_patch_{i+1:04d}.png"
            label_idx = int(y[i])
            label_name = (
                classes[label_idx]
                if 0 <= label_idx < len(classes)
                else f"class{label_idx}"
            )
            zf.writestr(f"patches/{fname}", _array_to_png_bytes(X[i]))
            rows.append(
                {"filename": fname, "label_idx": label_idx, "label": label_name}
            )
        zf.writestr(
            "labels.csv", pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        )
    return buf.getvalue()


import io as IO, os, tempfile
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime
import pandas as pd
import streamlit as st


def download_densenet_training_record():
    ss = st.session_state
    psize = int(ss.get("dn_input_size", 64))
    pzip = build_patchset_zip_from_session(psize)
    if not pzip:
        st.warning("No patches available.")
        return

    with ZipFile(IO.BytesIO(pzip)) as zin:
        labels = pd.read_csv(IO.BytesIO(zin.read("labels.csv")))
        params = dict(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            input_size=psize,
            epochs=int(ss.get("dn_max_epoch", 100)),
            batch_size=int(ss.get("dn_batch_size", 32)),
            val_split=float(ss.get("dn_val_split", 0.2)),
            patches=len(labels),
            classes=labels["label"].nunique(),
        )

        buf = IO.BytesIO()
        with ZipFile(buf, "w", ZIP_DEFLATED) as zout:
            # --- Models ---
            if ss.get("densenet_model_bytes"):
                zout.writestr("densenet_model.pt", ss["densenet_model_bytes"])

            if ss.get("densenet_model") is not None:
                # Keras 3: save to a real filepath with .keras extension
                tmp = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
                tmp_path = tmp.name
                tmp.close()
                try:
                    ss["densenet_model"].save(tmp_path)  # format inferred from ".keras"
                    with open(tmp_path, "rb") as f:
                        zout.writestr("densenet_model.keras", f.read())
                finally:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

            # --- Params ---
            zout.writestr(
                "params.csv",
                pd.Series(params)
                .rename_axis("parameter")
                .reset_index(name="value")
                .to_csv(index=False),
            )

            # --- Patchset (images + labels.csv) ---
            for n in zin.namelist():
                zout.writestr(n, zin.read(n))

            # --- Plots ---
            for k, p in [
                ("densenet_plot_losses_png", "plots/densenet_losses.png"),
                ("densenet_plot_confusion_png", "plots/densenet_confusion.png"),
            ]:
                if k in ss:
                    zout.writestr(p, ss[k])

    st.download_button(
        "Download DenseNet model, dataset and training metrics (ZIP)",
        data=buf.getvalue(),
        file_name=f"densenet_training_{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip",
        mime="application/zip",
        use_container_width=True,
        type="primary",
    )
