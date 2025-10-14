# panels/train_densenet.py
import random
from dataclasses import dataclass

import numpy as np
import streamlit as st
import cv2
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt

from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import preprocess_input

# ---- bring in existing app helpers ----
from helpers.state_ops import ordered_keys
from helpers.cellpose_functions import _save_fig_to_session

ss = st.session_state

# -------------------------------
#  Preprocessing helpers
# -------------------------------


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Cellpose-like normalization:
      - divide by mean intensity (scale to ~0.5 mean)
      - rescale to [0, 1]
    """
    im = image.astype(np.float32)
    mean_val = float(np.mean(im))
    if mean_val == 0:
        # fall back to simple scaling to avoid crashing a training run
        return (im - im.min()) / max(1e-6, (im.max() - im.min()))
    meannorm = im * (0.5 / mean_val)
    return rescale_intensity(meannorm, in_range="image", out_range=(0, 1))


def resize_with_aspect_ratio(img, target_size=64):
    """resize input image to square with target_size dimensions. maintains aspect ratio"""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # create square canvas
    canvas = np.zeros(
        (target_size, target_size, *([img.shape[2]] if img.ndim == 3 else [])),
        dtype=img.dtype,
    )
    y0 = (target_size - new_h) // 2
    x0 = (target_size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _resize_keep_aspect(img: np.ndarray, target=(64, 64)) -> np.ndarray:
    th, tw = target
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


def preprocess_patch_for_training(crop: np.ndarray, target=(64, 64)) -> np.ndarray:
    """
    - ensure 3-ch RGB
    - keep aspect ratio with padding to target
    - normalize to [0,1] via normalize_image
    - return float32
    """
    if crop.ndim == 2:
        crop3 = np.stack([crop] * 3, axis=-1)
    elif crop.ndim == 3 and crop.shape[2] == 4:
        crop3 = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
    elif crop.ndim == 3 and crop.shape[2] == 3:
        # assume BGR if from cv; convert to RGB for consistency in augmentation
        crop3 = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    else:
        crop3 = crop[..., :3]

    sq = _resize_keep_aspect(crop3, target)
    sq = normalize_image(sq)  # 0..1 float
    return sq.astype(np.float32)


# -------------------------------
#  Augementation functions
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
    return rotated


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
#  Model
# -------------------------------


def build_densenet(input_shape=(64, 64, 3), num_classes=2, base_trainable=False):
    # Fresh DenseNet (no ImageNet normalization dependency)
    base = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = base_trainable
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
#  Data building from uploaded session images
# -------------------------------


@dataclass
class PatchRecord:
    img: np.ndarray
    label_idx: int


def _iter_instance_patches():
    ims = ss.get("images", {}) or {}
    for k in ordered_keys():
        rec = ims.get(k) or {}
        img = rec.get("image")
        M = rec.get("masks")
        labmap = rec.get("labels", {}) or {}
        if img is None or not isinstance(M, np.ndarray) or M.ndim != 2 or not np.any(M):
            continue
        ids = [int(v) for v in np.unique(M) if v != 0]
        for iid in ids:
            cname = labmap.get(int(iid))
            if not cname or cname == "Remove label":
                continue
            # cropped RGB patch around this instance
            patch = extract_masked_cell_patch(img, (M == iid).astype(np.uint8), size=0)
            if patch is None or patch.size == 0:
                continue
            yield patch, cname


def _build_label_index():
    # Use current session classes (excluding "Remove label") in stable order
    classes = [c for c in ss.get("all_classes", []) if c != "Remove label"]
    # Guarantee presence
    classes = list(dict.fromkeys(classes))  # de-dupe keep order
    if not classes:
        classes = ["class0", "class1"]
    name_to_idx = {c: i for i, c in enumerate(classes)}
    return classes, name_to_idx


def load_patches_from_session(target=(64, 64)):
    classes, name_to_idx = _build_label_index()
    X, y = [], []
    for patch, cname in _iter_instance_patches():
        X.append(preprocess_patch_for_training(patch, target))
        y.append(name_to_idx.get(cname, None))
    # filter any None labels (shouldn't happen if classes contain all)
    pairs = [(xi, yi) for xi, yi in zip(X, y) if yi is not None]
    if not pairs:
        return (
            np.zeros((0, *target, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            classes,
        )
    Xf = np.stack([p[0] for p in pairs], axis=0)
    yf = np.array([p[1] for p in pairs], dtype=np.int64)
    return Xf, yf, classes


# -------------------------------
#  Keras Sequence with augmentation pipeline
# -------------------------------


class AugSequence(Sequence):
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
            if aug.ndim == 2:
                aug = np.stack([aug] * 3, axis=-1)
            elif aug.ndim == 3 and aug.shape[2] > 3:
                aug = aug[:, :, :3]
            aug = _resize_keep_aspect(aug, target=self.target_size)
            out.append(aug.astype(np.float32))

        Xo = np.stack(out, axis=0)
        return Xo, yb


def load_labeled_patches_from_session(patch_size: int = 64):
    """
    Build (X, y, classes) directly from the app's uploaded images + user labels.
    Uses the same extract_masked_cell_patch + label dicts as classification.
    """
    ims = st.session_state.get("images", {}) or {}
    all_classes = [
        c for c in st.session_state.get("all_classes", []) if c != "Remove label"
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
            if not cname or cname == "Remove label":
                continue
            patch = extract_masked_cell_patch(
                img, (M == iid).astype(np.uint8), size=patch_size
            )
            if patch is None or patch.size == 0:
                continue
            # convert to 3-ch RGB
            if patch.ndim == 2:
                patch = np.repeat(patch[..., None], 3, axis=2)
            elif patch.ndim == 3 and patch.shape[2] == 4:
                patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB)
            elif patch.ndim == 3 and patch.shape[2] == 3:
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch = resize_with_aspect_ratio(patch, patch_size)
            X.append(patch.astype(np.float32))
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


# -------------------------------
#  Model inference on patch
# -------------------------------


def extract_masked_cell_patch(
    image: np.ndarray, mask: np.ndarray, size: int | tuple[int, int] = 64
):
    im, m = np.asarray(image), np.asarray(mask, bool)
    if im.shape[:2] != m.shape:
        raise ValueError("image/mask size mismatch")
    if not m.any():
        return None
    if im.ndim == 3 and im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)

    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop, mc = im[y0:y1, x0:x1], m[y0:y1, x0:x1]
    crop = (crop * mc[..., None] if crop.ndim == 3 else crop * mc).astype(im.dtype)

    tw, th = (size, size) if isinstance(size, int) else map(int, size)
    h, w = crop.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    resized = cv2.resize(
        crop, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    )

    canvas = np.zeros(
        (th, tw) if resized.ndim == 2 else (th, tw, resized.shape[2]), dtype=im.dtype
    )  # black pad
    yx = ((th - nh) // 2, (tw - nw) // 2)
    canvas[yx[0] : yx[0] + nh, yx[1] : yx[1] + nw, ...] = resized
    return canvas


def classify_cells_with_densenet(rec: dict) -> None:
    """Classify segmented cell masks in `rec` using a DenseNet-121 model.
    Mutates `rec` and session_state, then triggers a rerun on success.
    """
    model = ss.get("densenet_model")
    if model is None:
        st.warning("Upload a DenseNet-121 classifier in the sidebar first.")
        return

    M = rec.get("masks")
    if not isinstance(M, np.ndarray) or M.ndim != 2 or not np.any(M):
        st.info("No masks to classify.")
        return

    # Build usable class names (fallback to two defaults)
    all_classes = [c for c in ss.get("all_classes", []) if c != "Remove label"] or [
        "class0",
        "class1",
    ]

    # Gather instance ids and extract 64x64 patches
    ids = [int(v) for v in np.unique(M) if v != 0]
    patches, keep_ids = [], []

    for iid in ids:
        patch = np.asarray(extract_masked_cell_patch(rec["image"], M == iid, size=64))
        if patch.ndim == 2:
            patch = np.repeat(patch[..., None], 3, axis=2)
        elif patch.ndim == 3 and patch.shape[2] == 4:
            patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB)
        elif patch.ndim == 3 and patch.shape[2] == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        patch = resize_with_aspect_ratio(patch, 64)
        patches.append(preprocess_input(patch.astype(np.float32)))
        keep_ids.append(iid)

    if not patches:
        st.info("No valid patches extracted.")
        return

    # Predict classes
    X = np.stack(patches, axis=0)
    preds = model.predict(X, verbose=0).argmax(axis=1)

    # Write back labels, extend class list if needed
    labels = rec.setdefault("labels", {})
    for iid, cls_idx in zip(keep_ids, preds):
        idx = int(cls_idx)
        name = all_classes[idx] if idx < len(all_classes) else str(idx)
        labels[int(iid)] = name
        if name and name != "Remove label" and name not in ss.get("all_classes", []):
            ss.setdefault("all_classes", []).append(name)

    # Persist updated record and rerun UI
    ss.images[ss.current_key] = rec


# -------------------------------
#  Functions for plotting training metrics
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
    plt.close(fig)
