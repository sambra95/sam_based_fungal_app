# panels/train_densenet.py
import random
from dataclasses import dataclass

import numpy as np
import streamlit as st
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
from skimage.exposure import rescale_intensity

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ---- bring in existing app helpers ----
from helpers.state_ops import ordered_keys, current
from helpers.classifying_functions import extract_masked_cell_patch

import cv2

# --- Drop-in replacements for the 'augment' package ---


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


ss = st.session_state

# -------------------------------
#  Augmentation pipeline (UNCHANGED)
# -------------------------------


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
#  Keras Sequence with your augmentation pipeline
# -------------------------------


class AugSequence(tf.keras.utils.Sequence):
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

    # ✅ new property
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


def resize_with_aspect_ratio(img, target_size=64):
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
#  Panel UI
# -------------------------------


def render_train_panel(key_ns: str = "train_densenet"):
    st.header("Train DenseNet on labeled cell patches")
    if not ordered_keys():
        st.info("Upload data and add labels in the other panels first.")
        return

    with st.expander("Training options", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        input_size = c1.selectbox("Input size", options=[64, 96, 128], index=0)
        batch_size = c2.selectbox("Batch size", options=[8, 16, 32, 64], index=2)
        base_trainable = c3.checkbox("Fine-tune base (unfreeze)", value=False)
        epochs = c4.number_input(
            "Max epochs",
            min_value=1,
            max_value=500,
            value=100,
            step=5,
            key="max_epoch_densenet",
        )
        val_split = st.slider("Validation split", 0.05, 0.4, 0.2, 0.05)

    # Load patches from session
    X, y, classes = load_labeled_patches_from_session(patch_size=input_size)
    if X.shape[0] < 2 or len(np.unique(y)) < 2:
        st.warning("Need at least 2 samples and 2 classes. Add more labeled cells.")
        return

    st.write(f"Found **{X.shape[0]}** patches across **{len(classes)}** classes.")
    st.caption(f"Classes: {classes}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, stratify=y, random_state=42
    )

    # Generators
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
        num_transforms=1,
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
        base_trainable=base_trainable,
    )

    # Train
    go = st.button("Start training", use_container_width=True, type="primary")
    if not go:
        return

    es = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )

    with st.spinner("Training DenseNet…"):
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=int(epochs),
            callbacks=[es],
            class_weight=class_weights_dict,
            verbose=0,
        )

    # Evaluate on validation set
    # Collect all val batches
    Xv, yv = [], []
    for i in range(len(val_gen)):
        xb, yb = val_gen[i]
        Xv.append(xb)
        yv.append(yb)
    Xv = np.concatenate(Xv, axis=0)
    yv = np.concatenate(yv, axis=0)

    y_probs = model.predict(Xv, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    acc = accuracy_score(yv, y_pred)
    prec = precision_score(yv, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(yv, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(yv, y_pred, labels=np.arange(len(classes)))

    st.success(
        f"Val accuracy: **{acc:.3f}** · precision: **{prec:.3f}** · F1: **{f1:.3f}**"
    )
    st.write("Confusion matrix (rows: true, cols: pred):")
    st.dataframe(cm, use_container_width=True)

    # Keep model in session (per your requirement)
    ss["densenet_ckpt_bytes"] = model
    ss["densenet_ckpt_name"] = "densenet_finetuned"
    # Also expose a simple predictor for the rest of the app
    ss["densenet_model"] = model

    st.info(
        "Model stored in session. You can use it immediately from the **Classify cells** panel."
    )


# panels/train_cellpose_panel.py

import streamlit as st
import numpy as np
import io as IO
import torch
from sklearn.model_selection import train_test_split
from cellpose import core, io, models, train

from helpers.state_ops import ordered_keys
from helpers.cellpose_functions import _plot_losses, compare_models_mean_iou_plot


def _normalize_for_cellpose(image: np.ndarray) -> np.ndarray:
    im = image.astype(np.float32)
    mean_val = float(np.mean(im)) if im.size else 0.0
    if mean_val <= 0:
        rng = float(im.max() - im.min())
        return (im - im.min()) / rng if rng > 0 else im * 0.0
    return im * (0.5 / mean_val)


def finetune_cellpose_from_records(
    recs: dict,
    base_model: str,
    epochs=100,
    learning_rate=0.00005,
    weight_decay=0.1,
    nimg_per_epoch=32,
):
    images = [_normalize_for_cellpose(recs[k]["image"]) for k in recs.keys()]
    masks = [recs[k]["masks"].astype("uint16") for k in recs.keys()]

    train_images, test_images, train_masks, test_masks = train_test_split(
        images, masks, test_size=0.2, random_state=42, shuffle=True
    )

    st.info(
        f"Training on **{len(train_images)} images** "
        f"(+ {len(test_images)} validation images)."
    )

    use_gpu = core.use_gpu()
    _ = io.logger_setup()

    init_model = None if base_model == "scratch" else base_model
    cell_model = models.CellposeModel(gpu=use_gpu, model_type=init_model)
    model_name = f"{base_model}_finetuned.pt"

    with st.spinner("Fine-tuning Cellpose…"):
        new_path, train_losses, test_losses = train.train_seg(
            cell_model.net,
            train_data=train_images,
            train_labels=train_masks,
            test_data=test_images,
            test_labels=test_masks,
            channels=[0, 0],
            n_epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            SGD=True,
            nimg_per_epoch=nimg_per_epoch,
            model_name=model_name,
            save_path=None,
        )

    # stash in session
    buf = IO.BytesIO()
    torch.save(cell_model.net.state_dict(), buf)
    st.session_state["cellpose_model_bytes"] = buf.getvalue()
    st.session_state["cellpose_model_name"] = model_name
    st.session_state["model_to_fine_tune"] = base_model

    return train_losses, test_losses, model_name


def render_cellpose_train_panel(key_ns="train_cellpose"):
    st.header("Fine-tune Cellpose on your labeled data")

    if not ordered_keys():
        st.info("Upload data and label masks first.")
        return

    with st.expander("Training options", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        base_model = c1.selectbox(
            "Base model",
            options=["cyto2", "cyto", "nuclei", "scratch"],
            index=0,
        )
        epochs = c2.number_input(
            "Max epochs", 1, 500, 100, step=5, key="max_epcoh_cellpose"
        )
        lr = c3.number_input(
            "Learning rate", min_value=1e-6, max_value=1e-2, value=5e-5, format="%.5f"
        )
        wd = c4.number_input(
            "Weight decay", min_value=0.0, max_value=1.0, value=0.1, step=0.05
        )

        nimg = st.slider("Images per epoch", 1, 128, 32, 1)

    go = st.button("Start fine-tuning", use_container_width=True, type="primary")
    if not go:
        return

    recs = {k: st.session_state["images"][k] for k in ordered_keys()}

    train_losses, test_losses, model_name = finetune_cellpose_from_records(
        recs,
        base_model=base_model,
        epochs=int(epochs),
        learning_rate=lr,
        weight_decay=wd,
        nimg_per_epoch=int(nimg),
    )

    st.success(f"Fine-tuning complete ✅ (model: {model_name})")

    st.session_state["train_losses"] = train_losses
    st.session_state["test_losses"] = test_losses

    _plot_losses(train_losses, test_losses)

    masks = [rec["masks"] for rec in recs.values()]
    compare_models_mean_iou_plot(
        [rec["image"] for rec in recs.values()],
        masks,
        base_model_name=base_model if base_model != "scratch" else "cyto2",
    )
