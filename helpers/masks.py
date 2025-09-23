import numpy as np
from PIL import Image, ImageDraw


def _resize_mask_nearest(mask_u8, out_h, out_w):
    return np.array(
        Image.fromarray(mask_u8).resize((out_w, out_h), resample=Image.NEAREST),
        dtype=np.uint8,
    )


def polygon_to_mask(obj, h, w):
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    pts = []
    for cmd in obj.get("path", []):
        if (
            len(cmd) >= 3
            and isinstance(cmd[1], (int, float))
            and isinstance(cmd[2], (int, float))
        ):
            pts.append((cmd[1], cmd[2]))
    if pts:
        draw.polygon(pts, outline=1, fill=1)
    return np.array(mask_img, dtype=np.uint8)


def toggle_at_point(active, masks_u8, x, y):
    if masks_u8 is None or not masks_u8.size:
        return active
    hits = [i for i in range(masks_u8.shape[0]) if active[i] and masks_u8[i, y, x] > 0]
    if hits:
        active[hits[-1]] = not active[hits[-1]]
    return active


def composite_over(image_u8, masks_u8, active, alpha=0.5):
    """
    Overlay active masks in red + white outline onto image_u8.
    Accepts masks shaped (H,W), (N,H,W), (N,H,W,1/3), or (H,W,1/3).
    Does NOT auto-transpose on square images.
    """
    H, W = image_u8.shape[:2]
    m = np.asarray(masks_u8)

    # Coerce to (N,H,W) WITHOUT any transpose guesses
    if m.ndim == 2:
        m = m[None, ...]
    elif m.ndim == 3:
        # (H,W,1/3) -> drop channel; else assume already (N,H,W)
        if m.shape[-1] in (1, 3) and m.shape[:2] == (H, W):
            m = m[..., 0][None, ...]
    elif m.ndim == 4:
        # (N,H,W,1/3) -> drop channel; (N,1/3,H,W) -> take first channel
        if m.shape[-1] in (1, 3):
            m = m[..., 0]
        elif m.shape[1] in (1, 3):
            m = m[:, 0, ...]
        # else: assume already (N,H,W) after channel handling

    # Final shape guard
    if m.ndim == 2:
        m = m[None, ...]
    if m.shape[-2:] != (H, W):
        m = np.stack(
            [_resize_mask_nearest(mi.astype(np.uint8), H, W) for mi in m], axis=0
        )

    masks = (m > 0).astype(np.uint8)
    N = masks.shape[0]
    if not isinstance(active, (list, tuple)) or len(active) != N:
        active = [True] * N

    out = image_u8.astype(np.float32) / 255.0
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    for i in range(N):
        if not active[i]:
            continue
        mi = (masks[i] > 0).astype(np.float32)  # (H,W)
        a = (mi * alpha)[..., None]  # (H,W,1)
        out = out * (1 - a) + red[None, None, :] * a

        mb = mi.astype(bool)
        interior = (
            mb
            & np.roll(mb, 1, 0)
            & np.roll(mb, -1, 0)
            & np.roll(mb, 1, 1)
            & np.roll(mb, -1, 1)
        )
        edge = mb & ~interior
        out[edge] = 1.0

    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def stack_to_instances(stack, tie_break="largest_first"):
    stack = (np.asarray(stack) > 0).astype(np.uint8)
    if stack.ndim == 2:
        stack = stack[None, ...]
    if stack.size == 0 or stack.shape[0] == 0:
        H, W = stack.shape[-2:] if stack.ndim == 3 else (0, 0)
        return np.zeros((H, W), dtype=np.uint16)

    N, H, W = stack.shape
    if tie_break == "largest_first":
        areas = stack.reshape(N, -1).sum(axis=1)
        order = np.argsort(-areas)
    else:
        order = np.arange(N)

    inst = np.zeros((H, W), dtype=np.uint32)
    for i, k in enumerate(order, start=1):
        m = stack[k].astype(bool)
        inst[m] = i

    if inst.max() > np.iinfo(np.uint16).max:
        inst = np.clip(inst, 0, np.iinfo(np.uint16).max)

    return inst.astype(np.uint16)
