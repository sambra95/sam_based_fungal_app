import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict, Any


def is_unique_box(box, boxes):
    x0, y0, x1, y1 = box
    for b in boxes:
        if b == (x0, y0, x1, y1):
            return False
    return True


def boxes_to_fabric_rects(boxes, scale=1.0) -> Dict[str, Any]:
    rects = []
    for x0, y0, x1, y1 in boxes:
        rects.append(
            {
                "type": "rect",
                "left": x0 * scale,
                "top": y0 * scale,
                "width": (x1 - x0) * scale,
                "height": (y1 - y0) * scale,
                "fill": "rgba(0, 0, 255, 0.25)",
                "stroke": "white",
                "strokeWidth": 2,
                "selectable": False,
                "evented": False,
                "hasControls": False,
                "lockMovementX": True,
                "lockMovementY": True,
                "hoverCursor": "crosshair",
            }
        )
    return {"objects": rects}


def draw_boxes_overlay(image_u8, boxes, alpha=0.25, outline_px=2):
    base = Image.fromarray(image_u8).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    fill = (0, 0, 255, int(alpha * 255))
    for x0, y0, x1, y1 in boxes:
        d.rectangle(
            [x0, y0, x1, y1], fill=fill, outline=(255, 255, 255, 255), width=outline_px
        )
    return np.array(Image.alpha_composite(base, overlay).convert("RGB"), dtype=np.uint8)
