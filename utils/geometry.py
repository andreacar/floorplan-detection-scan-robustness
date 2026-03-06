
def clamp_bbox_xywh(bbox, img_w: float, img_h: float):
    x, y, w, h = bbox
    x1 = max(0.0, float(x))
    y1 = max(0.0, float(y))
    x2 = min(float(x) + float(w), float(img_w))
    y2 = min(float(y) + float(h), float(img_h))
    new_w = x2 - x1
    new_h = y2 - y1
    if new_w <= 0 or new_h <= 0:
        return None
    return x1, y1, new_w, new_h

def compute_iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    iw = max(ix2 - ix1, 0)
    ih = max(iy2 - iy1, 0)

    inter = iw * ih
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter

    return inter / union if union > 0 else 0.0
