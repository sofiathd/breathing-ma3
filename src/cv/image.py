import numpy as np
import cv2

def apply_clahe(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def clamp_roi_to_frame(x0, y0, w, h, W, H):
    x0 = int(max(0, min(x0, W - 1)))
    y0 = int(max(0, min(y0, H - 1)))
    w = int(max(1, min(w, W - x0)))
    h = int(max(1, min(h, H - y0)))
    return x0, y0, w, h

def crop_pad_resize(im_bgr, x0, y0, w, h, out_w, out_h):
    H, W = im_bgr.shape[:2]
    x1 = min(W, x0 + w)
    y1 = min(H, y0 + h)

    crop = im_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        crop = np.zeros((h, w, 3), dtype=np.uint8)

    ch, cw = crop.shape[:2]
    if ch < h or cw < w:
        pad_bottom = max(0, h - ch)
        pad_right  = max(0, w - cw)
        crop = cv2.copyMakeBorder(
            crop, 0, pad_bottom, 0, pad_right,
            borderType=cv2.BORDER_REPLICATE
        )

    crop = crop[:h, :w]
    crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return crop

