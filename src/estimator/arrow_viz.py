from __future__ import annotations

import os
import glob
import gc
from typing import Tuple, Optional, Literal

import cv2
import numpy as np
import torch

from src.models import get_cotracker
from src.cv.image import crop_pad_resize, apply_clahe
from src.cv.roi import detect_ROI, clamp_roi_to_frame, segment_person_deeplab


def _infer_device_from_model(model: torch.nn.Module) -> torch.device:
    """Estimate device from model."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def save_frames_with_breathing_arrows(
    *,
    get_frame_bgr,
    T: int,
    tracks_xy: np.ndarray,
    vis: np.ndarray,
    out_dir: str,
    mode: Literal["from0", "prev"] = "from0",
    vis_thresh: float = 0.5,
    arrow_color_bgr: Tuple[int, int, int] = (0, 255, 0),  # green
    point_color_bgr: Tuple[int, int, int] = (255, 0, 255),  # mauve-ish (magenta)
    point_radius: int = 1,
    arrow_thickness: int = 1,
    tip_length: float = 0.25,
) -> None:
    """Save per-frame PNGs with arrows + points drawn.

    Parameters
    ----------
    get_frame_bgr:
        Callable f(t)->frame in BGR used for drawing (shape HxWx3 or HxW).
    tracks_xy:
        (T,N,2) array of point coordinates in the SAME coordinate system as get_frame_bgr.
    vis:
        (T,N) visibilities in [0,1]
    """
    os.makedirs(out_dir, exist_ok=True)

    T2, N = tracks_xy.shape[:2]
    assert T2 == T, f"tracks T={T2} mismatch with expected T={T}"

    ref_xy = tracks_xy[0].copy()

    for t in range(T):
        frame = get_frame_bgr(t)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame = frame.copy()

        H, W = frame.shape[:2]

        if mode == "prev" and t != 0:
            ref_xy = tracks_xy[t - 1]

        vmask = vis[t] > vis_thresh
        pts_now = tracks_xy[t]

        for n in range(N):
            if not vmask[n]:
                continue

            x_ref, y_ref = ref_xy[n]
            x_now, y_now = pts_now[n]

            p0 = (int(round(x_ref)), int(round(y_ref)))
            p1 = (int(round(x_now)), int(round(y_now)))

            if not (0 <= p0[0] < W and 0 <= p0[1] < H):
                continue

            p1 = (int(np.clip(p1[0], 0, W - 1)), int(np.clip(p1[1], 0, H - 1)))

            cv2.arrowedLine(frame, p0, p1, arrow_color_bgr, arrow_thickness, tipLength=tip_length)
            cv2.circle(frame, p1, point_radius, point_color_bgr, -1)

        cv2.imwrite(os.path.join(out_dir, f"frame_{t:05d}.png"), frame)


def export_arrow_viz_official_pipeline(
    *,
    frame_dir: str,
    camera: str,
    fps: float,
    roi_region: str,
    out_dir: str,
    mode: Literal["from0", "prev"] = "from0",
    vis_thresh: float = 0.5,
    max_inference_dim: int = 480,
    grid_size: int = 30,
    render_on: Literal["orig", "resized"] = "orig",
    apply_clahe_to_render: bool = True,
    arrow_color_bgr: Tuple[int, int, int] = (0, 255, 0),
    point_color_bgr: Tuple[int, int, int] = (255, 0, 255),
    point_radius: int = 1,
    arrow_thickness: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run CoTracker the same way as the main pipeline, then export arrow images.

    Returns
    -------
    tracks_orig:
        (T,N,2) tracks in ORIGINAL ROI coordinates (w,h).
    vis:
        (T,N) visibilities.
    """
    paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in {frame_dir}")

    # Read first frame to get size + optional person mask
    first = cv2.imread(paths[0], cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not read first frame: {paths[0]}")
    H0, W0 = first.shape[:2]

    if roi_region.lower() == "segmented":
        person_mask = segment_person_deeplab(first)
        x0, y0, w, h = (0, 0, W0, H0)
    else:
        person_mask = None
        x0, y0, w, h = detect_ROI(camera=camera, roi_region=roi_region)

    x0, y0, w, h = clamp_roi_to_frame(x0, y0, w, h, W0, H0)

    # Resize factor for CoTracker inference
    scale = min(1.0, float(max_inference_dim) / float(max(w, h)))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    print(f"[ArrowViz] T={len(paths)}, ROI={w}x{h}, resized={new_w}x{new_h}, scale={scale:.3f}, render_on={render_on}")

    # Build resized ROI video (what CoTracker will see)
    T = len(paths)
    video_roi_bgr = np.zeros((T, new_h, new_w, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        im_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if im_bgr is None:
            continue
        roi_bgr = crop_pad_resize(im_bgr, x0, y0, w, h, new_w, new_h)
        roi_bgr = apply_clahe(roi_bgr)
        video_roi_bgr[i] = roi_bgr

    # Run CoTracker
    video_roi_rgb = video_roi_bgr[..., ::-1].copy()
    video_t = torch.from_numpy(video_roi_rgb).permute(0, 3, 1, 2)[None].float() / 255.0  # (1,T,3,H,W)

    cotracker = get_cotracker()
    model_device = _infer_device_from_model(cotracker)
    video_t = video_t.to(model_device)

    with torch.inference_mode():
        pred_tracks, pred_vis = cotracker(video_t, grid_size=grid_size)

    tracks_resized = pred_tracks[0].detach().cpu().numpy()  # (T,N,2) in resized coords
    vis_np = pred_vis[0].detach().cpu().numpy()
    if vis_np.ndim == 3:
        vis = vis_np[..., 0]
    else:
        vis = vis_np
    vis = np.clip(vis, 0.0, 1.0)

    # free tensors
    del video_t, pred_tracks, pred_vis
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Convert tracks to original ROI coords (w,h)
    tracks_orig = tracks_resized / max(scale, 1e-12)

    # Apply segmented filtering (based on frame-0 positions in original ROI coords)
    if roi_region.lower() == "segmented" and person_mask is not None:
        mask_roi = person_mask[y0 : y0 + h, x0 : x0 + w]
        xy0 = tracks_orig[0]
        N = xy0.shape[0]
        valid = np.ones(N, dtype=bool)
        for n in range(N):
            ix = int(round(xy0[n, 0]))
            iy = int(round(xy0[n, 1]))
            ix = max(0, min(ix, w - 1))
            iy = max(0, min(iy, h - 1))
            if mask_roi[iy, ix] == 0:
                valid[n] = False
        vis[:, ~valid] = 0.0

    # Prepare drawing coordinate system
    if render_on == "resized":
        tracks_draw = tracks_resized
        def get_frame_bgr(t: int) -> np.ndarray:
            """Utility for get frame bgr."""
            return video_roi_bgr[t]
    else:
        tracks_draw = tracks_orig
        def get_frame_bgr(t: int) -> np.ndarray:
            """Utility for get frame bgr."""
            im_bgr = cv2.imread(paths[t], cv2.IMREAD_COLOR)
            if im_bgr is None:
                return np.zeros((h, w, 3), dtype=np.uint8)
            crop = im_bgr[y0 : y0 + h, x0 : x0 + w].copy()
            if apply_clahe_to_render:
                crop = apply_clahe(crop)
            return crop

    out_dir_mode = os.path.join(out_dir, f"arrows_{mode}_{render_on}")
    save_frames_with_breathing_arrows(
        get_frame_bgr=get_frame_bgr,
        T=T,
        tracks_xy=tracks_draw,
        vis=vis,
        out_dir=out_dir_mode,
        mode=mode,
        vis_thresh=vis_thresh,
        arrow_color_bgr=arrow_color_bgr,
        point_color_bgr=point_color_bgr,
        point_radius=point_radius,
        arrow_thickness=arrow_thickness,
    )

    # Also save a quick preview
    os.makedirs(out_dir, exist_ok=True)
    preview = get_frame_bgr(0)
    cv2.imwrite(os.path.join(out_dir, f"roi_preview_{render_on}.png"), preview)

    print(f"[ArrowViz] Saved {T} frames to {out_dir_mode}")
    return tracks_orig, vis
