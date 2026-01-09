import gc 
from src.models import get_cotracker, get_deeplab_model
from src.cv.image import apply_clahe, crop_pad_resize, clamp_roi_to_frame
from src.cv.roi import detect_ROI, segment_person_deeplab
from src.signals.preprocess import bandpass
from src.signals.rr import estimate_rr_robust
from src.signals.events import extract_breath_amplitude

def clear_torch_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

def estimate_rf_from_cotracker(frame_dir, camera, fps, roi_region="chest", with_pca=False, 
                               rf_hop_s=3.0, rf_win_s=30.0, 
                               lo_hz=0.07, hi_hz=1.0,           
                               save_roi_path=None):
    MAX_INFERENCE_DIM = 480     
    
    paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if not paths: 
        print(f"No PNGs in {frame_dir}")
        return np.nan, np.array([]), np.array([]), np.array([]), np.array([])
    
    first_im = cv2.imread(paths[0], cv2.IMREAD_COLOR)
    if first_im is None: return np.nan, np.array([]), np.array([]), np.array([]), np.array([])
    H, W = first_im.shape[:2]
    
    if roi_region == "segmented":
        person_mask = segment_person_deeplab(first_im)
        ROI = (0, 0, W, H)
    else:
        person_mask = None
        ROI = detect_ROI(paths[0], camera=camera, region=roi_region)
        
    x0, y0, w, h = ROI
    x0, y0, w, h = clamp_roi_to_frame(x0, y0, w, h, W, H)

    if save_roi_path:
        debug_im = first_im.copy()
        if roi_region == "segmented" and person_mask is not None:
            colored_mask = np.zeros_like(debug_im)
            colored_mask[:, :, 1] = 255 
            binary_mask = person_mask > 0
            debug_im[binary_mask] = cv2.addWeighted(debug_im[binary_mask], 0.5, colored_mask[binary_mask], 0.5, 0)
        else:
            cv2.rectangle(debug_im, (x0, y0), (x0+w, y0+h), (0, 255, 0), 2)
        cv2.imwrite(save_roi_path, debug_im)

    scale = 1.0
    if w > MAX_INFERENCE_DIM or h > MAX_INFERENCE_DIM:
        scale = MAX_INFERENCE_DIM / max(w, h)
    T = len(paths)
    new_w = max(2, int(w * scale))
    new_h = max(2, int(h * scale))
    print(f"[CoTracker] T={T}, ROI={w}x{h}, resized={new_w}x{new_h}, scale={scale:.3f}")
    
    video_roi = np.empty((T, new_h, new_w, 3), dtype=np.uint8)
    
    for i, p in enumerate(paths):
        im_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if im_bgr is None:
            video_roi[i] = 0
            continue
    
        roi_bgr = crop_pad_resize(im_bgr, x0, y0, w, h, new_w, new_h)
        roi_bgr = apply_clahe(roi_bgr)
    
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        video_roi[i] = roi_rgb
    
    gc.collect() 

    video_t = torch.from_numpy(video_roi).permute(0,3,1,2)[None].float().to(device) / 255.0
    
    del video_roi 
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cotracker = get_cotracker()
    
    try:
        with torch.inference_mode():
            pred_tracks, pred_vis = cotracker(video_t, grid_size=30)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"!!! GPU OOM in CoTracker for {frame_dir}. Returning NaNs.")
            del video_t
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return np.nan, np.zeros(T), np.zeros(T), np.array([]), np.array([])
        else:
            raise e

    tracks = pred_tracks[0].detach().cpu().numpy() # (T, N, 2)
    vis_np = pred_vis[0].detach().cpu().numpy() # (T, N)
    
    del video_t, pred_tracks, pred_vis
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    if vis_np.ndim == 3: vis = vis_np[..., 0]
    elif vis_np.ndim == 2: vis = vis_np
    else: vis = np.zeros((T, tracks.shape[1]))

    if scale != 1.0:
        tracks = tracks / scale

    N = tracks.shape[1]
    xy0 = tracks[0]
    valid_pts = np.ones(N, dtype=bool)
    
    if roi_region == "segmented" and person_mask is not None:
        mask_roi = person_mask[y0:y0+h, x0:x0+w]
        for n in range(N):
            ix, iy = int(round(xy0[n,0])), int(round(xy0[n,1]))
            ix = max(0, min(ix, w-1))
            iy = max(0, min(iy, h-1))
            if mask_roi[iy, ix] == 0:
                valid_pts[n] = False

    vis_mask = (vis > 0.5)
    vis_mask[:, ~valid_pts] = False

    dx = tracks[:,:,0] - tracks[0,:,0]
    dy = tracks[:,:,1] - tracks[0,:,1]
    
    sig = np.zeros(T)

    if with_pca:
        valid_consistency = np.mean(vis_mask, axis=0) > 0.8
        
        if np.sum(valid_consistency) < 5:
            return np.nan, np.zeros(T), np.zeros(T), np.array([]), np.array([])

        dx_sel = dx[:, valid_consistency]
        dy_sel = dy[:, valid_consistency]
        
        features = np.concatenate([dx_sel, dy_sel], axis=1)
        features_centered = features - np.mean(features, axis=0)
        
        try:
            U, s, Vt = np.linalg.svd(features_centered, full_matrices=False)
            pc1 = U[:, 0] * s[0]
            
            mean_dy = np.mean(dy_sel, axis=1)
            if np.corrcoef(pc1, mean_dy)[0,1] < 0:
                pc1 = -pc1
            sig = pc1
        except Exception as e:
            print(f"PCA SVD Failed: {e}")
            return np.nan, np.zeros(T), np.zeros(T), np.array([]), np.array([])
            
    else:
        for t_idx in range(T):
            m = vis_mask[t_idx]
            if np.any(m):
                sig[t_idx] = np.median(dy[t_idx, m])
            else:
                sig[t_idx] = np.nan

    valid_idx = np.where(~np.isnan(sig))[0]
    if len(valid_idx) < 32: # Need enough data for FFT
         return np.nan, np.zeros(T), np.zeros(T), np.array([]), np.array([])
         
    if len(valid_idx) < T:
        sig = np.interp(np.arange(T), valid_idx, sig[valid_idx])
        
    sig_detrend = detrend(sig)
    sig_bp = bandpass(sig_detrend, fs=fps, lo_hz=lo_hz, hi_hz=hi_hz)
    
    sigma_val = 0.1 * fps
    sig_smooth = gaussian_filter1d(sig_bp, sigma=sigma_val)

    burn_s = 2.0
    burn = int(burn_s * fps)
    
    if len(sig_smooth) <= burn:
        return np.nan, sig_smooth, sig, np.array([]), np.array([])

    sig_smooth_clipped = sig_smooth[burn:]

    rr_video_bpm = estimate_rr_robust(sig_smooth_clipped, fs=fps, lo_hz=lo_hz, hi_hz=hi_hz)
    
    t_amp_rel, amp_proxy = extract_breath_amplitudes(sig_smooth_clipped, fps, hi_hz=hi_hz)
    t_amp = t_amp_rel + burn_s

    return rr_video_bpm, sig_smooth, sig, t_amp, amp_proxy

