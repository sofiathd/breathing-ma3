import os, glob, json
import cv2
import numpy as np

def list_images(folder, ext):
    """Utility for list images."""
    return sorted(glob.glob(os.path.join(folder, ext)))

def basename_no_ext(path):
    """Utility for basename no ext."""
    return os.path.splitext(os.path.basename(path))[0]

def build_synced_filelists(paths, ext):
    """Build synced filelists."""
    per_cam = []
    for p in paths:
        files = list_images(p, ext)
        per_cam.append({basename_no_ext(f): f for f in files})

    common = set(per_cam[0].keys())
    for m in per_cam[1:]:
        common &= set(m.keys())
    common = sorted(common)

    if not common:
        raise RuntimeError("No common basenames found across the 4 camera folders.")

    synced = []
    for c in range(len(paths)):
        synced.append([per_cam[c][k] for k in common])

    print(f"[sync] Found {len(common)} common selected frames across 4 cameras.")
    return synced, common
