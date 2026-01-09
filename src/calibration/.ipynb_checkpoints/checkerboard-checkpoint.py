import os, glob, json
import cv2
import numpy as np

SUBPIX_WIN = (11, 11)

def make_objp(dims, square_size_mm):
    """Make objp."""
    cols, rows = dims
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm
    return objp

def get_checkerboard_corners(image_bgr, dims):
    """Utility for get checkerboard corners."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
             cv2.CALIB_CB_NORMALIZE_IMAGE)

    ret, corners = cv2.findChessboardCorners(gray, dims, flags)
    if not ret:
        return False, None, gray.shape[::-1]  # (w,h)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, SUBPIX_WIN, (-1, -1), criteria)
    return True, corners, gray.shape[::-1]

def detect_corners(img_bgr, dims):
    """Detect corners."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
             cv2.CALIB_CB_NORMALIZE_IMAGE +
             cv2.CALIB_CB_FAST_CHECK)
    ok, corners = cv2.findChessboardCorners(gray, dims, flags)
    if not ok:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners  # (N,1,2)
