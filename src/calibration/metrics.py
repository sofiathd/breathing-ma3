import os, glob, json
import cv2
import numpy as np

def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist):
    """Compute reprojection error."""
    total_err = 0.0
    n = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        total_err += err
        n += 1
    return total_err / max(n, 1)