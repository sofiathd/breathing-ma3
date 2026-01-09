import os, glob, json
import cv2
import numpy as np

def corners_to_grid(corners, dims):
    """Convert corners to grid."""
    cols, rows = dims
    pts = corners.reshape(-1, 2)
    # OpenCV gives row-major order (rows of corners, each row has cols points)
    grid = pts.reshape(rows, cols, 2)  # (rows, cols, 2)
    return grid

def grid_to_corners(grid):
    """Convert grid to corners."""
    pts = grid.reshape(-1, 2)
    return pts.reshape(-1, 1, 2).astype(np.float32)

def sym_id(grid):
    """Utility for sym id."""
    return grid

def sym_flip_h(grid):
    # flip horizontally (mirror columns)
    """Utility for sym flip h."""
    return grid[:, ::-1, :]

def sym_flip_v(grid):
    # flip vertically (mirror rows)
    """Utility for sym flip v."""
    return grid[::-1, :, :]

def sym_rot180(grid):
    """Utility for sym rot180."""
    return grid[::-1, ::-1, :]

SYM_FUNCS = [sym_id, sym_flip_h, sym_flip_v, sym_rot180]
SYM_NAMES = ["id", "flip_h", "flip_v", "rot180"]
