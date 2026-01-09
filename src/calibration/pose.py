import os, glob, json
import cv2
import numpy as np

def solve_board_pose(objp, corners, K, dist):
    """Estimate checkerboard pose (R,t) in a camera frame via PnP."""
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3, 1)

def rel_from_board(R0, t0, Rc, tc):
    # camera-to-camera: cam0 -> camc
    # If board pose in cam is: X_cam = R * X_obj + t
    # then camc relative to cam0:
    """Compute relative camera transform from two checkerboard poses."""
    R_c0 = Rc @ R0.T
    t_c0 = tc - R_c0 @ t0
    return R_c0, t_c0

def angle_deg(Ra, Rb):
    """Compute the angular distance (degrees) between two rotations."""
    R = Rb @ Ra.T
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))

def rotmat_to_quat(R):
    """Convert a rotation matrix to a unit quaternion (wxyz)."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q

def quat_to_rotmat(q):
    """Convert a unit quaternion (wxyz) to a rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)

def average_rotations_quat(R_list):
    """Average multiple rotations using quaternion averaging."""
    q0 = rotmat_to_quat(R_list[0])
    qs = []
    for R in R_list:
        q = rotmat_to_quat(R)
        if np.dot(q, q0) < 0:
            q = -q
        qs.append(q)
    q_avg = np.mean(qs, axis=0)
    q_avg /= np.linalg.norm(q_avg) + 1e-12
    return quat_to_rotmat(q_avg)

