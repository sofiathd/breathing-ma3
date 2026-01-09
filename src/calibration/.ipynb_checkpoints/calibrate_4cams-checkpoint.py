import os, glob, json
import cv2
import numpy as np

CHECKERBOARD_DIMS = (10, 7)  
SQUARE_SIZE_MM = 24.0
FILE_EXT = "*.png"

PATHS = [
    "../data/intermediate/cameras/calibration/selected_40/olympus", 
    "../data/intermediate/cameras/calibration/selected_40/phone", 
    "../data/intermediate/cameras/calibration/selected_40/flir",     
    "../data/intermediate/cameras/calibration/selected_40/gray",    
]

CAM_NAMES = ["olympus", "phone", "flir", "gray"] # [cam0, cam1, cam2, cam3]

SUBPIX_WIN = (11, 11)
VIS_THRESH = 0.5

POSE_ANG_THR_DEG = 6.0      
POSE_T_THR_MM = 200.0       

OUT_JSON = "calib_4cams_selected40_poseconsistent.json"

def main():
    """Utility for main."""
    synced_files_per_cam, keys = build_synced_filelists(PATHS, FILE_EXT)
    objp = make_objp(CHECKERBOARD_DIMS, SQUARE_SIZE_MM)

    all_corners = [[] for _ in range(4)]
    sizes = [None] * 4
    valid_idx = []

    for i in range(len(keys)):
        corners_i = []
        ok = True
        for c in range(4):
            img = cv2.imread(synced_files_per_cam[c][i])
            if img is None:
                ok = False
                break
            ret, corners, size = get_checkerboard_corners(img, CHECKERBOARD_DIMS)
            if not ret:
                ok = False
                break
            corners_i.append(corners)
            sizes[c] = size
        if ok:
            for c in range(4):
                all_corners[c].append(corners_i[c])
            valid_idx.append(i)

    print(f"[detect] Valid selected frames with corners in ALL 4 cams: {len(valid_idx)} / {len(keys)}")
    if len(valid_idx) < 10:
        raise RuntimeError("Too few valid frames across all 4 cameras.")

    objpoints = [objp for _ in range(len(valid_idx))]

    intr = []
    print("\n--- Calibrating intrinsics (per camera) ---")
    for c in range(4):
        imgpoints = [all_corners[c][k] for k in range(len(valid_idx))]
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, sizes[c], None, None
        )
        reproj = compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist)
        print(f"cam{c} ({CAM_NAMES[c]}): RMS={ret:.4f} | mean_reproj={reproj:.4f} px | size={sizes[c]}")
        intr.append({"K": K, "dist": dist})

    print("\n--- Symmetry selection by pose-consistency (cam0 reference) ---")
    reordered = [ [all_corners[0][k] for k in range(len(valid_idx))] ]  # cam0 unchanged
    sym_stats = {c: np.zeros(len(SYM_FUNCS), dtype=int) for c in [1,2,3]}

    K0, d0 = intr[0]["K"], intr[0]["dist"]

    for c in [1,2,3]:
        Kc, dc = intr[c]["K"], intr[c]["dist"]
        chosen = []

        Rrel_running = []
        trel_running = []

        for k in range(len(valid_idx)):
            corners0 = reordered[0][k]
            grid_c = corners_to_grid(all_corners[c][k], CHECKERBOARD_DIMS)

            # pose cam0
            R0, t0 = solve_board_pose(objp, corners0, K0, d0)
            if R0 is None:
                chosen.append(all_corners[c][k])
                continue

            best_cost = 1e18
            best_corners = None
            best_sym = 0
            best_Rrel = None
            best_trel = None

            # running reference transform (robust-ish)
            if len(Rrel_running) >= 5:
                R_ref = average_rotations_quat(Rrel_running[-10:])
                t_ref = np.median(np.hstack(trel_running[-10:]), axis=1).reshape(3,1)
            else:
                R_ref = None
                t_ref = None

            for s_idx, fn in enumerate(SYM_FUNCS):
                corners_sym = grid_to_corners(fn(grid_c))

                Rc, tc = solve_board_pose(objp, corners_sym, Kc, dc)
                if Rc is None:
                    continue

                Rrel, trel = rel_from_board(R0, t0, Rc, tc)

                # cost = angular deviation + scaled translation deviation
                if R_ref is None:
                    # early frames: just prefer smaller translation magnitude (weak prior)
                    cost = float(np.linalg.norm(trel))
                else:
                    ang = angle_deg(R_ref, Rrel)
                    dt = float(np.linalg.norm(trel - t_ref))
                    cost = ang + (dt / 50.0)  # 50mm ~= 1deg weight-ish

                if cost < best_cost:
                    best_cost = cost
                    best_corners = corners_sym
                    best_sym = s_idx
                    best_Rrel = Rrel
                    best_trel = trel

            if best_corners is None:
                best_corners = all_corners[c][k]

            chosen.append(best_corners)
            sym_stats[c][best_sym] += 1

            if best_Rrel is not None:
                Rrel_running.append(best_Rrel)
                trel_running.append(best_trel)

        reordered.append(chosen)

        print(f"cam{c} ({CAM_NAMES[c]}): sym_counts = {sym_stats[c].tolist()}  (order: {SYM_NAMES})")

    # reordered list is [cam0, cam1, cam2, cam3] BUT we appended in loop,
    # so it is currently length 4 with correct order.

    # 4) pose-based outlier rejection per pair (cam0, camc)
    print("\n--- Outlier rejection by relative pose consistency ---")
    keep = set(range(len(valid_idx)))

    # compute robust reference per cam, prune frames
    for c in [1,2,3]:
        Kc, dc = intr[c]["K"], intr[c]["dist"]

        Rrels, trels = [], []
        for k in range(len(valid_idx)):
            R0, t0 = solve_board_pose(objp, reordered[0][k], K0, d0)
            Rc, tc = solve_board_pose(objp, reordered[c][k], Kc, dc)
            if R0 is None or Rc is None:
                continue
            Rrel, trel = rel_from_board(R0, t0, Rc, tc)
            Rrels.append(Rrel)
            trels.append(trel)

        R_med = average_rotations_quat(Rrels)
        t_med = np.median(np.hstack(trels), axis=1).reshape(3,1)

        bad = set()
        for k in range(len(valid_idx)):
            R0, t0 = solve_board_pose(objp, reordered[0][k], K0, d0)
            Rc, tc = solve_board_pose(objp, reordered[c][k], Kc, dc)
            if R0 is None or Rc is None:
                bad.add(k); continue
            Rrel, trel = rel_from_board(R0, t0, Rc, tc)
            ang = angle_deg(R_med, Rrel)
            dt = float(np.linalg.norm(trel - t_med))
            if ang > POSE_ANG_THR_DEG or dt > POSE_T_THR_MM:
                bad.add(k)

        before = len(keep)
        keep -= bad
        print(f"cam{c} ({CAM_NAMES[c]}): removed {len(bad)} frames by pose filter | keep now {len(keep)}")

    keep = sorted(list(keep))
    print(f"Keeping {len(keep)}/{len(valid_idx)} frames after all pairwise pose filters.")
    if len(keep) < 10:
        print("WARNING: very few frames remain. Thresholds may be too strict or data is inconsistent.")

    # 5) stereoCalibrate cam0 with each cam (FIX_INTRINSIC)
    print("\n--- Estimating extrinsics relative to cam0 using stereoCalibrate (FIX_INTRINSIC) ---")
    extrinsics = []
    extrinsics.append({"R": np.eye(3), "T": np.zeros((3,1))})

    for c in [1,2,3]:
        imgp0 = [reordered[0][k] for k in keep]
        imgpc = [reordered[c][k] for k in keep]
        objp_list = [objp for _ in keep]

        Kc, dc = intr[c]["K"], intr[c]["dist"]

        rms, K0o, d0o, Kco, dco, R, T, E, F = cv2.stereoCalibrate(
            objp_list, imgp0, imgpc,
            K0, d0, Kc, dc,
            sizes[0],
            flags=cv2.CALIB_FIX_INTRINSIC
        )

        print(f"cam{c} ({CAM_NAMES[c]}): stereo_RMS={rms:.3f} px | |T|={float(np.linalg.norm(T)):.2f} mm")
        extrinsics.append({"R": R, "T": T})

    # 6) save JSON
    out = {
        "checkerboard": {"dims": list(CHECKERBOARD_DIMS), "square_size_mm": float(SQUARE_SIZE_MM)},
        "camera_paths": PATHS,
        "num_valid_frames_all4": int(len(valid_idx)),
        "num_frames_used_final": int(len(keep)),
        "intrinsics": [],
        "extrinsics_cam0_reference": []
    }

    for c in range(4):
        K = intr[c]["K"]
        dist = intr[c]["dist"].reshape(-1)
        out["intrinsics"].append({
            "cam_index": c,
            "cam_name": CAM_NAMES[c],
            "image_size": list(sizes[c]),
            "K": K.tolist(),
            "dist": dist.tolist(),
        })

    for c in range(4):
        R = extrinsics[c]["R"]
        T = extrinsics[c]["T"]
        Tw = np.eye(4)
        Tw[:3,:3] = R
        Tw[:3, 3:4] = T
        out["extrinsics_cam0_reference"].append({
            "cam_index": c,
            "cam_name": CAM_NAMES[c],
            "R_cam0_to_cam": R.tolist(),
            "t_cam0_to_cam_mm": T.reshape(-1).tolist(),
            "T_cam0_to_cam_4x4": Tw.tolist()
        })

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved calibration to: {OUT_JSON}")


if __name__ == "__main__":
    main()
