import json
import numpy as np
from enum import Enum

from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, RANSACRegressor

import camera_utils
import golf_utils

def get_pixel_path(tracking_buffer):
    path = []
    for i in range(len(tracking_buffer)):
        _, _, p = tracking_buffer[i]
        if all(p):
            path.append(p)
    return np.array(path, np.float32)

class PathResult(Enum):
    PULLED = 1
    PUSHED = 2

def score_direction(pv, cbv1, cbv2):
    # The chessboard might be oriented in four directions, in two pairs perpendicular
    # to each other, and two pairs opposite each other.
    # As a result, when scoring the detected path, we need do the scoring
    # in the two potential orientations, and take the average score.
    d1 = np.rad2deg(np.arccos(np.dot(pv, cbv1))) % 90.0
    d1 = min(90 - d1, d1)
    d2 = np.rad2deg(np.arccos(np.dot(pv, cbv2))) % 90.0
    d2 = min(90 - d2, d2)
    return np.mean([d1, d2])

def evaluate_path(target_line, pixel_path, camera_params):
    vx, vy, vz, x0, y0, z0 = camera_utils.fit_tracked_path(pixel_path,
                                                           camera_params)

    pv = np.array([vx, vy, vz], np.float32).flatten()
    pv = pv / np.linalg.norm(pv)

    # The fitted path doesn't yet have a reliable direction, owing to the underlying
    # opencv implementaiton.
    #
    # But we can make this directional by comparing to the vector of the first and last pixel
    world_coords_path = camera_utils.map_camera_to_world_coords(pixel_path,
                                                                camera_params,
                                                                camera_utils.golfball_perceived_center())
    path_direction = world_coords_path[-1] - world_coords_path[0]
    print("path direction", path_direction)
    if np.dot(path_direction, pv) < 0:
        pv = -pv

    a, b, _ = target_line
    tv = np.array([-b, a, 0.0], dtype=np.float32)
    tv = tv / np.linalg.norm(tv)

    # Make sure this vector has the same direction as the fitted line.
    if np.dot(pv, tv) < 0:
        tv = -tv
    
    path_result = "PUSHED"
    print("cross arguments", pv, tv, np.cross(pv, tv))
    if np.cross(pv, tv)[2] > 0:
        path_result = "PULLED"
    return np.degrees(np.arccos(np.dot(pv, tv))), path_result

def evaluate_distance(tracking_buffer, camera_params, residual_threshold=0.05, max_trials=100):
    path_world_coords = camera_utils.map_camera_to_world_coords(
        get_pixel_path(tracking_buffer),
        camera_params,
        z=camera_utils.golfball_perceived_center()
    ).astype(np.double)

    timestamp_path = []
    for i in range(len(tracking_buffer)):
        _, t, p = tracking_buffer[i]
        if all(p):
            timestamp_path.append(t)

    timestamp_path = np.array(timestamp_path, dtype=np.float64)

    # Safety check for minimum data points and alignment
    if len(timestamp_path) < 3 or len(path_world_coords) != len(timestamp_path):
        return 0.0

    t = timestamp_path - timestamp_path[0]  # Relative time (starts at 0)
    x = path_world_coords[:, 0]
    y = path_world_coords[:, 1]
    n = len(t)

    # 1. Sort indices chronologically and split into early and late halves
    sorted_time_indices = np.argsort(t)
    mid = n // 2
    early_indices = sorted_time_indices[:mid]
    late_indices = sorted_time_indices[mid:]

    if len(early_indices) == 0 or len(late_indices) == 0:
        return 0.0

    best_inlier_count = -1
    best_residual_score = float('inf')
    best_vx, best_vy = 0.0, 0.0

    # 2. RANSAC loop with stratified sampling and tie-breaking
    for _ in range(max_trials):
        # Force a large time gap by pulling one point from each half
        i = np.random.choice(early_indices)
        j = np.random.choice(late_indices)
        
        dt = t[j] - t[i]
        if dt <= 0:
            continue

        # Initial velocity hypothesis from the wide-interval pair
        vx = (x[j] - x[i]) / dt
        vy = (y[j] - y[i]) / dt

        x0 = x[i] - vx * t[i]
        y0 = y[i] - vy * t[i]

        # Evaluate residuals across all points
        pred_x = vx * t + x0
        pred_y = vy * t + y0
        residuals = np.sqrt((x - pred_x)**2 + (y - pred_y)**2)
        
        inliers = residuals < residual_threshold
        inlier_count = np.sum(inliers)

        if inlier_count >= 2:
            inlier_residual_sum = np.sum(residuals[inliers] ** 2)

            # Tie-breaker: Maximize inlier count first, then minimize residual sum
            if (inlier_count > best_inlier_count) or \
               (inlier_count == best_inlier_count and inlier_residual_sum < best_residual_score):
                
                best_inlier_count = inlier_count
                best_residual_score = inlier_residual_sum
                
                # Least-squares refinement over consensus inliers
                if np.var(t[inliers]) > 1e-6:
                    vx_ref, _ = np.polyfit(t[inliers], x[inliers], 1)
                    vy_ref, _ = np.polyfit(t[inliers], y[inliers], 1)
                    best_vx, best_vy = vx_ref, vy_ref
                else:
                    best_vx, best_vy = vx, vy

    # 3. Compute final speed and convert via existing utility
    speed_world = np.sqrt(best_vx**2 + best_vy**2)
    return golf_utils.velocity_to_distance(speed_world)
