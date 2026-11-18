import json
import numpy as np
from enum import Enum

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import os

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

def is_linear_trajectory(tracking_buffer, ratio_threshold=0.75, dist_thresh=10.0, max_iterations=50):
    points = get_pixel_path(tracking_buffer)
    n_points = len(points)
    
    # Insufficient points to evaluate a trajectory trend, so default to "yes"
    if n_points < 3:
        return True

    best_inliers_count = 0
    
    for _ in range(max_iterations):
        idx = np.random.choice(n_points, 2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]
        
        if np.allclose(p1, p2):
            # ignore similar poits
            continue
            
        # Line equation in general form: Ax + By + C = 0
        x1, y1 = p1
        x2, y2 = p2
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        denom = np.sqrt(A**2 + B**2)
        if denom == 0:
            continue
            
        # Compute perpendicular distance from all points to the candidate line
        distances = np.abs(A * points[:, 0] + B * points[:, 1] + C) / denom
        
        inliers_count = np.sum(distances < dist_thresh)
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            
        if best_inliers_count / n_points >= ratio_threshold:
            return True

    inlier_ratio = best_inliers_count / n_points
    return inlier_ratio >= ratio_threshold

def visualize_trajectory(tracking_buffer, camera_params, output_dir="."):
    """ Intended to help visualize whether it's worth it to model acceleration and velocity.
    So far, even with slow puts, there's not much of a visual curvature when plotting
    position against time."""
    pixel_points = []
    timestamp_path = []
    for _, t, p in tracking_buffer:
        if all(p):
            pixel_points.append(p)
            timestamp_path.append(t)
            
    if len(pixel_points) < 3:
        print("Insufficient points for visualization.")
        return None

    pixel_path = np.array(pixel_points, dtype=np.float32)
    t = np.array(timestamp_path, dtype=np.float64)
    t -= t[0]  # Relative time starting at 0

    golfball_height = camera_utils.golfball_perceived_center()
    path_world_coords = camera_utils.map_camera_to_world_coords(pixel_path, camera_params, golfball_height).astype(np.double)
    
    x = path_world_coords[:, 0]
    y = path_world_coords[:, 1]

    # Fit quadratic and linear models for time-series components
    poly_x_quad = np.polyfit(t, x, 2)
    poly_x_lin  = np.polyfit(t, x, 1)
    
    poly_y_quad = np.polyfit(t, y, 2)
    poly_y_lin  = np.polyfit(t, y, 1)

    # Extract velocities and accelerations
    v_x_quad, a_x = poly_x_quad[1], 2 * poly_x_quad[0]
    v_x_lin = poly_x_lin[0]

    v_y_quad, a_y = poly_y_quad[1], 2 * poly_y_quad[0]
    v_y_lin = poly_y_lin[0]
    
    t_dense = np.linspace(t.min(), t.max(), 200)
    x_quad = np.polyval(poly_x_quad, t_dense)
    x_lin  = np.polyval(poly_x_lin, t_dense)
    
    y_quad = np.polyval(poly_y_quad, t_dense)
    y_lin  = np.polyval(poly_y_lin, t_dense)

    # 2. Create 2-Panel Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # X vs Time Subplot
    ax1.scatter(t, x, color='black', label='Observed X Data', zorder=3)
    ax1.plot(t_dense, x_quad, color='red', linewidth=2, 
             label=f'Non-Linear (v₀ = {v_x_quad:.2f} mm/s, a = {a_x:.2f} mm/s²)')
    ax1.plot(t_dense, x_lin, color='blue', linestyle='--', linewidth=1.5, 
             label=f'Linear Model (v = {v_x_lin:.2f} mm/s)')
    
    ax1.set_title('X Coordinate vs. Time')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('World X Position (mm)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='best')

    # Y vs Time Subplot
    ax2.scatter(t, y, color='black', label='Observed Y Data', zorder=3)
    ax2.plot(t_dense, y_quad, color='red', linewidth=2, 
             label=f'Non-Linear (v₀ = {v_y_quad:.2f} mm/s, a = {a_y:.2f} mm/s²)')
    ax2.plot(t_dense, y_lin, color='blue', linestyle='--', linewidth=1.5, 
             label=f'Linear Model (v = {v_y_lin:.2f} mm/s)')
    
    ax2.set_title('Y Coordinate vs. Time')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('World Y Position (mm)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='best')
    plt.tight_layout()

    # Save to disk with datetime stamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"trajectory_spatial_panels_{timestamp_str}.png")
    plt.savefig(filename, dpi=300)
    plt.close("all")
    
    print(f"3-panel diagnostic plot saved to: {filename}")
    return filename
