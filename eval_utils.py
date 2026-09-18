import json
import numpy as np

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

def score_direction(pv, cbv1, cbv2):
    d1 = np.rad2deg(np.arccos(np.dot(pv, cbv1))) % 90.0
    d1 = min(90 - d1, d1)
    d2 = np.rad2deg(np.arccos(np.dot(pv, cbv2))) % 90.0
    d2 = min(90 - d2, d2)
    return np.mean([d1, d2])

def evaluate_path(cb_corners, pixel_path, camera_params):
    vx, vy, vz, x0, y0, z0 = camera_utils.fit_tracked_path(pixel_path,
                                                           camera_params)

    pv = np.array([vx, vy, vz], np.float32).flatten()
    pv = pv / np.linalg.norm(pv)

    # Chessboard corners are on the ground, so z=0.0 in these calculations.
    cb_world_coords = camera_utils.map_camera_to_world_coords(cb_corners,
                                                              camera_params,                                                 
                                                              z=0.0)
    
    # The chessboard might be oriented in four directions, in two pairs perpendicular
    # to each other, and two pairs opposite each other.
    # As a result, when scoring the detected path, we need do the scoring
    # in the two potential orientations, and take the average score.
    cbv1 = (cb_world_coords[21] - cb_world_coords[27]).flatten()
    cbv2 = (cb_world_coords[4] - cb_world_coords[32]).flatten()
    
    cbv1 = cbv1 / np.linalg.norm(cbv1)
    cbv2 = cbv2 / np.linalg.norm(cbv2)

    return score_direction(pv, cbv1, cbv2)

def evaluate_distance(tracking_buffer, camera_params):
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
    timestamp_path = np.array(timestamp_path, np.double)
    distances_traveled = np.linalg.norm(np.diff(path_world_coords, axis=0), axis=1)
    time_taken = np.diff(timestamp_path)
    velocities = distances_traveled / time_taken

    # To reduce noise, removing the largest and smallest velocities, and taking the
    # average of what remains.
    velocity = np.mean(np.sort(velocities)[1:-1])
    return golf_utils.velocity_to_distance(velocity)

def evaluate_distance_v2(tracking_buffer, camera_params):
    # 1. Filter out invalid/missing detection frames up front
    valid_indices = [
        i for i, (_, t, p) in enumerate(tracking_buffer) 
        if p is not None and all(p) and not np.isnan(p).any()
    ]
    
    if len(valid_indices) < 2:
        return golf_utils.velocity_to_distance(0.0)

    # Synchronize pixel array and timestamps to guarantee aligned indices
    valid_pixels = np.array([tracking_buffer[i][2] for i in valid_indices], dtype=np.double)
    valid_timestamps = np.array([tracking_buffer[i][1] for i in valid_indices], dtype=np.double)

    # 2. Map ONLY valid pixel coordinates to world coordinates
    # (Fixes the shape mismatch bug from the original snippet where world coords were unsliced)
    path_world_coords = camera_utils.map_camera_to_world_coords(
        valid_pixels,
        camera_params,
        z=camera_utils.golfball_perceived_center()
    ).astype(np.double)

    # 3. Normalize timestamps so t[0] = 0.0
    # This guarantees the linear term (coeff[-2]) directly equals initial velocity at t=0
    t_norm = valid_timestamps - valid_timestamps[0]

    # 4. Choose polynomial degree based on available frame count
    # Degree 2 requires >= 3 points; fall back to linear (degree 1) if only 2 points exist
    deg = 2 if len(valid_indices) >= 3 else 1

    v0_components = []
    num_dimensions = path_world_coords.shape[1]  # Works for 2D (X, Y) or 3D (X, Y, Z)

    # Fit coordinate dimensions independently against time
    for d in range(num_dimensions):
        coords_d = path_world_coords[:, d]
        coeffs = np.polyfit(t_norm, coords_d, deg=deg)
        
        # np.polyfit returns coefficients highest-degree first:
        # deg=2 -> [A, B, C] where A*(t^2) + B*t + C  => B is initial velocity (v0)
        # deg=1 -> [B, C]    where B*t + C            => B is constant velocity (v0)
        v0_d = coeffs[-2]
        v0_components.append(v0_d)

    # 5. Calculate initial speed (magnitude of initial velocity vector)
    initial_velocity = np.linalg.norm(v0_components)

    return golf_utils.velocity_to_distance(initial_velocity)

def evaluate_distance_v3(tracking_buffer, camera_params):
    # 1. Filter out missing/invalid detection frames up front
    valid_indices = [
        i for i, (_, t, p) in enumerate(tracking_buffer) 
        if p is not None and all(p) and not np.isnan(p).any()
    ]
    
    if len(valid_indices) < 2:
        return golf_utils.velocity_to_distance(0.0)

    # Synchronize pixel array and timestamps to guarantee aligned indices
    valid_pixels = np.array([tracking_buffer[i][2] for i in valid_indices], dtype=np.double)
    valid_timestamps = np.array([tracking_buffer[i][1] for i in valid_indices], dtype=np.double)

    # 2. Map ONLY valid pixel coordinates to world coordinates
    path_world_coords = camera_utils.map_camera_to_world_coords(
        valid_pixels,
        camera_params,
        z=camera_utils.golfball_perceived_center()
    ).astype(np.double)

    # 3. Normalize time so the first detection frame is t = 0.0
    t = valid_timestamps - valid_timestamps[0]

    # Handle minimal 2-point tracking edge case directly
    if len(valid_indices) == 2:
        dt = t[1] - t[0]
        v0 = np.linalg.norm(path_world_coords[1] - path_world_coords[0]) / dt if dt > 0 else 0.0
        return golf_utils.velocity_to_distance(v0)

    # 4. Spatial Noise Reduction via SVD (PCA Projection)
    # Subtract centroid to zero-center coordinates
    mean_coords = np.mean(path_world_coords, axis=0)
    centered_coords = path_world_coords - mean_coords
    
    # SVD extracts the primary line of motion (1st right-singular vector)
    _, _, vh = np.linalg.svd(centered_coords)
    motion_axis = vh[0]  
    
    # Project 2D/3D points onto a 1D distance scalar 's' along the main path
    s = centered_coords @ motion_axis
    
    # Guarantee distance 's' strictly increases over time
    if s[-1] < s[0]:
        s = -s

    # 5. Fit 1D Robust Kinematic Model: s(t) = 0.5 * a * t^2 + v0 * t + s0
    def kinematic_model(t, a, v0, s0):
        return 0.5 * a * t**2 + v0 * t + s0

    # Initial guesses: [a = 0, v0 = finite difference avg, s0 = starting position]
    avg_velocity = (s[-1] - s[0]) / t[-1] if t[-1] > 0 else 0.0
    p0 = [0.0, avg_velocity, s[0]]

    try:
        # 'huber' loss down-weights spatial tracking glitches at frame boundaries
        popt, _ = curve_fit(
            kinematic_model, 
            t, 
            s, 
            p0=p0,
            method='trf',
            loss='huber', 
            f_scale=1.0
        )
        _, v0, _ = popt
    except RuntimeError:
        print("v3 exception")
        # Fallback to linear average velocity if optimization fails to converge
        v0 = avg_velocity

    # Ensure velocity is non-negative along the direction of travel
    v0 = max(0.0, v0)

    return golf_utils.velocity_to_distance(v0)

def evaluate_distance_v4(tracking_buffer, camera_params, spatial_outlier_threshold=0.015):
    """
    v4 trajectory evaluation using SVD dimensionality reduction combined with 
    a physically bounded Huber loss kinematic fit (a <= 0).
    """
    # 1. Filter out missing/invalid detection frames up front
    valid_indices = [
        i for i, (_, t, p) in enumerate(tracking_buffer) 
        if p is not None and all(p) and not np.isnan(p).any()
    ]
    
    if len(valid_indices) < 2:
        return golf_utils.velocity_to_distance(0.0)

    # Synchronize pixel array and timestamps to guarantee aligned indices
    valid_pixels = np.array([tracking_buffer[i][2] for i in valid_indices], dtype=np.double)
    valid_timestamps = np.array([tracking_buffer[i][1] for i in valid_indices], dtype=np.double)

    # 2. Map ONLY valid pixel coordinates to world coordinates
    path_world_coords = camera_utils.map_camera_to_world_coords(
        valid_pixels,
        camera_params,
        z=camera_utils.golfball_perceived_center()
    ).astype(np.double)

    # 3. Normalize time so the first detection frame is t = 0.0
    t = valid_timestamps - valid_timestamps[0]

    # Handle minimal 2-point tracking edge case directly
    if len(valid_indices) == 2:
        dt = t[1] - t[0]
        v0 = np.linalg.norm(path_world_coords[1] - path_world_coords[0]) / dt if dt > 0 else 0.0
        return golf_utils.velocity_to_distance(v0)

    # 4. Spatial Noise Reduction via SVD (PCA Projection)
    # Subtract centroid to zero-center coordinates
    mean_coords = np.mean(path_world_coords, axis=0)
    centered_coords = path_world_coords - mean_coords
    
    # SVD extracts the primary line of motion (1st right-singular vector)
    _, _, vh = np.linalg.svd(centered_coords)
    motion_axis = vh[0]  
    
    # Project 2D/3D points onto a 1D distance scalar 's' along the main path
    s = centered_coords @ motion_axis
    
    # Guarantee distance 's' strictly increases over time
    if s[-1] < s[0]:
        s = -s

    # 5. Fit 1D Bounded Kinematic Model: s(t) = 0.5 * a * t^2 + v0 * t + s0
    def kinematic_model(t, a, v0, s0):
        return 0.5 * a * t**2 + v0 * t + s0

    # Initial guesses: [a = 0, v0 = finite difference avg, s0 = initial position]
    avg_velocity = (s[-1] - s[0]) / t[-1] if t[-1] > 0 else 0.0
    p0 = [0.0, max(0.0, avg_velocity), s[0]]

    # Physical Bounds:
    #   a  in (-inf, 0.0]  -> Enforces friction deceleration / zero acceleration
    #   v0 in [0.0, inf)   -> Initial speed must be non-negative
    #   s0 in (-inf, inf)  -> Position offset can vary freely
    bounds = (
        [-np.inf, 0.0, -np.inf],
        [0.0, np.inf, np.inf]
    )

    try:
        # Robust Trust Region Reflective algorithm with Huber loss thresholding
        popt, _ = curve_fit(
            kinematic_model, 
            t, 
            s, 
            p0=p0,
            bounds=bounds,
            method='trf',
            loss='huber', 
            f_scale=spatial_outlier_threshold  # 1.5 cm threshold in world units (meters)
        )
        _, v0, _ = popt
    except (RuntimeError, ValueError):
        print("v4 exception")
        # Fallback to linear average velocity if optimization fails to converge
        v0 = max(0.0, avg_velocity)

    return golf_utils.velocity_to_distance(v0)

def evaluate_distance_v5(tracking_buffer, camera_params, residual_threshold=0.015):
    """
    v5 trajectory evaluation using SVD dimensionality reduction combined with 
    scikit-learn's RANSACRegressor for hard outlier rejection, featuring 
    physical deceleration enforcement (a <= 0).
    """
    # 1. Filter out missing/invalid detection frames up front
    valid_indices = [
        i for i, (_, t, p) in enumerate(tracking_buffer) 
        if p is not None and all(p) and not np.isnan(p).any()
    ]
    
    if len(valid_indices) < 2:
        return golf_utils.velocity_to_distance(0.0)

    # Synchronize pixel array and timestamps to guarantee aligned indices
    valid_pixels = np.array([tracking_buffer[i][2] for i in valid_indices], dtype=np.double)
    valid_timestamps = np.array([tracking_buffer[i][1] for i in valid_indices], dtype=np.double)

    # 2. Map ONLY valid pixel coordinates to world coordinates
    path_world_coords = camera_utils.map_camera_to_world_coords(
        valid_pixels,
        camera_params,
        z=camera_utils.golfball_perceived_center()
    ).astype(np.double)

    # 3. Normalize time so the first detection frame is t = 0.0
    t = valid_timestamps - valid_timestamps[0]

    # Handle minimal 2-point tracking edge case directly
    if len(valid_indices) == 2:
        dt = t[1] - t[0]
        v0 = np.linalg.norm(path_world_coords[1] - path_world_coords[0]) / dt if dt > 0 else 0.0
        return golf_utils.velocity_to_distance(v0)

    # 4. Spatial Noise Reduction via SVD (PCA Projection)
    mean_coords = np.mean(path_world_coords, axis=0)
    centered_coords = path_world_coords - mean_coords
    
    # SVD extracts the primary line of motion (1st right-singular vector)
    _, _, vh = np.linalg.svd(centered_coords)
    motion_axis = vh[0]  
    
    # Project 2D/3D points onto a 1D distance scalar 's' along the main path
    s = centered_coords @ motion_axis
    
    # Guarantee distance 's' strictly increases over time
    if s[-1] < s[0]:
        s = -s

    # 5. Build Feature Matrix for Kinematic Fit: s(t) = s0 + v0*t + 0.5*a*t^2
    # Feature 0: t       -> Coefficient is v0 (initial velocity)
    # Feature 1: 0.5*t^2 -> Coefficient is a  (acceleration)
    X = np.column_stack((t, 0.5 * t**2))

    # 6. Fit RANSAC via scikit-learn
    ransac = RANSACRegressor(
        estimator=LinearRegression(fit_intercept=True),
        min_samples=3,
        residual_threshold=residual_threshold,  # 1.5 cm default spatial tolerance
        random_state=42
    )

    try:
        ransac.fit(X, s)
        
        # Extract coefficients: coef_[0] = v0, coef_[1] = a
        v0 = ransac.estimator_.coef_[0]
        a = ransac.estimator_.coef_[1]
        inliers = ransac.inlier_mask_

        print(ransac.estimator_.coef_)

        # 7. Physical Deceleration Guard (a <= 0)
        # If noise causes the quadratic fit to predict positive acceleration (speeding up),
        # re-fit a 1D linear model (constant velocity, a = 0) over the RANSAC inliers.
        if a > 0 or v0 < 0:
            print("ransac positive acceleration")
            if np.sum(inliers) >= 2:
                v0, _ = np.polyfit(t[inliers], s[inliers], deg=1)
            else:
                v0 = (s[-1] - s[0]) / t[-1] if t[-1] > 0 else 0.0

    except (ValueError, RuntimeError):
        print("v5 exception")
        # Fallback to linear average velocity if RANSAC fails (e.g., singular matrix)
        v0 = (s[-1] - s[0]) / t[-1] if t[-1] > 0 else 0.0

    v0 = max(0.0, v0)
    return golf_utils.velocity_to_distance(v0)

def evaluate_distance_v6(tracking_buffer, camera_params, residual_threshold=0.015):
    """
    v5 trajectory evaluation using SVD dimensionality reduction combined with 
    scikit-learn's RANSACRegressor under a constant velocity model (a = 0)
    for robust velocity estimation under heavy spatial noise.
    """
    # 1. Filter out missing/invalid detection frames up front
    valid_indices = [
        i for i, (_, t, p) in enumerate(tracking_buffer) 
        if p is not None and all(p) and not np.isnan(p).any()
    ]
    
    if len(valid_indices) < 2:
        return golf_utils.velocity_to_distance(0.0)

    # Synchronize pixel array and timestamps to guarantee aligned indices
    valid_pixels = np.array([tracking_buffer[i][2] for i in valid_indices], dtype=np.double)
    valid_timestamps = np.array([tracking_buffer[i][1] for i in valid_indices], dtype=np.double)

    # 2. Map ONLY valid pixel coordinates to world coordinates
    path_world_coords = camera_utils.map_camera_to_world_coords(
        valid_pixels,
        camera_params,
        z=camera_utils.golfball_perceived_center()
    ).astype(np.double)

    # 3. Normalize time so the first detection frame is t = 0.0
    t = valid_timestamps - valid_timestamps[0]

    # Handle minimal 2-point tracking edge case directly
    if len(valid_indices) == 2:
        dt = t[1] - t[0]
        v0 = np.linalg.norm(path_world_coords[1] - path_world_coords[0]) / dt if dt > 0 else 0.0
        return golf_utils.velocity_to_distance(v0)

    # 4. Spatial Noise Reduction via SVD (PCA Projection)
    mean_coords = np.mean(path_world_coords, axis=0)
    centered_coords = path_world_coords - mean_coords
    
    # SVD extracts the primary line of motion (1st right-singular vector)
    _, _, vh = np.linalg.svd(centered_coords)
    motion_axis = vh[0]  
    
    # Project 2D/3D points onto a 1D distance scalar 's' along the main path
    s = centered_coords @ motion_axis
    
    # Guarantee distance 's' strictly increases over time
    if s[-1] < s[0]:
        s = -s

    # 5. Build Feature Matrix for Constant Velocity Fit: s(t) = s0 + v0*t
    # Reshape t to (N, 1) for scikit-learn single-feature regression
    X = t.reshape(-1, 1)

    # 6. Fit RANSAC via scikit-learn
    # min_samples=2 is sufficient for a 1D linear model fit
    ransac = RANSACRegressor(
        estimator=LinearRegression(fit_intercept=True),
        min_samples=2,
        residual_threshold=residual_threshold,  # Spatial tolerance along motion axis
        random_state=42
    )

    try:
        ransac.fit(X, s)
        
        # Extract slope v0 (coef_[0])
        v0 = ransac.estimator_.coef_[0]
        inliers = ransac.inlier_mask_

        # 7. Direction Guard
        # If noise causes a negative velocity prediction despite orientation check:
        if v0 < 0:
            if np.sum(inliers) >= 2:
                v0, _ = np.polyfit(t[inliers], s[inliers], deg=1)
            else:
                v0 = (s[-1] - s[0]) / t[-1] if t[-1] > 0 else 0.0

    except (ValueError, RuntimeError):
        # Fallback to linear average velocity if RANSAC fails (e.g., singular matrix)
        v0 = (s[-1] - s[0]) / t[-1] if t[-1] > 0 else 0.0

    v0 = max(0.0, v0)
    return golf_utils.velocity_to_distance(v0)

def evaluate_distance_v7(tracking_buffer, camera_params):
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

    print(path_world_coords[-1], path_world_coords[0])
    d = np.linalg.norm(path_world_coords[-1] - path_world_coords[0])
    t = timestamp_path[-1] - timestamp_path[0]
    return golf_utils.velocity_to_distance(d/t)
