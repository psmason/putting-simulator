import cv2
import numpy as np

import params

def find_chessboard(gray):
  """
  Finds and refines a standard chess board.

  The input image is expected to be grayscale.
  """
  if len(gray.shape) != 2:
      raise "find_chessboard image isn't grayscale"
  cb_found, cb = cv2.findChessboardCorners(gray, params.CHESSBOARD_TILES)
  if not cb_found:
    return False, None

  # Refining corner location.
  return True, cv2.cornerSubPix(gray, cb, (11,11), (-1,-1),
   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

def calibrate_camera(gray, cb):
    cbcols, cbrows = params.CHESSBOARD_TILES
    
    objp = np.zeros((1, cbrows*cbcols, 3), np.float32)
    # TODO: is cbcols and cbrows correct?
    objp[0,:,:2] = np.mgrid[0:cbcols, 0:cbrows].T.reshape(-1, 2)
    # Millimeter coordinates.
    # On the test chessboard, each tile is 2x2 inches.
    objp = objp * params.CHESSBOARD_TILE_LENGTH

    # TODO: do a geometry test for calibration.

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [cb], gray.shape[::-1], None, None)
    camera_params = {}
    camera_params["mtx"] = mtx
    camera_params["dist"] = dist
    camera_params["rvecs"] = rvecs
    camera_params["tvecs"] = tvecs

    test_calibration_geometry(cb, camera_params)
    return camera_params

def test_calibration_geometry(cb, camera_params):
    cbcols, cbrows = params.CHESSBOARD_TILES
    if cbrows*cbcols != len(cb):
      raise "Cannot evaluate geometry. Unexpected number of corner positions"
    
    # z=0.0 because corner points are assumed to be on the ground.
    cb_world_coords = map_camera_to_world_coords(cb,
                                                 camera_params,
                                                 z=0.0)

    # Testing if the central cross lines of a chessboard are perpindicular.
    # Assumes row-major order.
    center_row = cbrows // 2
    center_col = cbcols // 2
    # For 7x7, this was position 21 - position 27
    v1 = (cb_world_coords[center_row*cbcols] - cb_world_coords[center_row*cbcols + cbcols-1]).flatten()
    # For 7x7, this was position 4 - position 32
    v2 = (cb_world_coords[center_col] - cb_world_coords[center_col + cbcols*(cbrows-1)]).flatten()

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Should be close to 90 degrees, within 0.25 degrees.
    cross_angle = np.rad2deg(np.arccos(np.dot(v1, v2)))
    print("Geometry eval: orthogonality (degrees) = {:.2f}".format(cross_angle))
    if abs(cross_angle - 90.0) >= 0.25:
        return False
    return True

def draw_world_grid_overlay_v1(
    img,
    camera_params,
    grid_spacing_mm=100.0,
):
    """Draws discrete points along both chessboard axes 
    starting from the origin using vectorized batch projection.
    """
    height, width = img.shape[:2]
    num_intervals = 30

    # 1. Generate all world points for both axes in a single batch
    i_vals = np.arange(num_intervals, dtype=np.float32) * grid_spacing_mm
    zeros = np.zeros_like(i_vals)

    # X-axis points: (X increases, Y=0, Z=0)
    x_world_pts = np.column_stack((i_vals, zeros, zeros))
    # Y-axis points: (X=0, Y increases, Z=0)
    y_world_pts = np.column_stack((zeros, i_vals, zeros))

    # Combine into a single batch array of shape (40, 3)
    all_world_pts = np.vstack((x_world_pts, y_world_pts)).astype(np.float32)

    # 2. Project the entire batch in a single call (returns an (N, 2) array)
    px_pts = map_world_to_camera_coords(all_world_pts, camera_params)

    # 3. Filter and draw valid points
    if px_pts is not None:
        for pt in px_pts:
            pt_int = (int(pt[0]), int(pt[1]))
            
            # Draw point if within image bounds
            if 0 <= pt_int[0] < width and 0 <= pt_int[1] < height:
                cv2.circle(img, pt_int, 5, (0, 0, 255), -1)

def draw_world_grid_overlay(
    img,
    camera_params,
    grid_spacing_mm=100.0,
):
    """Draws discrete points along a single chessboard axis 
    starting from the origin using an iterative loop.
    """
    height, width = img.shape[:2]
    
    # Iterate for 10 intervals along the X-axis starting from origin (0, 0, 0)
    for i in range(40):
        # Construct a single 3D point (X increases, Y=0, Z=0)
        world_pt = np.array([i * grid_spacing_mm, 0.0, 0.0], dtype=np.float32)
        
        # Project single point
        px_pt = map_world_to_camera_coords(world_pt, camera_params)
        
        if px_pt is not None:
            pt_int = (int(px_pt[0]), int(px_pt[1]))
            
            # Draw point if within image bounds
            if 0 <= pt_int[0] < width and 0 <= pt_int[1] < height:
                cv2.circle(img, pt_int, 5, (0, 0, 255), -1)

                world_pt = np.array([i * grid_spacing_mm, 0.0, 0.0], dtype=np.float32)
        
        # Project single point
        world_pt = np.array([0.0, i * grid_spacing_mm, 0.0], dtype=np.float32)
        py_pt = map_world_to_camera_coords(world_pt, camera_params)
        
        if py_pt is not None:
            pt_int = (int(py_pt[0]), int(py_pt[1]))
            
            # Draw point if within image bounds
            if 0 <= pt_int[0] < width and 0 <= pt_int[1] < height:
                cv2.circle(img, pt_int, 5, (0, 0, 255), -1)

def fit_neoprene_mask_ransac(
    cleaned_mask,
    camera_params,
    neoprene_width_mm,
    max_iterations=8000,
    z_plane=0.0,
):
  """Fits strictly parallel borders by maximizing mask occupancy (area overlap)

  rather than fitting sparse Canny edge points.
  """
  # 1. Get all foreground pixel coordinates from the binary mask
  # Note: argwhere returns [y, x]. Convert to world coordinates.
  y_indices, x_indices = np.nonzero(cleaned_mask)
  if len(x_indices) < 100:
    return None, None, None

  # Subsample mask points if the array is too dense to keep RANSAC fast
  if len(x_indices) > 2000:
    indices = np.random.choice(len(x_indices), 2000, replace=False)
    x_indices, y_indices = x_indices[indices], y_indices[indices]

  image_pts = np.column_stack((x_indices, y_indices)).astype(np.float32)
  image_pts_reshaped = image_pts.reshape(-1, 1, 2)

  # Map sampled mask pixels to 3D world coordinates once
  world_pts = map_camera_to_world_coords(
      image_pts_reshaped, camera_params, z=z_plane
  )
  pts_2d = world_pts[:, :2]  # (X, Y) on the ground plane

  best_score = -1
  best_lines = None

  half_width = neoprene_width_mm / 2.0

  for _ in range(max_iterations):
    # 2. Sample two random world points to define the centerline orientation
    idx = np.random.choice(len(pts_2d), 2, replace=False)
    p1, p2 = pts_2d[idx[0]], pts_2d[idx[1]]

    v = p2 - p1
    baseline = np.linalg.norm(v)
    if baseline < 300.0:  # Minimum baseline check in mm
      continue
    v /= baseline
    a, b = -v[1], v[0]  # Normal vector [a, b]

    # Centerline offset
    d_center = -(a * p1[0] + b * p1[1])

    # The two parallel borders at exactly +W/2 and -W/2 from the centerline
    d1 = d_center + half_width
    d2 = d_center - half_width

    # 3. Region Scoring:
    # Calculate perpendicular distance of every mask point to the centerline
    center_distances = a * pts_2d[:, 0] + b * pts_2d[:, 1] + d_center

    # A point is inside the neoprene strip if its distance from the centerline
    # is less than half_width.
    inside_strip = np.abs(center_distances) <= half_width

    # Score is simply the count of mask pixels correctly falling inside the predicted strip width
    score = np.sum(inside_strip)

    if score > best_score:
      best_score = score
      best_lines = ((a, b, d1), (a, b, d2))

  if best_lines is None:
    return None, None, None

  return best_lines[0], best_lines[1], None

def fit_neoprene_mask_ransac_v2(
    cleaned_mask,
    camera_params,
    neoprene_width_mm,
    max_iterations=16000,
    refine_iterations=16000,
    max_angle_deg=2.0,
    max_offset_mm=50.0,
    z_plane=0.0,
):
    """Fits strictly parallel borders by maximizing mask occupancy (area overlap)
    using a two-stage RANSAC pipeline with normally distributed local perturbations.
    """
    y_indices, x_indices = np.nonzero(cleaned_mask)
    if len(x_indices) < 100:
        return None, None, None

    if len(x_indices) > 2000:
        indices = np.random.choice(len(x_indices), 2000, replace=False)
        x_indices, y_indices = x_indices[indices], y_indices[indices]

    image_pts = np.column_stack((x_indices, y_indices)).astype(np.float32)
    image_pts_reshaped = image_pts.reshape(-1, 1, 2)

    world_pts = map_camera_to_world_coords(
        image_pts_reshaped, camera_params, z=z_plane
    )
    pts_2d = world_pts[:, :2]

    best_score = -1
    best_a, best_b, best_d_center = None, None, None

    half_width = neoprene_width_mm / 2.0

    # ==========================================
    # STAGE 1: Global Minimal-Sample RANSAC
    # ==========================================
    for _ in range(max_iterations):
        idx = np.random.choice(len(pts_2d), 2, replace=False)
        p1, p2 = pts_2d[idx[0]], pts_2d[idx[1]]

        v = p2 - p1
        baseline = np.linalg.norm(v)
        if baseline < 300.0:  # Minimum baseline check in mm
            continue
        v /= baseline
        a, b = -v[1], v[0]  # Normal vector [a, b]

        d_center = -(a * p1[0] + b * p1[1])

        center_distances = a * pts_2d[:, 0] + b * pts_2d[:, 1] + d_center
        inside_strip = np.abs(center_distances) <= half_width
        score = np.sum(inside_strip)

        if score > best_score:
            best_score = score
            best_a, best_b, best_d_center = a, b, d_center

    if best_a is None:
        return None, None, None

    # ==========================================
    # STAGE 2: Local Refinement (Normal Distributions)
    # ==========================================
    centroid = np.mean(pts_2d, axis=0)
    dist_to_centroid = best_a * centroid[0] + best_b * centroid[1] + best_d_center
    pivot = centroid - np.array([best_a, best_b]) * dist_to_centroid

    refined_best_score = best_score
    refined_a, refined_b, refined_d_center = best_a, best_b, best_d_center

    max_angle_rad = np.radians(max_angle_deg)
    sigma_angle = max_angle_rad / 3.0
    sigma_offset = max_offset_mm / 3.0

    for _ in range(refine_iterations):
        # 1. Bounded normal distribution for perpendicular shift (mm)
        d_shift = np.clip(
            np.random.normal(0.0, sigma_offset),
            -max_offset_mm,
            max_offset_mm
        )
        
        # 2. Bounded normal distribution for rotation angle (radians)
        d_theta = np.clip(
            np.random.normal(0.0, sigma_angle),
            -max_angle_rad,
            max_angle_rad
        )
        
        cos_t, sin_t = np.cos(d_theta), np.sin(d_theta)

        # Apply rotation to base normal vector [best_a, best_b]
        a_cand = cos_t * best_a - sin_t * best_b
        b_cand = sin_t * best_a + cos_t * best_b

        norm_val = np.hypot(a_cand, b_cand)
        if norm_val < 1e-6:
            continue
        a_cand /= norm_val
        b_cand /= norm_val

        # 3. Combine shifted baseline and rotated normal
        d_center_cand = -(a_cand * pivot[0] + b_cand * pivot[1]) + d_shift

        # Score candidate
        center_distances = a_cand * pts_2d[:, 0] + b_cand * pts_2d[:, 1] + d_center_cand
        inside_strip = np.abs(center_distances) <= half_width
        score = np.sum(inside_strip)

        if score > refined_best_score:
            refined_best_score = score
            refined_a, refined_b, refined_d_center = a_cand, b_cand, d_center_cand

    d1 = refined_d_center + half_width
    d2 = refined_d_center - half_width

    return (refined_a, refined_b, d1), (refined_a, refined_b, d2), None


def intersect_lines(line1, line2):
    """Finds the 3D world intersection point (X, Y, Z=0) of two lines: Ax + By + D = 0."""
    a1, b1, d1 = line1
    a2, b2, d2 = line2
    
    A = np.array([[a1, b1], [a2, b2]], dtype=np.float32)
    b = np.array([-d1, -d2], dtype=np.float32)
    
    if abs(np.linalg.det(A)) < 1e-6:
        raise ValueError("Lines are parallel or coincident; intersection undefined.")
    
    xy = np.linalg.solve(A, b)
    return np.array([xy[0], xy[1], 0.0], dtype=np.float32)

def get_neoprene_roi(top_line, bottom_line, roi_offset_mm):
    cbcols, cbrows = params.CHESSBOARD_TILES
    center_x = ((cbcols - 1) * params.CHESSBOARD_TILE_LENGTH) / 2.0
    center_y = ((cbrows - 1) * params.CHESSBOARD_TILE_LENGTH) / 2.0
    center_z = 0.0
    cb_center = np.array([center_x, center_y, center_z], dtype=np.float32)

    a, b, _ = top_line  # Unit normal vector [a, b] across the neoprene width
    Xc, Yc, _ = cb_center
    
    # Perpendicular normal vector running across the strip through the chessboard center
    a_perp = -b
    b_perp = a
    
    # Constant term for centerline passing through (Xc, Yc)
    d_center = -(a_perp * Xc + b_perp * Yc)
    
    # Offset lines at +X and -X millimeters
    d_left = d_center + roi_offset_mm
    d_right = d_center - roi_offset_mm
    
    centerline = (a_perp, b_perp, d_center)
    left_line  = (a_perp, b_perp, d_left)
    right_line = (a_perp, b_perp, d_right)

    tl = intersect_lines(top_line, left_line)
    tr = intersect_lines(top_line, right_line)
    br = intersect_lines(bottom_line, right_line)
    bl = intersect_lines(bottom_line, left_line)

    return tl, tr, br, bl

def get_camera_pose(rvecs, tvecs):
    """Computes the rotation matrix R and camera origin C in world coordinates."""
    r, _ = cv2.Rodrigues(rvecs[0])
    ir = r.T # The transpose of a rotation matrix is its inverse.
    
    # Camera location in world coordinates: C_w = -R^T * t
    camera_position = np.matmul(-ir, tvecs[0])
    
    return ir, camera_position

def get_calibrated_world_coordinates(pixel_coords, mtx, rvecs, tvecs, z):
    ir, camera_position = get_camera_pose(rvecs, tvecs)
    imtx = np.linalg.inv(mtx)

    camera_coords = np.array([pixel_coords[0,0], pixel_coords[0,1], 1], np.float32)[:, np.newaxis]
    camera_ray = np.matmul(imtx, camera_coords)
    world_ray = np.matmul(ir, camera_ray)

    # In this ray in world coodinates, where does it get to the targeted heigh "z"
    ray_z_dir = world_ray[2, 0]
    cam_z_pos = camera_position[2, 0]
    s = (z - cam_z_pos) / ray_z_dir

    world_point = camera_position + s * world_ray
    return world_point.flatten()

def map_world_to_camera_coords(points, camera_params):
    mtx, rvec, tvec, dist = camera_params["mtx"], camera_params["rvecs"][0], camera_params["tvecs"][0], camera_params["dist"]
    dist = np.zeros((5, 1), dtype=np.float32)
    
    result, _ = cv2.projectPoints(points, rvec, tvec, mtx, dist)

    # If a single point (1D array of shape (3,)) was passed, return a single coordinate pair
    if points.ndim == 1:
        return result[0][0]
    
    # If a batch of points (2D array of shape (N, 3)) was passed, return an (N, 2) array
    return result.reshape(-1, 2)

def map_camera_to_world_coords(points, camera_params, z):
    dist = np.zeros((5, 1), dtype=np.float32)
    undistorted = cv2.undistortPoints(points,
                                      camera_params["mtx"],
                                      #camera_params["dist"],
                                      dist,
                                      P=camera_params["mtx"])
    return np.array([
        get_calibrated_world_coordinates(p,
                                         camera_params["mtx"],
                                         camera_params["rvecs"],
                                         camera_params["tvecs"],
                                         z)
        for p in undistorted
    ], np.float32)

def golfball_perceived_center():
    return params.GOLF_BALL_DIAMETER * params.PERCEIVED_HEIGHT_FRACTION

def fit_tracked_path(pixel_path, camera_params):
    golfball_height = golfball_perceived_center()
    path_world_coords = map_camera_to_world_coords(pixel_path,
                                                   camera_params,
                                                   golfball_height)
    #return cv2.fitLine(path_world_coords, cv2.DIST_L2, 0, 0.01, 0.01)
    return cv2.fitLine(path_world_coords, cv2.DIST_HUBER, 3, 0.01, 0.01)
