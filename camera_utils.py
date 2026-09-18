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
  cb_found, cb = cv2.findChessboardCorners(gray, (7,7))
  if not cb_found:
    return False, None

  # Refining corner location.
  return True, cv2.cornerSubPix(gray, cb, (11,11), (-1,-1),
   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

def calibrate_camera(gray, cb):
    objp = np.zeros((1, 7*7, 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)
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
    # z=0.0 because corner points are assumed to be on the ground.
    cb_world_coords = map_camera_to_world_coords(cb,
                                                 camera_params,
                                                 z=0.0)

    # Testing if the central cross lines of a chessboard are perpindicular.
    v1 = (cb_world_coords[21] - cb_world_coords[27]).flatten()
    v2 = (cb_world_coords[4] - cb_world_coords[32]).flatten()

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Should be close to 90 degrees, within 0.25 degrees.
    cross_angle = np.rad2deg(np.arccos(np.dot(v1, v2)))
    print("Geometry eval: orthogonality (degrees) = {:.2f}".format(cross_angle))
    if abs(cross_angle - 90.0) >= 0.25:
        return False
    return True
    

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

def map_world_to_camera_coords(point, camera_params):
    mtx, rvec, tvec, dist = camera_params["mtx"], camera_params["rvecs"][0], camera_params["tvecs"][0], camera_params["dist"]
    result, _ = cv2.projectPoints(point, rvec, tvec, mtx, dist)
    return result[0][0]

def map_camera_to_world_coords(points, camera_params, z):
    undistorted = cv2.undistortPoints(points,
                                      camera_params["mtx"],
                                      camera_params["dist"],
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
