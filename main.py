import cv2
from datetime import datetime
import hashlib
import json
import time

import numpy as np

from collections import deque

import camera_utils
import golf_utils
import params

def write_text(frame, text):
    coordinates = (50, 100)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)  # Green color in BGR format
    thickness = 2
    line_type = cv2.LINE_AA  # Smooth anti-aliased line type

    cv2.putText(frame, text, coordinates, font, font_scale, color, thickness, line_type)

def debug_string(state):
    time_delta = time.time() - state["timer"]
    if time_delta >= 1.0:
        state["frame_rate"] = 1.0 * state["frame_count"] / time_delta
        state["timer"] = time.time()
        state["frame_count"] = 0
    return "state: {} :: frame rate: {:.2f}".format(state["state"], state["frame_rate"])

# def annotate(frame, state):
    
    

def get_chessboard_bounding_rect(cb):
  hull = cv2.convexHull(cb, returnPoints=True)
  return cv2.boundingRect(hull)  

def get_chessboard_subimage(frame, cb):
  """
  Masks a grayscale image to the found chessboard, expanded with a surrounding
  buffer to also capture the edge of the chessboard.
  """
  x, y, w, h = get_chessboard_bounding_rect(cb)

  # in computer vision rows and columns are flipped.
  return frame[y:y+h, x:x+w, :]

def get_median_pixels(frames):
  blurred_l, blurred_a, blurred_b = [], [], []

  target_shape = frames[0].shape
  
  for f in frames:
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
    blurred = cv2.GaussianBlur(lab, (11, 11), 2)
    l, a, b = cv2.split(blurred)

    blurred_l.append(l)
    blurred_a.append(a)
    blurred_b.append(b)

  if len(blurred_l) < 10:
      raise "Insufficient l channel images"
  if len(blurred_a) < 10:
      raise "Insufficient a channel images"
  if len(blurred_b) < 10:
      raise "Insufficient b channel images"
    
  stacked_l = np.stack(blurred_l, axis=2)
  stacked_a = np.stack(blurred_a, axis=2)
  stacked_b = np.stack(blurred_b, axis=2)
  return np.median(stacked_l, axis=2), np.median(stacked_a, axis=2), np.median(stacked_b, axis=2)

def diff_against_median(frame, median_channels):
  ml, ma, mb = median_channels["ml"], median_channels["ma"], median_channels["mb"]
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

  lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
  blurred = cv2.GaussianBlur(lab, (7, 7), 2)
  l, a, b = cv2.split(blurred)
  l = l.astype(np.double)
  a = a.astype(np.double)
  b = b.astype(np.double)

  ml = ml.astype(np.double)
  ma = ma.astype(np.double)
  mb = mb.astype(np.double)

  print("Script 2 L, a, b means:", np.mean(l), np.mean(a), np.mean(b))

  ab_distance = np.square(np.square(a - ma) + np.square(b - mb))
  l_d = np.square(l-ml)
  return cv2.morphologyEx(l_d * ab_distance, cv2.MORPH_OPEN, kernel)

def get_background_diff(frames, median_channels):
    m00s = []
    for f in frames:
        if f.shape[0:2] != median_channels["ml"].shape:
            continue
        if f.shape[0:2] != median_channels["ma"].shape:
            continue
        if f.shape[0:2] != median_channels["mb"].shape:
            continue
        d = diff_against_median(f, median_channels)
        m00s.append(cv2.moments(d, binaryImage=False)["m00"])

    if len(m00s) < 10:
        raise "Insufficient background moments"
        
    return np.mean(m00s)

def find_golfball(diffed_frame, background_mass):
  M = cv2.moments(diffed_frame, binaryImage=False)
  if M["m00"] < 25.0 * background_mass:
      return None, None
  cX = int(M["m10"] / M["m00"])
  cY = int(M["m01"] / M["m00"])
  return cX, cY

def score_direction(pv, cbv1, cbv2):
    d1 = np.rad2deg(np.arccos(np.dot(pv, cbv1))) % 90.0
    d1 = min(90 - d1, d1)

    d2 = np.rad2deg(np.arccos(np.dot(pv, cbv2))) % 90.0
    d2 = min(90 - d2, d2)

    return np.mean([d1, d2])

def fit_tracked_path(pixel_path, state):
    golfball_height = camera_utils.golfball_perceived_center()
    path_world_coords = camera_utils.map_camera_to_world_coords(pixel_path,
                                                                state,
                                                                z=golfball_height)
    return cv2.fitLine(path_world_coords, cv2.DIST_L2, 0, 0.01, 0.01)
     
def estimate_path(state):
    pixel_path = np.array([x for x in state["tracked_object_buffer"] if all(x)],
                          np.float32)
    #golfball_height = camera_utils.golfball_perceived_center()
    #path_world_coords = camera_utils.map_camera_to_world_coords(pixel_path,
    #                                                            state,
    #                                                            z=golfball_height)
    #cv2.fitLine(path_world_coords, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, vz, x0, y0, z0 = fit_tracked_path(pixel_path, state)
    # Because z is a constant, we can ignore vz and z0 terms if we want.

    pv = np.array([vx, vy, vz], np.float32).flatten()
    pv = pv / np.linalg.norm(pv)

    cb_world_coords = camera_utils.map_camera_to_world_coords(state["cb"],
                                                              state,
                                                              z=0.0)
    # The chessboard might be oriented in four directions, in two pairs perpendicular
    # to each other, and two pairs opposite each other.
    # As a result, when scoring the detected path, we need do the scoring
    # in the two potential orientations, and taking the best score.
    cbv1 = (cb_world_coords[21] - cb_world_coords[27]).flatten()
    cbv2 = (cb_world_coords[4] - cb_world_coords[32]).flatten()

    cbv1 = cbv1 / np.linalg.norm(cbv1)
    cbv2 = cbv2 / np.linalg.norm(cbv2)

    print("Path eval: direction (degrees) = {:.2f}".format(
        score_direction(pv, cbv1, cbv2)
    ))

def estimate_velocity(state):
    """Estimates the velocity as mms per second."""
    if len(state["tracked_object_buffer"]) != len(state["tracked_timestamp_buffer"]):
        raise "tracked object buffer and timestamp buffers aren't the same length"
    
    path_indices = []
    for i in range(len(state["tracked_object_buffer"])):
        if all(state["tracked_object_buffer"][i]):
            path_indices.append(i)
    if len(path_indices) == 0:
        raise "estimate_velocity called with no tracked path"

    pixel_path = np.array([state["tracked_object_buffer"][i] for i in path_indices],
                          np.double)
    timestamp_path = np.array([state["tracked_timestamp_buffer"][i] for i in path_indices],
                              np.double)

    path_world_coords = camera_utils.map_camera_to_world_coords(
        pixel_path,
        state,
        z=camera_utils.golfball_perceived_center())

    distances_traveled = np.linalg.norm(np.diff(path_world_coords, axis=0), axis=1)
    time_taken = np.diff(timestamp_path)
    velocities = distances_traveled / time_taken

    # Returning the average after removing outliers (largest and smallest)
    return np.mean(np.sort(velocities)[1:-1])
    

def draw_result(frame, state):
    pixel_path = np.array([x for x in state["tracked_object_buffer"] if all(x)],
                          np.float32)
    # Blue dots where the putt was tracked
    for x, y in pixel_path:
        cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), 2)

    # Green line for the estimated path, drawn within the
    # chessboard's bounding box.
    vx, vy, vz, x0, y0, z0 = fit_tracked_path(pixel_path, state)

    # We calibrated the camera with physical dimensions,
    # so calculating the bounding box is trivial.
    cb_world_box = (0, 0, int(50.8*7), int(50.8*7))

    s = 10*50.8
    #print(vz)
    p1 = (int(x0[0] + s*vx[0]), int(y0[0] + s*vy[0]))
    p2 = (int(x0[0] - s*vx[0]), int(y0[0] - s*vy[0]))

    inside, cp1, cp2 = cv2.clipLine(cb_world_box, p1, p2)
    if inside:
        #print(cp1, cp2, (x0, y0, z0))
        cp1 = camera_utils.map_world_to_camera_coords(
            np.array((cp1[0],
                      cp1[1],
                      camera_utils.golfball_perceived_center()), np.float32),
            state)
        cp2 = camera_utils.map_world_to_camera_coords(
            np.array((cp2[0],
                      cp2[1],
                      camera_utils.golfball_perceived_center()), np.float32),
            state)
        cv2.line(frame,
                 (int(cp1[0]), int(cp1[1])),
                 (int(cp2[0]), int(cp2[1])),
                 (0, 255, 0),
                 thickness=1)
        
    #print(inside, cp1, cp2, (x0, y0, z0))
    
    rp = camera_utils.map_world_to_camera_coords(np.array((x0, y0, z0), np.float32),
                                                 state)
    x, y = rp
    cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), 2)


    cb_world_coords = camera_utils.map_camera_to_world_coords(state["cb"],
                                                              state,
                                                              z=0.0)
    #print(cb_world_coords[0],
    #      cb_world_coords[6],
    #      cb_world_coords[7*6],
    #      cb_world_coords[7*7-1])
    #x, y, w, h = get_chessboard_bounding_rect(state["cb"])

def is_tracking(tracked_object_buffer):
    if len(tracked_object_buffer) < 20:
        return False
    found_count = 0
    for i in range(1, 10):
        if all(tracked_object_buffer[-i]):
            found_count += 1
    return found_count >= 5

def is_lost(tracked_object_buffer):
    if len(tracked_object_buffer) < 20:
        print("early exit")
        return True
    found_count = 0
    for i in range(1, 10):
        if all(tracked_object_buffer[-i]):
            found_count += 1
    return found_count == 0

def test_geometry(state):
    """
    Using the calibrated camera transformations, can we recreate that the
    
    """
    # z=0.0 because corner points are assumed to be on the ground.
    cb_world_coords = camera_utils.map_camera_to_world_coords(state["cb"],
                                                              state,
                                                              z=0.0)

    # Testing if the central cross lines of a chessboard are perpindicular.
    v1 = (cb_world_coords[21] - cb_world_coords[27]).flatten()
    v2 = (cb_world_coords[4] - cb_world_coords[32]).flatten()

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    print("Geometry eval: orthogonality (degrees) = {:.2f}".format(np.rad2deg(np.arccos(np.dot(v1, v2)))))

def save_video(state):
    frames = state["frame_buffer"]
    
    output_file = 'tracked-{}.mp4'.format(datetime.now().strftime("%Y-%m-%dT%H%M%S.sss"))
    fps = state["frame_rate"]
    frame_size = frames[0].shape[:-1][::-1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_file, fourcc, state["frame_rate"], frame_size)

    for f in frames:
        out.write(f)
    out.release()

    print("video saved to disk: {}".format(output_file))

cv2.namedWindow("testing")
vc = cv2.VideoCapture(0)

if not vc.isOpened():
    raise "Failed to open camera"

state = {}
state["timer"] = time.time()
state["frame_count"] = 0
state["state"] = "FINDING_CHESSBOARD"
state["cb"] = None
state["frame_rate"] = 0.0
state["chessboard_buffer"] = []
state["median_channels"] = {}
state["background_mass"] = 0.0
state["frame_buffer"] = deque(maxlen=500)
state["tracked_object_buffer"] = deque(maxlen=500)
state["tracked_timestamp_buffer"] = deque(maxlen=500)
state["searching_start_time"] = None
state["tracking_start_time"] = None
state["show_result_start_time"] = None


while True:
    rval, frame = vc.read()
    if not rval:
        break
    state["frame_count"] += 1
    state["frame_dimensions"] = frame.shape[:-1]

    if state["state"] == "FINDING_CHESSBOARD":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cb_found, cb = camera_utils.find_chessboard(gray)
        if cb_found:
            state["cb"] = cb
            frame = cv2.drawChessboardCorners(frame, (7,7), cb, patternWasFound=True)
            state["state"] = "CHESSBOARD_FOUND"
    elif state["state"] == "CHESSBOARD_FOUND":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("debug", get_chessboard_subimage(frame, state["cb"]))
        cv2.waitKey(1)
        
        #frame = cv2.drawChessboardCorners(frame, (7,7), state["cb"], patternWasFound=True)
        state["chessboard_buffer"].append(get_chessboard_subimage(frame, state["cb"]))
        if len(state["chessboard_buffer"]) == 20:
            # TODO: determine if the chessboard moved in the frame
            # This can be done by checking the area of intersecting convex hulls.
            ml, ma, mb = get_median_pixels(state["chessboard_buffer"])
            state["median_channels"]["ml"] = ml
            state["median_channels"]["ma"] = ma
            state["median_channels"]["mb"] = mb

            state["background_mass"] = get_background_diff(state["chessboard_buffer"],
                                                           state["median_channels"])
            
            print("background mass", np.log(state["background_mass"]))

            camera_params = camera_utils.calibrate_camera(gray, cb)
            state["mtx"] = camera_params["mtx"]
            state["dist"] = camera_params["dist"]
            state["rvecs"] = camera_params["rvecs"]
            state["tvecs"] = camera_params["tvecs"]
            test_geometry(state)

            state["chessboard_buffer"].clear()
            state["state"] = "FINDING_OBJECTS"
            state["searching_start_time"] = time.time()
    elif state["state"] == "FINDING_OBJECTS":
        #frame = cv2.drawChessboardCorners(frame, (7,7), cb, patternWasFound=True)
        diffed = diff_against_median(get_chessboard_subimage(frame, state["cb"]),
                                     state["median_channels"])
        cX, cY = find_golfball(diffed, state["background_mass"])
        if all((cX, cY)): #cX is not None and cY is not None:
            x, y, _, _ = get_chessboard_bounding_rect(state["cb"])
            state["tracked_object_buffer"].append((x+cX, y+cY))
            state["tracked_timestamp_buffer"].append(time.time())
            cv2.circle(frame, (x+cX, y+cY), 20, (255), 2)
        else:
            state["tracked_object_buffer"].append((None, None))
            state["tracked_timestamp_buffer"].append(time.time())
        state["frame_buffer"].append(frame.copy())

        if is_tracking(state["tracked_object_buffer"]):
            state["state"] = "TRACKING_OBJECT"
            state["tracking_start_time"] = time.time()

        if time.time() - state["searching_start_time"] >= 10.0:
            state["state"] = "FINDING_CHESSBOARD"
            state["searching_start_time"] = None
            state["tracked_object_buffer"].clear()
            state["tracked_timestamp_buffer"].clear()
            state["frame_buffer"].clear()
    elif state["state"] == "TRACKING_OBJECT":
        #frame = cv2.drawChessboardCorners(frame, (7,7), state["cb"], patternWasFound=True)
        diffed = diff_against_median(get_chessboard_subimage(frame, state["cb"]),
                                     state["median_channels"])
        cX, cY = find_golfball(diffed, state["background_mass"])
        if all((cX, cY)):
            x, y, _, _ = get_chessboard_bounding_rect(state["cb"])
            state["tracked_object_buffer"].append((x+cX, y+cY))
            state["tracked_timestamp_buffer"].append(time.time())
            cv2.circle(frame, (x+cX, y+cY), 20, (255), 2)
        else:
            state["tracked_object_buffer"].append((None, None))
            state["tracked_timestamp_buffer"].append(time.time())

        state["frame_buffer"].append(frame.copy())
            
        if is_lost(state["tracked_object_buffer"]):
            print("tracked frames: {}".format(len([x for x in state["tracked_object_buffer"] if all(x)])))
            state["state"] = "SHOWING_RESULT"
            state["show_result_start_time"] = time.time()
            estimate_path(state)
            print("Distance eval (feet):  {:.2f}".format(
                golf_utils.velocity_to_distance(estimate_velocity(state)))
            )

        if time.time() - state["tracking_start_time"] >= 3.0:
            # Something wrong happened. There are no results to show.
            state["state"] = "FINDING_CHESSBOARD"
            state["searching_start_time"] = None
            state["tracked_object_buffer"].clear()
            state["tracked_timestamp_buffer"].clear()
            state["frame_buffer"].clear()
    elif state["state"] == "SHOWING_RESULT":
        draw_result(frame, state)
        state["frame_buffer"].append(frame.copy())
        if time.time() - state["show_result_start_time"] >= 5.0:
            save_video(state)
            state["state"] = "FINDING_CHESSBOARD"
            state["tracked_object_buffer"].clear()
            state["tracked_timestamp_buffer"].clear()
            state["frame_buffer"].clear()
            
    write_text(frame, debug_string(state))

    cv2.imshow("testing", frame)
    key = cv2.waitKey(1)
    
cv2.destroyAllWindows()
    
