import cv2
import numpy as np
import time

from collections import deque
from datetime import datetime
from enum import Enum

import camera_utils
import chessboard_utils
import eval_utils
import golf_utils

def write_text(frame, text, coordinates):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)  # Green color in BGR format
    thickness = 2
    line_type = cv2.LINE_AA  # Smooth anti-aliased line type
    cv2.putText(frame, text, coordinates, font, font_scale, color, thickness, line_type)

def is_tracking(tracked_frame_buffer):
    if len(tracked_frame_buffer) < 20:
        return False
    found_count = 0
    for i in range(1, 10):
        _, _, p = tracked_frame_buffer[-i]
        if all(p):
            found_count += 1
    return found_count >= 3

def is_lost(tracked_frame_buffer):
    if len(tracked_frame_buffer) < 20:
        return True
    found_count = 0
    for i in range(1, 10):
        _, _, p = tracked_frame_buffer[-i]
        if all(p):
            found_count += 1
    return found_count == 0

def is_noise(tracked_frame_buffer):
    found_count = 0
    for i in range(len(tracked_frame_buffer)):
        _, _, p = tracked_frame_buffer[i]
        if all (p):
            found_count += 1
    return found_count >= 30

def get_pixel_path(tracking_buffer):
    path = []
    for i in range(len(tracking_buffer)):
        _, _, p = tracking_buffer[i]
        if all(p):
            path.append(p)
    return np.array(path, np.float32)

class TrackingState(Enum):
    FINDING_CHESSBOARD = 1
    CHESSBOARD_FOUND = 2
    FINDING_OBJECTS = 3
    TRACKING_OBJECT = 4
    SHOWING_RESULT = 5

class Tracker:
    def __init__(self, save_recording=False):
        # Processing frame rate
        self.frame_rate = 0.0

        # Stopwatch used to determine frame rate
        self.debug_stopwatch = None

        # Frame count used to determine frame rate
        self.frame_count = 0

        # Initial tracker state
        self.state = TrackingState.FINDING_CHESSBOARD

        # Chessboard positions produced by the CHESSBOARD_FOUND state
        self.cb = None

        # Buffer is filled when initializing chessboard for object detection
        self.cb_buffer = []

        # Median pixel values taken from the chessboard buffer
        self.median_channels = {}

        # Background moment m00. A significant change in m00 signals an object
        # on the chessboard.
        self.background_mass = None

        self.camera_params = {}

        # Timestamp when searching for objects began.
        self.search_start_time = None
        #self.tracked_object_buffer = deque(maxlen=500)
        #self.tracked_timestamp_buffer = deque(maxlen=500)
        self.tracked_frame_buffer = deque(maxlen=500)

        self.tracking_start_time = None

        self.show_result_start_time = None

        self.recent_results = deque(maxlen=3)

    def reset(self):
        self.state = TrackingState.FINDING_CHESSBOARD
        self.cb = None
        self.cb_buffer.clear()
        self.median_channels = {}
        self.background_mass = None
        self.camera_params = {}
        self.search_start_time = None
        #self.tracked_object_buffer.clear()
        #self.tracked_timestamp_buffer.clear()
        self.tracked_frame_buffer.clear()
        self.tracking_start_time = None
        self.show_result_start_time = None
        
    def find_chessboard(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cb_found, cb = camera_utils.find_chessboard(gray)
        if not cb_found:
            return

        self.cb = cb
        self.camera_params = camera_utils.calibrate_camera(gray, cb)
        self.state = TrackingState.CHESSBOARD_FOUND

    def initialize_chessboard(self, frame, t):
        cb_frame = chessboard_utils.get_chessboard_subimage(frame, self.cb)
        self.cb_buffer.append(
            chessboard_utils.get_chessboard_subimage(frame, self.cb)
        )

        if len(self.cb_buffer) == 10:
            ml, ma, mb = chessboard_utils.get_median_pixels(self.cb_buffer)
            self.median_channels["ml"] = ml
            self.median_channels["ma"] = ma
            self.median_channels["mb"] = mb
            self.background_mass = chessboard_utils.get_background_diff(self.cb_buffer,
                                                                        self.median_channels)
            print("background mass", np.log(self.background_mass))

            self.cb_buffer.clear()
            self.search_start_time = t
            self.state = TrackingState.FINDING_OBJECTS

    def find_objects(self, frame, t):
        diffed = chessboard_utils.diff_against_median(
            chessboard_utils.get_chessboard_subimage(frame, self.cb),
            self.median_channels
        )
        cX, cY = chessboard_utils.find_golfball(diffed, self.background_mass)
        if all((cX, cY)):
            x, y, _, _ = chessboard_utils.get_chessboard_bounding_rect(self.cb)
            self.tracked_frame_buffer.append((frame.copy(), t, (x+cX, y+cY)))
        else:
            self.tracked_frame_buffer.append((frame.copy(), t, (None, None)))

        if is_tracking(self.tracked_frame_buffer):
            self.state = TrackingState.TRACKING_OBJECT
            self.tracking_start_time = t

        # Give up. Restart from the beginning state.
        if t - self.search_start_time >= 10.0:
            self.reset()

    def track_object(self, frame, t):
        diffed = chessboard_utils.diff_against_median(
            chessboard_utils.get_chessboard_subimage(frame, self.cb),
            self.median_channels
        )
        cX, cY = chessboard_utils.find_golfball(diffed, self.background_mass)
        if all((cX, cY)):
            x, y, _, _ = chessboard_utils.get_chessboard_bounding_rect(self.cb)
            self.tracked_frame_buffer.append((frame.copy(), t, (x+cX, y+cY)))
        else:
            self.tracked_frame_buffer.append((frame.copy(), t, (None, None)))

        if is_lost(self.tracked_frame_buffer):
            if is_noise(self.tracked_frame_buffer):
                self.state = TrackingState.FINDING_CHESSBOARD
                self.reset()
                return
            
            self.state = TrackingState.SHOWING_RESULT
            self.show_result_start_time = t

            pixel_path = get_pixel_path(self.tracked_frame_buffer)
            self.recent_results.appendleft(
                "Distance eval v7 (feet): {:.2f} :: Path eval (degrees): {:.2f}".format(
                    eval_utils.evaluate_distance_v7(self.tracked_frame_buffer,
                                                    self.camera_params),
                    eval_utils.evaluate_path(self.cb,
                                             pixel_path,
                                             self.camera_params)
                )
            )

            print("Distance eval v1 (feet): {:.2f}".format(
                eval_utils.evaluate_distance(self.tracked_frame_buffer,
                                             self.camera_params)
            ))
            print("Distance eval v2 (feet): {:.2f}".format(
                eval_utils.evaluate_distance_v2(self.tracked_frame_buffer,
                                                self.camera_params)
            ))
            print("Distance eval v3 (feet): {:.2f}".format(
                eval_utils.evaluate_distance_v3(self.tracked_frame_buffer,
                                                self.camera_params)
            ))
            print("Distance eval v4 (feet): {:.2f}".format(
                eval_utils.evaluate_distance_v4(self.tracked_frame_buffer,
                                                self.camera_params)
            ))
            print("Distance eval v5 (feet): {:.2f}".format(
                eval_utils.evaluate_distance_v5(self.tracked_frame_buffer,
                                                self.camera_params)
            ))
            print("Distance eval v6 (feet): {:.2f}".format(
                eval_utils.evaluate_distance_v6(self.tracked_frame_buffer,
                                                self.camera_params)
            ))
            print("Distance eval v7 (feet): {:.2f}".format(
                eval_utils.evaluate_distance_v7(self.tracked_frame_buffer,
                                                self.camera_params)
            ))


        if t - self.tracking_start_time >= 3.0:
            # There's no good reason to track for this amount of time.
            # Reset to the beginning state.
            self.reset()

    def show_result(self, frame, t):
        self.tracked_frame_buffer.append((frame.copy(), t, (None, None)))
        if t - self.show_result_start_time >= 3.0:
            # Reset for the next putt.
            self.reset()

    def process_for_state(self, frame, t):
         match self.state:
            case TrackingState.FINDING_CHESSBOARD:
                self.find_chessboard(frame)
                return
            case TrackingState.CHESSBOARD_FOUND:
                self.initialize_chessboard(frame, t)
                return
            case TrackingState.FINDING_OBJECTS:
                self.find_objects(frame, t)
                return
            case TrackingState.TRACKING_OBJECT:
                self.track_object(frame, t)
                return
            case TrackingState.SHOWING_RESULT:
                self.show_result(frame, t)
                return
                
    def update_debug_stats(self, t):
        if self.debug_stopwatch is None:
            self.debug_stopwatch = t
            return 
        time_delta = t - self.debug_stopwatch
        if time_delta >= 1.0:
            self.frame_rate = 1.0 * self.frame_count / time_delta
            self.debug_stopwatch = t
            self.frame_count = 0       

    def annotate(self, frame, timestamp):        
        height, width, _ = frame.shape

        write_text(frame, "Recent putts:", (100, int(height * 0.1)))
        for i, result in enumerate(self.recent_results):
            write_text(frame,
                       "{:d}) {}".format(i+1, result),
                       (100, int(height * (0.1 + 0.05 * (i+1)))))

        # Draw detected chessboard onto the frame.
        if self.cb is not None:
            frame = cv2.drawChessboardCorners(frame, (7,7), self.cb, patternWasFound=True)

        # Draw detected golfball positions.
        path_indices = []
        for i in range(len(self.tracked_frame_buffer)):
            _, _, p = self.tracked_frame_buffer[i]
            if all(p):
                cv2.circle(frame, (p[0], p[1]), 20, (255), 2)
                path_indices.append(i)

        if self.state == TrackingState.SHOWING_RESULT:
            pixel_path = get_pixel_path(self.tracked_frame_buffer)
            #pixel_path = np.array([x for x in self.tracked_object_buffer if all(x)],
            #                      np.float32)
                     
            # Drawing the fitted line onto the frame.
            vx, vy, vz, x0, y0, z0 = camera_utils.fit_tracked_path(pixel_path,
                                                                   self.camera_params)
            cb_world_box = (0, 0, int(50.8*7), int(50.8*7))
            s = 10*50.8
            p1 = (int(x0[0] + s*vx[0]), int(y0[0] + s*vy[0]))
            p2 = (int(x0[0] - s*vx[0]), int(y0[0] - s*vy[0]))
            inside, cp1, cp2 = cv2.clipLine(cb_world_box, p1, p2)
            if inside:
                cp1 = camera_utils.map_world_to_camera_coords(
                    np.array((cp1[0],
                              cp1[1],
                              camera_utils.golfball_perceived_center()), np.float32),
                    self.camera_params)
                cp2 = camera_utils.map_world_to_camera_coords(
                    np.array((cp2[0],
                              cp2[1],
                              camera_utils.golfball_perceived_center()), np.float32),
                    self.camera_params)
                cv2.line(frame,
                         (int(cp1[0]), int(cp1[1])),
                         (int(cp2[0]), int(cp2[1])),
                         (0, 255, 0),
                         thickness=2)

            # Draw the fitted center onto the frame.
            rp = camera_utils.map_world_to_camera_coords(np.array((x0, y0, z0), np.float32),
                                                         self.camera_params)
            x, y = rp
            cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), 2)

        # Bottom left for debugging information.
        debug_coords = (100, int(height * 0.9))
        write_text(frame,
                   "state: {} :: frame rate: {:.2f}".format(
                       self.state,
                       self.frame_rate),
                   debug_coords)
        timestamp_coords = (100, int(height * 0.95))
        write_text(frame,
                   "t: {}".format(datetime.fromtimestamp(timestamp).strftime("%Y-%m-%dT%H%M%S.%f")),
                   timestamp_coords)

        return frame

    def process(self, frame, t):
        self.frame_count += 1
        self.process_for_state(frame, t)
        self.update_debug_stats(t)
        return self.annotate(frame.copy(), t)
