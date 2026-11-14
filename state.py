import cv2
import numpy as np
import time

from collections import deque
from enum import Enum

import camera_utils
import chessboard_utils
import eval_utils
import frame_annotator
import geometry_utils
import golf_utils
import tracking_utils
import tts_utils
import params

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

class TrackingState(Enum):
    FINDING_CHESSBOARD = 1
    CHESSBOARD_FOUND = 2
    INIT_ROI = 3
    FINDING_OBJECTS = 4
    TRACKING_OBJECT = 5
    SHOWING_RESULT = 6

class Tracker:
    def __init__(self, voice_assistant):
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

        # Median pixel values taken from the chessboard buffer
        self.median_channels = {}

        # Background moment m00. A significant change in m00 signals an object
        # on the chessboard.
        self.background_mass = None

        self.camera_params = {}

        # Timestamp when searching for objects began.
        self.search_start_time = None
        self.tracked_frame_buffer = deque(maxlen=500)

        self.tracking_start_time = None

        self.show_result_start_time = None

        self.recent_results = deque(maxlen=3)

        self.voice_assistant = voice_assistant

        # Borders of the putting surfaced, determined with RANSAC sampling.
        self.surface_lines = None
        self.surface_roi = None
        self.roi_buffer = []

    def reset(self):
        self.state = TrackingState.FINDING_CHESSBOARD
        self.cb = None
        self.median_channels = {}
        self.background_mass = None
        self.camera_params = {}
        self.search_start_time = None
        self.tracked_frame_buffer.clear()
        self.tracking_start_time = None
        self.show_result_start_time = None
        self.surface_lines = None
        self.surface_roi = None
        self.roi_buffer.clear()
        
    def find_chessboard(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cb_found, cb = camera_utils.find_chessboard(gray)
        if not cb_found:
            return

        self.cb = cb
        self.camera_params = camera_utils.calibrate_camera(gray, cb)
        self.state = TrackingState.CHESSBOARD_FOUND

    def find_surface(self, frame, t):
        lower_dark_blue = np.array([100, 50, 20])
        upper_dark_blue = np.array([135, 255, 180])

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        color_mask = cv2.inRange(hsv, lower_dark_blue, upper_dark_blue)
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        #cv2.imshow("surface", cleaned_mask)
        
        line1, line2, inliers = camera_utils.fit_neoprene_mask_ransac(
            cleaned_mask,
            self.camera_params,
            params.SURFACE_WIDTH_MM,
        )

        self.surface_lines = (line1, line2)
        self.surface_roi = tuple(camera_utils.get_neoprene_roi(line1, line2, 300.0))
        self.state = TrackingState.INIT_ROI

    def init_roi(self, frame, t):
        roi_x, roi_y, roi_w, roi_h = camera_utils.get_roi_rect(self.surface_roi,
                                                               self.camera_params)
        roi_subframe = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w, :]
        #cv2.imshow("sub-image", roi_subframe)
        self.roi_buffer.append(roi_subframe)

        if len(self.roi_buffer) == 10:
            ml, ma, mb = tracking_utils.get_median_pixels(self.roi_buffer)
            self.median_channels["ml"] = ml
            self.median_channels["ma"] = ma
            self.median_channels["mb"] = mb
            self.background_mass = tracking_utils.get_background_diff(self.roi_buffer,
                                                                      self.median_channels)
            print("background mass", np.log(self.background_mass))

            self.search_start_time = t
            self.state = TrackingState.FINDING_OBJECTS

    def find_objects(self, frame, t):
        roi_x, roi_y, roi_w, roi_h = camera_utils.get_roi_rect(self.surface_roi,
                                                               self.camera_params)
        roi_subframe = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w, :]
        diffed = tracking_utils.diff_against_median(
            roi_subframe,
            self.median_channels
        )
        #max_val = np.max(diffed)
        #if max_val > 1e-6:
        #    normalized_diff = (diffed / max_val) * 255.0
        #else:
        #    normalized_diff = diffed
        #vis_img = normalized_diff.astype(np.uint8)
        #cv2.imshow("diffed", vis_img)
        
        cX, cY = tracking_utils.find_golfball(diffed, self.background_mass)
        if all((cX, cY)):
            self.tracked_frame_buffer.append((frame.copy(), t, (roi_x+cX, roi_y+cY)))
        else:
            self.tracked_frame_buffer.append((frame.copy(), t, (None, None)))

        if is_tracking(self.tracked_frame_buffer):
            self.state = TrackingState.TRACKING_OBJECT
            self.tracking_start_time = t

        # Give up. Restart from the beginning state.
        if t - self.search_start_time >= 20.0:
            self.reset()

    def track_object(self, frame, t):
        roi_x, roi_y, roi_w, roi_h = camera_utils.get_roi_rect(self.surface_roi,
                                                               self.camera_params)
        roi_subframe = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w, :]
        diffed = tracking_utils.diff_against_median(
            roi_subframe,
            self.median_channels
        )
        #max_val = np.max(diffed)
        #if max_val > 1e-6:
        #    normalized_diff = (diffed / max_val) * 255.0
        #else:
        #    normalized_diff = diffed
        #vis_img = normalized_diff.astype(np.uint8)
        #cv2.imshow("diffed", vis_img)
        cX, cY = tracking_utils.find_golfball(diffed, self.background_mass)
        if all((cX, cY)):
            self.tracked_frame_buffer.append((frame.copy(), t, (roi_x+cX, roi_y+cY)))
        else:
            self.tracked_frame_buffer.append((frame.copy(), t, (None, None)))

        if is_lost(self.tracked_frame_buffer):
            if not eval_utils.is_linear_trajectory(self.tracked_frame_buffer):
                # Noise unrelated to a golf ball (like a person walking in frame).
                self.reset()
                return
                
            
            self.state = TrackingState.SHOWING_RESULT
            self.show_result_start_time = t

            pixel_path = eval_utils.get_pixel_path(self.tracked_frame_buffer)
            estimated_distance = eval_utils.evaluate_distance(self.tracked_frame_buffer,
                                                              self.camera_params)
            self.voice_assistant.speak("Distance eval: {:.2f} feet".format(estimated_distance))
            estimated_direction, direction_result = eval_utils.evaluate_path(
                geometry_utils.middle_line(self.surface_lines[0], self.surface_lines[1]),
                pixel_path,
                self.camera_params)
            if estimated_direction < 1.0:
                self.voice_assistant.speak(
                    "Path eval: {:.2f} degrees".format(estimated_direction)
                )
            else:
                self.voice_assistant.speak(
                    "Path eval: {}, {:.2f} degrees".format(direction_result,
                                                           estimated_direction)
                )
            
            self.recent_results.appendleft(
                "Distance eval (feet): {:.2f} :: Path eval (degrees): {:.2f}".format(
                    estimated_distance,
                    estimated_direction
                )
            )

        if t - self.tracking_start_time >= 3.0:
            # There's no good reason to track for this amount of time.
            # Reset to the beginning state.
            self.reset()

    def show_result(self, frame, t):
        self.tracked_frame_buffer.append((frame.copy(), t, (None, None)))
        if t - self.show_result_start_time >= 6.0:
            # Reset for the next putt.
            self.reset()

    def process_for_state(self, frame, t):
         match self.state:
            case TrackingState.FINDING_CHESSBOARD:
                self.find_chessboard(frame)
                return
            case TrackingState.CHESSBOARD_FOUND:
                self.find_surface(frame, t)
                return
            case TrackingState.INIT_ROI:
                self.init_roi(frame, t)
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

    def process(self, frame, t):
        self.frame_count += 1
        self.process_for_state(frame, t)
        self.update_debug_stats(t)
        return frame_annotator.annotate(frame.copy(), t, self)
