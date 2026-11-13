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
import tracking_utils
import tts_utils
import params

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
    return found_count >= 60

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
            if is_noise(self.tracked_frame_buffer):
                print("found noise. resetting")
                self.state = TrackingState.FINDING_CHESSBOARD
                self.reset()
                return
            
            self.state = TrackingState.SHOWING_RESULT
            self.show_result_start_time = t

            pixel_path = get_pixel_path(self.tracked_frame_buffer)
            estimated_distance = eval_utils.evaluate_distance(self.tracked_frame_buffer,
                                                              self.camera_params)
            self.voice_assistant.speak("Distance eval: {:.2f} feet".format(estimated_distance))
            def middle_line(l1, l2):
                a1, b1, d1 = l1
                a2, b2, d2 = l2

                am = (a1 + a2) / 2.0
                bm = (b1 + b2) / 2.0
                dm = (d1 + d2) / 2.0
                return am, bm, dm
            estimated_direction, direction_result = eval_utils.evaluate_path(
                middle_line(self.surface_lines[0], self.surface_lines[1]),
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
        if t - self.show_result_start_time >= 3.0:
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

    def draw_world_line(self,
                        frame,
                        line_params,
                        length_mm=1219.2,
                        color=(0, 255, 0),
                        thickness=2,
    ):
        """Projects a 3D world line back to 2D image space.
        """
        a, b, d = line_params

        # 1. Compute unit direction vector along the line
        v = np.array([-b, a, 0.0], dtype=np.float32)
        v /= np.linalg.norm(v)

        # 2. Find a base point on the line at z=0
        if abs(a) > 1e-5:
            p_on_line = np.array([-d / a, 0.0, 0.0], dtype=np.float32)
        else:
            p_on_line = np.array([0.0, -d / b, 0.0], dtype=np.float32)

        # 3. Define endpoints spanning the 4-foot length
        p1_world = p_on_line - v * (length_mm / 2.0)
        p2_world = p_on_line + v * (length_mm / 2.0)

        # Format as (1, 3) array for your utility function
        pt1_3d = np.array([p1_world], dtype=np.float32)
        pt2_3d = np.array([p2_world], dtype=np.float32)

        # 4. Map back to 2D pixels using your existing function
        pt1_2d = camera_utils.map_world_to_camera_coords(pt1_3d, self.camera_params)
        pt2_2d = camera_utils.map_world_to_camera_coords(pt2_3d, self.camera_params)

        # 5. Draw
        cv2.line(
            frame,
            tuple(pt1_2d[0].astype(int)),
            tuple(pt2_2d[0].astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )

    def annotate(self, frame, timestamp):        
        height, width, _ = frame.shape
        cbrows, cbcols = params.CHESSBOARD_TILES
        
        write_text(frame, "Recent putts:", (100, int(height * 0.1)))
        for i, result in enumerate(self.recent_results):
            write_text(frame,
                       "{:d}) {}".format(i+1, result),
                       (100, int(height * (0.1 + 0.05 * (i+1)))))

        # Draw detected chessboard onto the frame.
        if self.cb is not None:
            frame = cv2.drawChessboardCorners(frame, params.CHESSBOARD_TILES, self.cb, patternWasFound=True)
            camera_utils.draw_world_grid_overlay(frame,
                                                 self.camera_params,
                                                 params.CHESSBOARD_TILE_LENGTH)

        # Draw putting surface edges
        if self.surface_lines is not None:
            def middle_line(l1, l2):
                a1, b1, d1 = l1
                a2, b2, d2 = l2

                am = (a1 + a2) / 2.0
                bm = (b1 + b2) / 2.0
                dm = (d1 + d2) / 2.0
                return am, bm, dm
                
            self.draw_world_line(frame, self.surface_lines[0], 2000)
            self.draw_world_line(frame, self.surface_lines[1], 2000)
            self.draw_world_line(
                frame, 
                middle_line(self.surface_lines[0], self.surface_lines[1]),
                length_mm=2000, 
                color=(0, 0, 255),    # Red
                thickness=1
            )

        # Draw the ROI corners
        if self.surface_roi is not None:
            for c in self.surface_roi:
                cp = camera_utils.map_world_to_camera_coords(np.array(c, np.float32),
                                                              self.camera_params)
                x, y = cp
                cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), 2)

        # Draw detected golfball positions.
        path_indices = []
        for i in range(len(self.tracked_frame_buffer)):
            _, _, p = self.tracked_frame_buffer[i]
            if all(p):
                cv2.circle(frame, (p[0], p[1]), 20, (255), 2)
                path_indices.append(i)

        if self.state == TrackingState.SHOWING_RESULT:
            pixel_path = get_pixel_path(self.tracked_frame_buffer)
                     
            # Drawing the fitted line onto the frame.
            vx, vy, vz, x0, y0, z0 = camera_utils.fit_tracked_path(pixel_path,
                                                                   self.camera_params)
            s = 20*params.CHESSBOARD_TILE_LENGTH 
            p1 = (int(x0[0] + s*vx[0]), int(y0[0] + s*vy[0]))
            p2 = (int(x0[0] - s*vx[0]), int(y0[0] - s*vy[0]))

            def world_rect_to_bounds(world_corners):
                corners = np.array(world_corners, dtype=np.float32)
                
                x_coords = corners[:, 0]
                y_coords = corners[:, 1]
    
                # Cast bounds explicitly to standard Python int
                x_min = int(np.floor(np.min(x_coords)))
                x_max = int(np.ceil(np.max(x_coords)))
                y_min = int(np.floor(np.min(y_coords)))
                y_max = int(np.ceil(np.max(y_coords)))
    
                w = x_max - x_min
                h = y_max - y_min
    
                # Return as a tuple of Python integers representing (x, y, w, h)
                return (x_min, y_min, w, h)
            
            inside, cp1, cp2 = cv2.clipLine(world_rect_to_bounds(self.surface_roi), p1, p2)
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
            x, y = rp.ravel()[:2] # TODO, fix this
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
