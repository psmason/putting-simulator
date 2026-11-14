import cv2
import numpy as np

from datetime import datetime

import camera_utils
import eval_utils
import geometry_utils
import params

def write_text(frame, text, coordinates):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)  # Green color in BGR format
    thickness = 2
    line_type = cv2.LINE_AA  # Smooth anti-aliased line type
    cv2.putText(frame, text, coordinates, font, font_scale, color, thickness, line_type)

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

def draw_world_line(frame,
                    camera_params,
                    line_params,
                    length_mm=1219.2,
                    color=(0, 255, 0),
                    thickness=2):
    a, b, d = line_params

    v = np.array([-b, a, 0.0], dtype=np.float32)
    v /= np.linalg.norm(v)

    if abs(a) > 1e-5:
        p_on_line = np.array([-d / a, 0.0, 0.0], dtype=np.float32)
    else:
        p_on_line = np.array([0.0, -d / b, 0.0], dtype=np.float32)

    p1_world = p_on_line - v * (length_mm / 2.0)
    p2_world = p_on_line + v * (length_mm / 2.0)

    pt1_3d = np.array([p1_world], dtype=np.float32)
    pt2_3d = np.array([p2_world], dtype=np.float32)

    pt1_2d = camera_utils.map_world_to_camera_coords(pt1_3d, camera_params)
    pt2_2d = camera_utils.map_world_to_camera_coords(pt2_3d, camera_params)

    cv2.line(
        frame,
        tuple(pt1_2d[0].astype(int)),
        tuple(pt2_2d[0].astype(int)),
        color,
        thickness,
        cv2.LINE_AA,
    )

def annotate(frame, timestamp, state):     
    height, width, _ = frame.shape
    cbrows, cbcols = params.CHESSBOARD_TILES
        
    write_text(frame, "Recent putts:", (100, int(height * 0.1)))
    for i, result in enumerate(state.recent_results):
        write_text(frame,
                   "{:d}) {}".format(i+1, result),
                   (100, int(height * (0.1 + 0.05 * (i+1)))))

    # Draw detected chessboard onto the frame.
    if state.cb is not None:
        frame = cv2.drawChessboardCorners(frame, params.CHESSBOARD_TILES, state.cb, patternWasFound=True)
        camera_utils.draw_world_grid_overlay(frame,
                                             state.camera_params,
                                             params.CHESSBOARD_TILE_LENGTH)

    # Draw putting surface edges
    if state.surface_lines is not None:                
        draw_world_line(frame, state.camera_params, state.surface_lines[0], 2000)
        draw_world_line(frame, state.camera_params, state.surface_lines[1], 2000)
        draw_world_line(
            frame,
            state.camera_params,
            geometry_utils.middle_line(state.surface_lines[0], state.surface_lines[1]),
            length_mm=2000, 
            color=(0, 0, 255), # Red
            thickness=1
        )

    # Draw the ROI corners
    if state.surface_roi is not None:
        for c in state.surface_roi:
            x, y = camera_utils.map_world_to_camera_coords(np.array(c, np.float32),
                                                         state.camera_params)
            cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), 2)

    # Draw detected golfball positions.
    for i in range(len(state.tracked_frame_buffer)):
        _, _, p = state.tracked_frame_buffer[i]
        if all(p):
            cv2.circle(frame, (p[0], p[1]), 20, (255), 2)

    if state.state.name == "SHOWING_RESULT":
        pixel_path = eval_utils.get_pixel_path(state.tracked_frame_buffer)
                     
        # Drawing the fitted line onto the frame.
        vx, vy, vz, x0, y0, z0 = camera_utils.fit_tracked_path(pixel_path,
                                                               state.camera_params)
        s = 20*params.CHESSBOARD_TILE_LENGTH 
        p1 = (int(x0[0] + s*vx[0]), int(y0[0] + s*vy[0]))
        p2 = (int(x0[0] - s*vx[0]), int(y0[0] - s*vy[0]))
    
        inside, cp1, cp2 = cv2.clipLine(world_rect_to_bounds(state.surface_roi), p1, p2)
        if inside:
            cp1 = camera_utils.map_world_to_camera_coords(
                np.array((cp1[0],
                          cp1[1],
                          camera_utils.golfball_perceived_center()), np.float32),
                state.camera_params)
            cp2 = camera_utils.map_world_to_camera_coords(
                np.array((cp2[0],
                          cp2[1],
                          camera_utils.golfball_perceived_center()), np.float32),
                state.camera_params)
            cv2.line(frame,
                     (int(cp1[0]), int(cp1[1])),
                     (int(cp2[0]), int(cp2[1])),
                     (0, 255, 0),
                     thickness=2)

        # Draw the fitted center onto the frame.
        rp = camera_utils.map_world_to_camera_coords(np.array((x0, y0, z0), np.float32),
                                                     state.camera_params)
        x, y = rp.ravel()[:2] # TODO, fix this
        cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), 2)

    # Bottom left for debugging information.
    debug_coords = (100, int(height * 0.9))
    write_text(frame,
               "state: {} :: frame rate: {:.2f}".format(
                   state.state,
                   state.frame_rate),
               debug_coords)
    timestamp_coords = (100, int(height * 0.95))
    write_text(frame,
               "t: {}".format(datetime.fromtimestamp(timestamp).strftime("%Y-%m-%dT%H%M%S.%f")),
               timestamp_coords)

    return frame
