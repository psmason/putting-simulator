import cv2
import numpy as np

from enum import Enum

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
  if M["m00"] < 100.0 * background_mass:
      return None, None
  cX = int(M["m10"] / M["m00"])
  cY = int(M["m01"] / M["m00"])
  return cX, cY

class GolfBallTrackerState(Enum):
  NOT_READY = 0
  READY = 1

class GolfBallTracker:
  def __init__(self):
    self.roi_buffer = []
    self.median_channels = {}
    # Background moment m00. A significant change in m00 signals a golf ball in the
    # region of interest.
    self.background_mass = None

  def init(self, roi):
    #cv2.imshow("sub-image", roi)
    
    self.roi_buffer.append(roi)
    if len(self.roi_buffer) == 10:
      ml, ma, mb = get_median_pixels(self.roi_buffer)
      self.median_channels["ml"] = ml
      self.median_channels["ma"] = ma
      self.median_channels["mb"] = mb
      self.background_mass = get_background_diff(self.roi_buffer,
                                                 self.median_channels)
      print("background mass", np.log(self.background_mass))
      return GolfBallTrackerState.READY
    return GolfBallTrackerState.NOT_READY

  def find_golfball(self, roi):
    diffed = diff_against_median(roi, self.median_channels)
    #max_val = np.max(diffed)
    #if max_val > 1e-6:
    #    normalized_diff = (diffed / max_val) * 255.0
    #else:
    #    normalized_diff = diffed
    #vis_img = normalized_diff.astype(np.uint8)
    #cv2.imshow("diffed", vis_img)
    return find_golfball(diffed, self.background_mass)

  
    
