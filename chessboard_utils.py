import cv2
import numpy as np

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

  #print("Script 1 L, a, b means:", np.mean(l), np.mean(a), np.mean(b))

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
