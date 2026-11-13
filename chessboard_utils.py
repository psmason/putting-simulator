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

