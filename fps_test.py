import time
import cv2

# Open the default camera (index 0)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))

# 2. Explicitly request 60 FPS from the virtual device
cap.set(cv2.CAP_PROP_FPS, 60)

# Variables for FPS calculation
prev_time = 0

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    break

  # Calculate FPS
  current_time = time.time()
  fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
  prev_time = current_time

  # Draw FPS text on the frame
  cv2.putText(
      frame,
      f"FPS: {int(fps)}",
      (10, 30),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (0, 255, 0),
      2,
  )

  # Display the image
  cv2.imshow("FPS Test", frame)

  # Exit on pressing 'q'
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
