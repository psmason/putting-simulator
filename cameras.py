import cv2

# Loop through potential indexes to find Camo
for index in range(0, 5):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Camera found at index: {index}")
        # Capture one frame to confirm it works
        ret, frame = cap.read()
        if ret:
            print(f"Successfully read frame from index {index}")
            cap.release()
            break
        cap.release()
