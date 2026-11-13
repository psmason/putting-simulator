import cv2
import time

from datetime import datetime

import state
import tts_utils

if __name__ == '__main__':
    assistant = tts_utils.VoiceAssistant()

    output_file = 'putting-demo-{}.mp4'.format(datetime.now().strftime("%Y-%m-%dT%H%M%S"))
    video_writer = None

    vc = cv2.VideoCapture(0)

    if not vc.isOpened():
        raise "Failed to open camera"

    stop_watch = time.time()

    tracker = state.Tracker(assistant)
    while True:
        if time.time() - stop_watch >= 5.0 and video_writer is None:
            frame_size = frame.shape[:-1][::-1]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            #video_writer = cv2.VideoWriter(output_file,
            #                               fourcc,
            #                               tracker.frame_rate,
            #                               frame_size)
        
        rval, frame = vc.read()
        if not rval:
            break
        annotated = tracker.process(frame, time.time())
        if video_writer is not None:
            video_writer.write(annotated)
        cv2.imshow("tracker", annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

    if video_writer is not None:
        video_writer.release()
    print("video saved to disk: {}".format(output_file))
