import cv2
import mediapipe as mp
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

success, frame = camera.read()

if not success:
        print("Could not access the webcam.")
        camera.release()
        exit()

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,  
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2 
)
landmarker = vision.HandLandmarker.create_from_options(options)

frame_number = 0
while True:
    success, frame = camera.read()

    if not success:
        print("Could not access the webcam.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame
    )
    timestamp_ms =frame_number * 33

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    frame_number += 1

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):   
            break

landmarker.close()
camera.release()
cv2.destroyAllWindows()
