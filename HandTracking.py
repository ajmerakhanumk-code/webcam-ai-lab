import cv2
import mediapipe as mp
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
visionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=visionRunningMode.IMAGE,num_hands=2)

with HandLandmarker.create_from_options(options)as landmarker:

    while True:
        success,frame = camera.read()

        if not success:
            print("Could not access the webcam.")
            break

        rgb_frame = cv2.cvtColour(frame,cv2.COLOUR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        camera.release()
        cv2.destroyAllWindows()
