import cv2
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
success, frame = camera.read()

if not success:
    print("Could not access the webcam.")
    camera.release()
    exit()

height,width = frame.shape[:2]

face_detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (width, height))
while True:
    success, frame = camera.read()

    if not success:
        print("Could not access the webcam.")
        break

    faces = face_detector.detect(frame)

    if faces[1] is not None:
        for face in faces[1]:
            x, y, w, h = face[:4].astype(int)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

camera.release()
cv2.destroyAllWindows()