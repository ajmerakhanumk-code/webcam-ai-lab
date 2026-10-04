import cv2

camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Could not access the webcam.")
    exit()

while True:
    success, frame = camera.read()

    if not success:
        print("Could not access the webcam.")
        break

    cv2.imshow("Webcam AI Lab", frame)

    key = cv2.waitKey(1) & 0xFF

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
