import cv2

camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)

grayscale = False

while True:
    success, frame = camera.read()

    if not success:
        print("Could not access the webcam.")
        break

    if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Webcam AI Lab", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("g"):
        grayscale = not grayscale

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
