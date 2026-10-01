import cv2
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    success, frame = camera.read()

    if not success:
        print("Could not access the webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    cv2.imshow("Webcam AI Lab - Edge Detection", edges)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()