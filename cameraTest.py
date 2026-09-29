import cv2
camera = cv2.VideoCapture(0)
while True:
    success, frame = camera.read()
    if not success:
        print("Could not access the webcam.")
        break
    cv2.imshow("Webcam Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break
camera.release()
cv2.destroyAllWindows()
