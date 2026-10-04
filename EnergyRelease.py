import cv2
import mediapipe as mp
import math
import random


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not access the webcam.")
    exit()


from mediapipe.tasks import python
from mediapipe.tasks.python import vision


base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

landmarker = vision.HandLandmarker.create_from_options(options)


frame_number = 0


def distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )


def draw_flame(frame, x, y):

    overlay = frame.copy()

    # Flame size
    height = random.randint(90, 125)
    width = random.randint(45, 65)

    # Outer flame
    outer_points = [
        (x, y),
        (x - width, y - 20),
        (x - width + random.randint(5, 20), y - 55),
        (x - width // 2, y - 45),
        (x - random.randint(15, 35), y - height),
        (x + random.randint(-10, 10), y - height + 30),
        (x + random.randint(10, 35), y - height // 2),
        (x + width // 2, y - 65),
        (x + width, y - 25),
        (x + width - random.randint(5, 20), y - 5),
    ]

    cv2.fillPoly(
        overlay,
        [__import__("numpy").array(outer_points, dtype="int32")],
        (0, 0, 255)
    )

    # Orange middle flame
    middle_points = [
        (x, y),
        (x - width // 2, y - 15),
        (x - width // 3, y - 45),
        (x - 12, y - height + 35),
        (x + 5, y - height // 2),
        (x + width // 3, y - 55),
        (x + width // 2, y - 15),
    ]

    cv2.fillPoly(
        overlay,
        [__import__("numpy").array(middle_points, dtype="int32")],
        (0, 100, 255)
    )

    # Yellow-hot inner flame
    inner_points = [
        (x, y),
        (x - 20, y - 15),
        (x - 12, y - 40),
        (x, y - random.randint(45, 70)),
        (x + 12, y - 35),
        (x + 20, y - 12),
    ]

    cv2.fillPoly(
        overlay,
        [__import__("numpy").array(inner_points, dtype="int32")],
        (0, 220, 255)
    )

    # Glow
    glow = cv2.GaussianBlur(overlay, (31, 31), 0)

    frame[:] = cv2.addWeighted(
        frame,
        0.55,
        glow,
        0.45,
        0
    )

    # Put the sharp flame back on top
    frame[:] = cv2.addWeighted(
        frame,
        0.65,
        overlay,
        0.35,
        0
    )


while True:

    success, frame = camera.read()

    if not success:
        print("Could not read webcam frame.")
        continue

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = frame_number * 33

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    frame_number += 1


    if result.hand_landmarks:

        for hand_landmark in result.hand_landmarks:

            wrist = hand_landmark[0]

            index_tip = hand_landmark[8]
            middle_tip = hand_landmark[12]
            ring_tip = hand_landmark[16]
            pinky_tip = hand_landmark[20]

            index_mcp = hand_landmark[5]
            middle_mcp = hand_landmark[9]
            ring_mcp = hand_landmark[13]
            pinky_mcp = hand_landmark[17]


            index_distance = distance(
                index_tip,
                wrist
            )

            middle_distance = distance(
                middle_tip,
                wrist
            )

            ring_distance = distance(
                ring_tip,
                wrist
            )

            pinky_distance = distance(
                pinky_tip,
                wrist
            )


            index_base_distance = distance(
                index_mcp,
                wrist
            )

            middle_base_distance = distance(
                middle_mcp,
                wrist
            )

            ring_base_distance = distance(
                ring_mcp,
                wrist
            )

            pinky_base_distance = distance(
                pinky_mcp,
                wrist
            )


            index_folded = (
                index_distance
                < index_base_distance * 1.4
            )

            middle_folded = (
                middle_distance
                < middle_base_distance * 1.4
            )

            ring_folded = (
                ring_distance
                < ring_base_distance * 1.4
            )

            pinky_folded = (
                pinky_distance
                < pinky_base_distance * 1.4
            )


            fist = (
                index_folded
                and middle_folded
                and ring_folded
                and pinky_folded
            )


            # Find palm center
            palm_x = int(
                (
                    hand_landmark[0].x
                    + hand_landmark[5].x
                    + hand_landmark[9].x
                    + hand_landmark[13].x
                    + hand_landmark[17].x
                )
                / 5
                * frame.shape[1]
            )

            palm_y = int(
                (
                    hand_landmark[0].y
                    + hand_landmark[5].y
                    + hand_landmark[9].y
                    + hand_landmark[13].y
                    + hand_landmark[17].y
                )
                / 5
                * frame.shape[0]
            )


            # FIRE 🔥
            if fist:

                draw_flame(
                    frame,
                    palm_x,
                    palm_y
                )


            # Hand landmarks
            for landmark in hand_landmark:

                x = int(
                    landmark.x
                    * frame.shape[1]
                )

                y = int(
                    landmark.y
                    * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


    cv2.imshow(
        "Energy Release",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


landmarker.close()

camera.release()

cv2.destroyAllWindows()