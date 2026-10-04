import cv2
import mediapipe as mp
import math
import random
import numpy as np


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


# ============================================================
# FIRE 🔥
# ============================================================

def draw_flame(frame, x, y):

    overlay = frame.copy()

    outer = [
        (x, y + 10),
        (x - 35, y - 15),
        (x - 48, y - 55),
        (x - 30, y - 45),
        (x - 25, y - 105),
        (x - 8, y - 75),
        (x + 8, y - 135),
        (x + 18, y - 75),
        (x + 42, y - 105),
        (x + 35, y - 50),
        (x + 55, y - 20),
    ]

    cv2.fillPoly(
        overlay,
        [np.array(outer, dtype=np.int32)],
        (0, 60, 255)
    )

    inner = [
        (x, y),
        (x - 25, y - 20),
        (x - 18, y - 55),
        (x - 5, y - 35),
        (x + 2, y - 95),
        (x + 15, y - 55),
        (x + 30, y - 75),
        (x + 25, y - 30),
        (x + 38, y - 15),
    ]

    cv2.fillPoly(
        overlay,
        [np.array(inner, dtype=np.int32)],
        (0, 150, 255)
    )

    core = [
        (x, y),
        (x - 13, y - 18),
        (x - 8, y - 40),
        (x + 2, y - 65),
        (x + 12, y - 35),
        (x + 18, y - 15),
    ]

    cv2.fillPoly(
        overlay,
        [np.array(core, dtype=np.int32)],
        (180, 255, 255)
    )

    glow = cv2.GaussianBlur(
        overlay,
        (51, 51),
        0
    )

    frame[:] = cv2.addWeighted(
        frame,
        0.55,
        glow,
        0.65,
        0
    )

    frame[:] = cv2.addWeighted(
        frame,
        0.65,
        overlay,
        0.55,
        0
    )

    for i in range(14):

        ember_x = x + random.randint(-65, 65)
        ember_y = y - random.randint(20, 150)

        radius = random.randint(1, 4)

        cv2.circle(
            frame,
            (ember_x, ember_y),
            radius,
            (80, 220, 255),
            -1
        )


# ============================================================
# ICE ❄️
# ============================================================

def draw_crystal(frame, x, y, size, angle):

    points = []

    for i in range(6):

        current_angle = angle + (math.pi / 3) * i

        px = int(
            x + math.cos(current_angle) * size
        )

        py = int(
            y + math.sin(current_angle) * size
        )

        points.append((px, py))

    cv2.fillPoly(
        frame,
        [np.array(points, dtype="int32")],
        (255, 190, 40)
    )

    inner_size = size * 0.55
    inner_points = []

    for i in range(6):

        current_angle = angle + (math.pi / 3) * i

        px = int(
            x + math.cos(current_angle) * inner_size
        )

        py = int(
            y + math.sin(current_angle) * inner_size
        )

        inner_points.append((px, py))

    cv2.fillPoly(
        frame,
        [np.array(inner_points, dtype="int32")],
        (255, 255, 255)
    )


def draw_ice(frame, x, y):

    overlay = frame.copy()

    cv2.circle(
        overlay,
        (x, y),
        55,
        (255, 220, 80),
        -1
    )

    glow = cv2.GaussianBlur(
        overlay,
        (41, 41),
        0
    )

    frame[:] = cv2.addWeighted(
        frame,
        0.65,
        glow,
        0.35,
        0
    )

    for i in range(5):

        crystal_x = x + random.randint(-35, 35)
        crystal_y = y + random.randint(-55, 25)

        size = random.randint(12, 22)

        angle = random.uniform(
            0,
            math.pi
        )

        draw_crystal(
            frame,
            crystal_x,
            crystal_y,
            size,
            angle
        )

    for i in range(12):

        particle_x = x + random.randint(-70, 70)
        particle_y = y + random.randint(-90, 50)

        radius = random.randint(2, 5)

        cv2.circle(
            frame,
            (particle_x, particle_y),
            radius,
            (255, 255, 255),
            -1
        )


# ============================================================
# SHIELD 🛡️
# ============================================================

def draw_shield(frame, x, y):

    overlay = frame.copy()

    radius = 150

    cv2.circle(
        overlay,
        (x, y),
        radius,
        (180, 50, 255),
        18
    )

    cv2.circle(
        overlay,
        (x, y),
        radius + 12,
        (255, 80, 220),
        5
    )

    cv2.circle(
        overlay,
        (x, y),
        radius - 25,
        (120, 40, 255),
        3
    )

    cv2.circle(
        overlay,
        (x, y),
        radius - 45,
        (80, 30, 180),
        2
    )

    for angle in range(0, 360, 30):

        radians = math.radians(angle)

        inner_x = int(
            x + math.cos(radians) * (radius - 55)
        )

        inner_y = int(
            y + math.sin(radians) * (radius - 55)
        )

        outer_x = int(
            x + math.cos(radians) * (radius - 5)
        )

        outer_y = int(
            y + math.sin(radians) * (radius - 5)
        )

        cv2.line(
            overlay,
            (inner_x, inner_y),
            (outer_x, outer_y),
            (255, 180, 255),
            2
        )

    for i in range(25):

        angle = random.uniform(
            0,
            math.pi * 2
        )

        distance_from_center = random.randint(
            radius - 10,
            radius + 35
        )

        particle_x = int(
            x + math.cos(angle) * distance_from_center
        )

        particle_y = int(
            y + math.sin(angle) * distance_from_center
        )

        cv2.circle(
            overlay,
            (particle_x, particle_y),
            random.randint(2, 5),
            (255, 200, 255),
            -1
        )

    glow = cv2.GaussianBlur(
        overlay,
        (51, 51),
        0
    )

    frame[:] = cv2.addWeighted(
        frame,
        0.55,
        glow,
        0.65,
        0
    )

    frame[:] = cv2.addWeighted(
        frame,
        0.75,
        overlay,
        0.45,
        0
    )


# ============================================================
# WATER 💧
# ============================================================

def draw_water(frame, x, y, time):

    overlay = frame.copy()

    cv2.circle(
        overlay,
        (x, y),
        28,
        (255, 150, 30),
        -1
    )

    for i in range(28):

        angle = (
            random.uniform(0, math.pi * 2)
            + time * 0.035
        )

        radius = random.randint(25, 105)

        drop_x = int(
            x + math.cos(angle) * radius
        )

        drop_y = int(
            y + math.sin(angle) * radius
        )

        drop_size = random.randint(3, 8)

        cv2.circle(
            overlay,
            (drop_x, drop_y),
            drop_size + 5,
            (255, 110, 20),
            -1
        )

        cv2.ellipse(
            overlay,
            (drop_x, drop_y),
            (drop_size, drop_size + 3),
            random.randint(0, 180),
            0,
            360,
            (255, 210, 100),
            -1
        )

        cv2.circle(
            overlay,
            (
                drop_x - 1,
                drop_y - 2
            ),
            max(1, drop_size // 3),
            (255, 255, 255),
            -1
        )

    for stream in range(5):

        points = []

        phase = stream * (math.pi * 2 / 5)

        for i in range(50):

            angle = (
                phase
                + time * 0.045
                + i * 0.13
            )

            radius = 18 + i * 1.5

            px = int(
                x + math.cos(angle) * radius
            )

            py = int(
                y + math.sin(angle) * radius * 0.75
            )

            points.append(
                (px, py)
            )

        cv2.polylines(
            overlay,
            [np.array(points, dtype=np.int32)],
            False,
            (255, 200, 70),
            5
        )

    for i in range(8):

        angle = (
            time * 0.06
            + i * math.pi / 4
        )

        radius = 35 + math.sin(
            time * 0.08 + i
        ) * 15

        px = int(
            x + math.cos(angle) * radius
        )

        py = int(
            y + math.sin(angle) * radius
        )

        cv2.circle(
            overlay,
            (px, py),
            3,
            (255, 255, 255),
            -1
        )

    glow = cv2.GaussianBlur(
        overlay,
        (41, 41),
        0
    )

    frame[:] = cv2.addWeighted(
        frame,
        0.60,
        glow,
        0.50,
        0
    )

    frame[:] = cv2.addWeighted(
        frame,
        0.72,
        overlay,
        0.60,
        0
    )


# ============================================================
# LIGHTNING ⚡
# ============================================================

def draw_lightning(frame, x, y, time):

    # Lightning will be the BIGGEST effect.
    # The hand becomes the source and the bolt explodes upward.

    height = frame.shape[0]
    width = frame.shape[1]

    overlay = np.zeros_like(frame)

    # Main lightning destination
    target_x = x + random.randint(-80, 80)
    target_y = max(20, y - 430)

    # --------------------------------------------------------
    # Create a branching lightning bolt
    # --------------------------------------------------------

    def create_bolt(start_x, start_y, end_x, end_y, segments=14):

        points = [(start_x, start_y)]

        for i in range(1, segments):

            progress = i / segments

            px = (
                start_x
                + (end_x - start_x) * progress
            )

            py = (
                start_y
                + (end_y - start_y) * progress
            )

            # Strong but controlled electric movement
            offset = random.randint(-35, 35)

            px += offset

            points.append(
                (int(px), int(py))
            )

        points.append(
            (int(end_x), int(end_y))
        )

        return points


    main_bolt = create_bolt(
        x,
        y - 5,
        target_x,
        target_y,
        18
    )


    # --------------------------------------------------------
    # Huge outer electric glow
    # --------------------------------------------------------

    for i in range(len(main_bolt) - 1):

        cv2.line(
            overlay,
            main_bolt[i],
            main_bolt[i + 1],
            (255, 80, 0),
            32
        )


    # --------------------------------------------------------
    # Medium blue energy glow
    # --------------------------------------------------------

    for i in range(len(main_bolt) - 1):

        cv2.line(
            overlay,
            main_bolt[i],
            main_bolt[i + 1],
            (255, 170, 20),
            16
        )


    # --------------------------------------------------------
    # Bright electric core
    # --------------------------------------------------------

    for i in range(len(main_bolt) - 1):

        cv2.line(
            overlay,
            main_bolt[i],
            main_bolt[i + 1],
            (255, 255, 255),
            5
        )


    # --------------------------------------------------------
    # Major branches
    # --------------------------------------------------------

    branch_positions = [
        4,
        7,
        10,
        13
    ]

    for position in branch_positions:

        if position >= len(main_bolt):
            continue

        start_x, start_y = main_bolt[position]

        branch_end_x = (
            start_x
            + random.randint(-180, 180)
        )

        branch_end_y = (
            start_y
            - random.randint(80, 190)
        )

        branch = create_bolt(
            start_x,
            start_y,
            branch_end_x,
            branch_end_y,
            7
        )

        for i in range(len(branch) - 1):

            cv2.line(
                overlay,
                branch[i],
                branch[i + 1],
                (255, 90, 0),
                18
            )

        for i in range(len(branch) - 1):

            cv2.line(
                overlay,
                branch[i],
                branch[i + 1],
                (255, 210, 30),
                9
            )

        for i in range(len(branch) - 1):

            cv2.line(
                overlay,
                branch[i],
                branch[i + 1],
                (255, 255, 255),
                3
            )


    # --------------------------------------------------------
    # Smaller electrical branches
    # --------------------------------------------------------

    for i in range(10):

        if len(main_bolt) < 4:
            break

        position = random.randint(
            2,
            len(main_bolt) - 2
        )

        start_x, start_y = main_bolt[position]

        end_x = start_x + random.randint(
            -110,
            110
        )

        end_y = start_y - random.randint(
            30,
            100
        )

        small_branch = create_bolt(
            start_x,
            start_y,
            end_x,
            end_y,
            5
        )

        for j in range(len(small_branch) - 1):

            cv2.line(
                overlay,
                small_branch[j],
                small_branch[j + 1],
                (255, 180, 20),
                6
            )

            cv2.line(
                overlay,
                small_branch[j],
                small_branch[j + 1],
                (255, 255, 255),
                2
            )


    # --------------------------------------------------------
    # Massive glow
    # --------------------------------------------------------

    glow_large = cv2.GaussianBlur(
        overlay,
        (81, 81),
        0
    )

    frame[:] = cv2.addWeighted(
        frame,
        0.45,
        glow_large,
        0.95,
        0
    )


    glow_medium = cv2.GaussianBlur(
        overlay,
        (31, 31),
        0
    )

    frame[:] = cv2.addWeighted(
        frame,
        0.55,
        glow_medium,
        0.85,
        0
    )


    # --------------------------------------------------------
    # Main lightning
    # --------------------------------------------------------

    frame[:] = cv2.addWeighted(
        frame,
        0.70,
        overlay,
        0.95,
        0
    )


    # --------------------------------------------------------
    # Electric explosion around hand
    # --------------------------------------------------------

    energy_x = x
    energy_y = y - 10

    for i in range(30):

        angle = random.uniform(
            0,
            math.pi * 2
        )

        radius = random.randint(
            40,
            170
        )

        particle_x = int(
            energy_x
            + math.cos(angle) * radius
        )

        particle_y = int(
            energy_y
            + math.sin(angle) * radius
        )

        cv2.circle(
            frame,
            (particle_x, particle_y),
            random.randint(1, 4),
            (255, 240, 180),
            -1
        )


    # --------------------------------------------------------
    # Bright energy core at palm
    # --------------------------------------------------------

    core = frame.copy()

    cv2.circle(
        core,
        (energy_x, energy_y),
        45,
        (255, 255, 255),
        -1
    )

    core_glow = cv2.GaussianBlur(
        core,
        (51, 51),
        0
    )

    frame[:] = cv2.addWeighted(
        frame,
        0.70,
        core_glow,
        0.50,
        0
    )


# ============================================================
# MAIN LOOP
# ============================================================

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

            thumb_tip = hand_landmark[4]
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


            index_extended = (
                index_distance
                > index_base_distance * 1.4
            )

            middle_extended = (
                middle_distance
                > middle_base_distance * 1.4
            )

            ring_extended = (
                ring_distance
                > ring_base_distance * 1.4
            )

            pinky_extended = (
                pinky_distance
                > pinky_base_distance * 1.4
            )


            fist = (
                index_folded
                and middle_folded
                and ring_folded
                and pinky_folded
            )


            one_finger = (
                index_extended
                and middle_folded
                and ring_folded
                and pinky_folded
            )


            open_palm = (
                index_extended
                and middle_extended
                and ring_extended
                and pinky_extended
            )


            pinch_distance = distance(
                thumb_tip,
                index_tip
            )

            pinch = (
                pinch_distance < 0.08
                and middle_extended
                and ring_extended
                and pinky_extended
            )   
            palm_dx = (
                middle_mcp.x
                - wrist.x
            )

            palm_dy = (
                middle_mcp.y
                - wrist.y
            )

            horizontal_palm = (
                open_palm
                and abs(palm_dx)
                > abs(palm_dy) * 1.35
            )


            # =================================================
            # FIRE 🔥
            # =================================================

            if fist:

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

                draw_flame(
                    frame,
                    palm_x,
                    palm_y
                )


            # =================================================
            # ICE ❄️
            # =================================================

            elif one_finger:

                ice_x = int(
                    index_tip.x
                    * frame.shape[1]
                )

                ice_y = int(
                    index_tip.y
                    * frame.shape[0]
                )

                draw_ice(
                    frame,
                    ice_x,
                    ice_y
                )


            # =================================================
            # LIGHTNING ⚡
            # =================================================

            elif horizontal_palm:

                lightning_x = int(
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

                lightning_y = int(
                    (
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
                    - 20
                )

                draw_lightning(
                    frame,
                    lightning_x,
                    lightning_y,
                    frame_number
                )


            # =================================================
            # SHIELD 🛡️
            # =================================================

            elif open_palm:

                shield_x = int(
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

                shield_y = int(
                    (
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
                    - 80
                )

                draw_shield(
                    frame,
                    shield_x,
                    shield_y
                )


            # =================================================
            # WATER 💧
            # =================================================

            elif pinch:

                water_x = int(
                    (
                        thumb_tip.x
                        + index_tip.x
                    )
                    / 2
                    * frame.shape[1]
                )

                water_y = int(
                    (
                        (
                            thumb_tip.y
                            + index_tip.y
                        )
                        / 2
                        * frame.shape[0]
                    )
                    - 20
                )

                draw_water(
                    frame,
                    water_x,
                    water_y,
                    frame_number
                )


            # =================================================
            # HAND LANDMARKS
            # =================================================

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