from __future__ import annotations

import argparse
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


Point = tuple[int, int]
Trail = deque[Point]


@dataclass
class HandPointer:
    point: Point
    is_pinching: bool
    camera_distance_cm: float
    z: float


@dataclass
class LiveBoxState:
    center: tuple[float, float]
    size: tuple[float, float, float]
    rotation: tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control a live 3D AR box with two hands."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index.")
    parser.add_argument(
        "--pinch-threshold",
        type=float,
        default=0.045,
        help="Normalized thumb-index distance that counts as a pinch.",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="Maximum number of hands to track.",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=18,
        help="Number of fingertip positions kept for the AR trail.",
    )
    return parser.parse_args()


def landmark_to_point(landmark: object, width: int, height: int) -> Point:
    x = int(min(max(getattr(landmark, "x") * width, 0), width - 1))
    y = int(min(max(getattr(landmark, "y") * height, 0), height - 1))
    return x, y


def distance(a: object, b: object) -> float:
    return math.dist(
        (getattr(a, "x"), getattr(a, "y"), getattr(a, "z", 0.0)),
        (getattr(b, "x"), getattr(b, "y"), getattr(b, "z", 0.0)),
    )


def point_distance(a: Point, b: Point) -> float:
    return math.dist(a, b)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def lerp(current: float, target: float, amount: float) -> float:
    return current + (target - current) * amount


def smooth_tuple(
    current: Sequence[float], target: Sequence[float], amount: float
) -> tuple[float, ...]:
    return tuple(lerp(a, b, amount) for a, b in zip(current, target, strict=True))


def estimate_camera_distance_cm(
    landmarks: Sequence[object],
    width: int,
    height: int,
    mp_hands: object,
) -> float:
    """Approximate distance using palm pixel width and a rough human-palm prior."""
    index_mcp = landmark_to_point(
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP], width, height
    )
    pinky_mcp = landmark_to_point(
        landmarks[mp_hands.HandLandmark.PINKY_MCP], width, height
    )
    palm_px = max(1.0, point_distance(index_mcp, pinky_mcp))
    palm_width_cm = 8.0
    focal_length_px = width * 0.95
    return (palm_width_cm * focal_length_px) / palm_px


def blend_overlay(cv2: object, frame: object, draw: object, alpha: float) -> None:
    cv2.addWeighted(draw, alpha, frame, 1 - alpha, 0, frame)


def rotation_matrix(roll: float, yaw: float, pitch: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    return rz @ ry @ rx


def draw_live_3d_box(
    cv2: object,
    frame: object,
    center: Point,
    size: tuple[float, float, float],
    rotation: tuple[float, float, float],
    frame_index: int,
) -> None:
    width, height, depth = size
    roll, yaw, pitch = rotation
    half_w, half_h, half_d = width / 2, height / 2, depth / 2
    vertices = np.array(
        [
            [-half_w, -half_h, -half_d],
            [half_w, -half_h, -half_d],
            [half_w, half_h, -half_d],
            [-half_w, half_h, -half_d],
            [-half_w, -half_h, half_d],
            [half_w, -half_h, half_d],
            [half_w, half_h, half_d],
            [-half_w, half_h, half_d],
        ],
        dtype=np.float32,
    )

    rotated = vertices @ rotation_matrix(roll, yaw, pitch).T
    focal = 760.0
    camera_offset = 980.0
    projected: list[Point] = []
    for x, y, z in rotated:
        scale = focal / (camera_offset + z)
        projected.append((int(center[0] + x * scale), int(center[1] + y * scale)))

    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    face_order = sorted(faces, key=lambda face: float(np.mean(rotated[list(face), 2])))
    pulse = 0.55 + 0.45 * math.sin(frame_index * 0.16)
    cyan = (255, 235, 40)
    amber = (70, 210, 255)
    magenta = (255, 80, 180)

    overlay = frame.copy()
    for face in face_order:
        points = np.array([projected[index] for index in face], dtype=np.int32)
        face_depth = float(np.mean(rotated[list(face), 2]))
        face_color = amber if face_depth > 0 else cyan
        cv2.fillPoly(overlay, [points], face_color)
    blend_overlay(cv2, frame, overlay, 0.08 + 0.05 * pulse)

    glow = frame.copy()
    for thickness, alpha in ((18, 0.04), (10, 0.08), (5, 0.16)):
        for start, end in edges:
            color = magenta if start >= 4 or end >= 4 else cyan
            cv2.line(glow, projected[start], projected[end], color, thickness, cv2.LINE_AA)
        blend_overlay(cv2, frame, glow, alpha)
        glow = frame.copy()

    for start, end in edges:
        color = magenta if start >= 4 or end >= 4 else cyan
        cv2.line(frame, projected[start], projected[end], color, 2, cv2.LINE_AA)

    front_face = np.array([projected[index] for index in (0, 1, 2, 3)], dtype=np.int32)
    back_face = np.array([projected[index] for index in (4, 5, 6, 7)], dtype=np.int32)
    cv2.polylines(frame, [front_face], True, (80, 255, 255), 3, cv2.LINE_AA)
    cv2.polylines(frame, [back_face], True, magenta, 2, cv2.LINE_AA)

    for ratio in (0.25, 0.5, 0.75):
        top = np.array(projected[0], dtype=np.float32) * (1 - ratio) + np.array(
            projected[3], dtype=np.float32
        ) * ratio
        bottom = np.array(projected[1], dtype=np.float32) * (1 - ratio) + np.array(
            projected[2], dtype=np.float32
        ) * ratio
        cv2.line(
            frame,
            tuple(top.astype(int)),
            tuple(bottom.astype(int)),
            (220, 250, 255),
            1,
            cv2.LINE_AA,
        )

    label = f"LIVE 3D {int(width)}x{int(height)}x{int(depth)}"
    cv2.rectangle(frame, (center[0] - 92, center[1] - 24), (center[0] + 104, center[1]), (10, 18, 22), -1)
    cv2.putText(
        frame,
        label,
        (center[0] - 84, center[1] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (80, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_finger_trail(
    cv2: object,
    frame: object,
    trail: Sequence[Point],
    color_start: tuple[int, int, int] = (255, 120, 40),
    color_end: tuple[int, int, int] = (255, 255, 40),
) -> None:
    if len(trail) < 2:
        return
    for index in range(1, len(trail)):
        alpha = index / len(trail)
        thickness = max(1, int(8 * alpha))
        color = tuple(
            int(color_start[channel] * (1 - alpha) + color_end[channel] * alpha)
            for channel in range(3)
        )
        cv2.line(frame, trail[index - 1], trail[index], color, thickness, cv2.LINE_AA)


def draw_two_hand_guides(
    cv2: object,
    frame: object,
    left: HandPointer,
    right: HandPointer,
    locked: bool,
) -> None:
    color = (80, 255, 255) if locked else (255, 235, 40)
    cv2.line(frame, left.point, right.point, color, 2, cv2.LINE_AA)
    for point in (left.point, right.point):
        cv2.circle(frame, point, 15, color, 2, cv2.LINE_AA)
        cv2.circle(frame, point, 4, (255, 255, 255), -1, cv2.LINE_AA)


def draw_hud(
    cv2: object, frame: object, status: str, active: bool, detail: str
) -> None:
    color = (40, 220, 255) if active else (230, 230, 230)
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (560, 108), (10, 18, 22), -1)
    blend_overlay(cv2, frame, overlay, 0.76)
    cv2.rectangle(frame, (12, 12), (560, 108), (255, 235, 40), 1, cv2.LINE_AA)
    cv2.putText(frame, status, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(
        frame,
        "2 hands: stretch distance | roll hands to rotate | depth from camera distance",
        (24, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
    )
    cv2.putText(
        frame,
        f"{detail} | c: clear trail | q/esc: quit",
        (24, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
    )


def run_app(
    camera: int, pinch_threshold: float, max_hands: int, trail_length: int
) -> int:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"Could not open camera index {camera}.")
        return 1

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    trails: list[Trail] = [
        deque(maxlen=max(2, trail_length)),
        deque(maxlen=max(2, trail_length)),
    ]
    live_box: LiveBoxState | None = None
    frame_index = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        model_complexity=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            frame_index += 1
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_pointers: list[HandPointer] = []

            if results.multi_hand_landmarks:
                for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    landmarks = hand_landmarks.landmark
                    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    is_pinching = distance(thumb_tip, index_tip) <= pinch_threshold
                    pointer = landmark_to_point(index_tip, width, height)
                    camera_distance_cm = estimate_camera_distance_cm(
                        landmarks, width, height, mp_hands
                    )
                    hand_pointers.append(
                        HandPointer(
                            point=pointer,
                            is_pinching=is_pinching,
                            camera_distance_cm=camera_distance_cm,
                            z=getattr(index_tip, "z", 0.0),
                        )
                    )

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                    if hand_index < len(trails):
                        trails[hand_index].append(pointer)
                    cv2.circle(frame, pointer, 8, (40, 220, 255), -1)
            else:
                for trail in trails:
                    trail.clear()

            if len(hand_pointers) < 2:
                trails[1].clear()

            two_hand_active = len(hand_pointers) >= 2

            draw_finger_trail(cv2, frame, trails[0])
            draw_finger_trail(cv2, frame, trails[1], (40, 120, 255), (40, 255, 255))
            detail = "show 2 hands"
            if two_hand_active:
                hand_pointers = sorted(hand_pointers[:2], key=lambda hand: hand.point[0])
                left, right = hand_pointers[0], hand_pointers[1]
                two_hand_locked = left.is_pinching and right.is_pinching
                draw_two_hand_guides(
                    cv2,
                    frame,
                    left,
                    right,
                    two_hand_locked,
                )
                center = (
                    int((left.point[0] + right.point[0]) / 2),
                    int((left.point[1] + right.point[1]) / 2),
                )
                span = point_distance(left.point, right.point)
                avg_distance = (left.camera_distance_cm + right.camera_distance_cm) / 2
                distance_delta = right.camera_distance_cm - left.camera_distance_cm
                z_delta = right.z - left.z
                box_width = clamp(span * 1.08, 90, width * 0.88)
                box_height = clamp(span * 0.66, 70, height * 0.76)
                box_depth = clamp((2600 / max(avg_distance, 18)) + span * 0.22, 45, 260)
                if two_hand_locked:
                    box_depth *= 1.18

                roll = math.atan2(
                    right.point[1] - left.point[1], right.point[0] - left.point[0]
                )
                yaw = clamp(distance_delta / 45 + z_delta * 3.2, -1.05, 1.05)
                pitch = clamp((90 - avg_distance) / 150, -0.45, 0.45) - 0.22
                target_box = LiveBoxState(
                    center=(float(center[0]), float(center[1])),
                    size=(box_width, box_height, box_depth),
                    rotation=(roll, yaw, pitch),
                )
                if live_box is None:
                    live_box = target_box
                else:
                    live_box = LiveBoxState(
                        center=smooth_tuple(live_box.center, target_box.center, 0.34),
                        size=smooth_tuple(live_box.size, target_box.size, 0.26),
                        rotation=smooth_tuple(
                            live_box.rotation, target_box.rotation, 0.22
                        ),
                    )

                draw_live_3d_box(
                    cv2,
                    frame,
                    (int(live_box.center[0]), int(live_box.center[1])),
                    live_box.size,
                    live_box.rotation,
                    frame_index,
                )
                status = "LIVE 3D GRAB: pinch boosts depth" if two_hand_locked else "LIVE 3D AR BOX: move hands to sculpt"
                hud_active = True
                detail = (
                    f"distance: {avg_distance:.0f}cm | yaw: {math.degrees(yaw):.0f}deg"
                )
            elif hand_pointers and hand_pointers[0].is_pinching:
                live_box = None
                status = "ONE HAND SEEN: add second hand for 3D control"
                hud_active = True
            else:
                live_box = None
                status = "SHOW 2 HANDS TO CONTROL LIVE 3D BOX"
                hud_active = False
            draw_hud(cv2, frame, status, hud_active, detail)
            cv2.imshow("Gesture Box Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                for trail in trails:
                    trail.clear()
                live_box = None

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(
        run_app(args.camera, args.pinch_threshold, args.max_hands, args.trail_length)
    )


if __name__ == "__main__":
    main()
