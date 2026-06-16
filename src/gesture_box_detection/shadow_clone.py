from __future__ import annotations

import argparse
import math
import os
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


Point = tuple[int, int]

CLONE_PRESETS: list[tuple[float, float, float, float, float]] = [
    (-1.05, 0.00, 0.80, 0.82, 0.0),
    (1.05, 0.00, 0.80, 0.82, 1.7),
    (-0.62, -0.06, 0.72, 0.74, 3.1),
    (0.62, -0.06, 0.72, 0.74, 4.4),
    (0.00, -0.12, 0.86, 0.80, 2.4),
    (-1.55, -0.02, 0.66, 0.62, 5.0),
]


@dataclass(slots=True)
class HandGeom:
    center: Point
    angle: float
    length: float


@dataclass(slots=True)
class Sprite:
    rgb: np.ndarray
    alpha: np.ndarray
    w: int
    h: int


@dataclass(slots=True)
class Puff:
    x: float
    y: float
    age: int = 0
    life: int = 24
    r0: float = 8.0
    r1: float = 130.0


@dataclass
class CloneState:
    active: bool = False
    sprite: Sprite | None = None
    puffs: list[Puff] = field(default_factory=list)
    spawn_frame: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kage Bunshin no Jutsu: summon shadow clones with a two-hand plus gesture."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index.")
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="Maximum number of hands to track.",
    )
    parser.add_argument(
        "--clones",
        type=int,
        default=5,
        help="Number of shadow clones summoned (1-6).",
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=6,
        help="Frames the plus gesture must be held before summoning.",
    )
    return parser.parse_args()


def landmark_to_point(landmark: object, width: int, height: int) -> Point:
    x = int(min(max(getattr(landmark, "x") * width, 0), width - 1))
    y = int(min(max(getattr(landmark, "y") * height, 0), height - 1))
    return x, y


def point_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.dist(a, b)


def hand_geometry(landmarks: Sequence[object], width: int, height: int) -> HandGeom:
    wrist = landmark_to_point(landmarks[0], width, height)
    mid_mcp = landmark_to_point(landmarks[9], width, height)
    mid_tip = landmark_to_point(landmarks[12], width, height)
    center = ((wrist[0] + mid_tip[0]) // 2, (wrist[1] + mid_tip[1]) // 2)
    dx = mid_mcp[0] - wrist[0]
    dy = mid_mcp[1] - wrist[1]
    angle = math.degrees(math.atan2(dy, dx)) % 180.0
    length = point_distance(wrist, mid_tip)
    return HandGeom(center=center, angle=angle, length=length)


def detect_plus(hands: Sequence[HandGeom]) -> Point | None:
    if len(hands) < 2:
        return None
    a, b = hands[0], hands[1]
    diff = abs(a.angle - b.angle)
    diff = min(diff, 180.0 - diff)
    if not (50.0 < diff < 130.0):
        return None
    span = max(a.length, b.length, 1.0)
    if point_distance(a.center, b.center) > 1.25 * span:
        return None
    cx = (a.center[0] + b.center[0]) / 2
    cy = (a.center[1] + b.center[1]) / 2
    return (int(cx), int(cy))


def person_region(
    mask: np.ndarray, thresh: float = 0.5
) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > thresh)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def extract_sprite(frame: np.ndarray, mask: np.ndarray) -> Sprite | None:
    region = person_region(mask)
    if region is None:
        return None
    x0, y0, x1, y1 = region
    pad = 12
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(frame.shape[1], x1 + pad)
    y1 = min(frame.shape[0], y1 + pad)
    rgb = frame[y0:y1, x0:x1].copy()
    alpha = (mask[y0:y1, x0:x1] * 255.0).clip(0, 255).astype(np.uint8)
    if rgb.size == 0:
        return None
    return Sprite(rgb=rgb, alpha=alpha, w=x1 - x0, h=y1 - y0)


def blit_sprite(
    cv2: object,
    layer_f: np.ndarray,
    sprite: Sprite,
    center: Point,
    scale: float,
    opacity: float,
    tint: tuple[int, int, int],
) -> None:
    nw = max(1, int(sprite.w * scale))
    nh = max(1, int(sprite.h * scale))
    tinted = np.clip(
        sprite.rgb.astype(np.float32) * 0.82 + np.array(tint, dtype=np.float32) * 0.18,
        0,
        255,
    ).astype(np.uint8)
    rgb = cv2.resize(tinted, (nw, nh), interpolation=cv2.INTER_AREA)
    alpha = (
        cv2.resize(sprite.alpha, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
        / 255.0
    )
    rgb_f = rgb.astype(np.float32)

    cx, cy = int(center[0]), int(center[1])
    x0 = cx - nw // 2
    y0 = cy - nh // 2
    x1, y1 = x0 + nw, y0 + nh
    cx0 = max(0, x0)
    cy0 = max(0, y0)
    cx1 = min(layer_f.shape[1], x1)
    cy1 = min(layer_f.shape[0], y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    sx0 = cx0 - x0
    sy0 = cy0 - y0
    sx1 = sx0 + (cx1 - cx0)
    sy1 = sy0 + (cy1 - cy0)
    region = layer_f[cy0:cy1, cx0:cx1]
    a = alpha[sy0:sy1, sx0:sx1] * opacity
    a3 = a[..., None]
    layer_f[cy0:cy1, cx0:cx1] = region * (1.0 - a3) + rgb_f[sy0:sy1, sx0:sx1] * a3


def spawn_puffs(puffs: list[Puff], positions: Sequence[Point], rng: random.Random) -> None:
    for pos in positions:
        for _ in range(2):
            jx = rng.uniform(-22, 22)
            jy = rng.uniform(-22, 22)
            life = rng.randint(20, 30)
            r1 = rng.uniform(90, 160)
            puffs.append(
                Puff(x=pos[0] + jx, y=pos[1] + jy, life=life, r1=r1)
            )


def draw_smoke_puffs(cv2: object, frame: np.ndarray, puffs: list[Puff]) -> None:
    height, width = frame.shape[:2]
    for puff in list(puffs):
        if puff.age >= puff.life:
            puffs.remove(puff)
            continue
        t = puff.age / puff.life
        eased = 1.0 - (1.0 - t) ** 2
        radius = int(puff.r0 + (puff.r1 - puff.r0) * eased)
        alpha = max(0.0, (1.0 - t)) * 0.55
        px, py = int(puff.x), int(puff.y)
        x0 = max(0, px - radius)
        y0 = max(0, py - radius)
        x1 = min(width, px + radius)
        y1 = min(height, py + radius)
        if x0 < x1 and y0 < y1:
            roi = frame[y0:y1, x0:x1]
            overlay = roi.copy()
            cv2.circle(overlay, (px - x0, py - y0), radius, (215, 215, 215), -1)
            cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
            core = max(2, int(radius * 0.35))
            overlay = roi.copy()
            cv2.circle(overlay, (px - x0, py - y0), core, (245, 245, 245), -1)
            cv2.addWeighted(overlay, alpha * 0.8, roi, 1.0 - alpha * 0.8, 0, roi)
        puff.age += 1


def draw_plus_guide(
    cv2: object, frame: np.ndarray, center: Point | None, hold: int, hold_target: int
) -> None:
    if center is None:
        return
    ratio = min(1.0, hold / hold_target) if hold_target > 0 else 0.0
    radius = int(34 + 26 * ratio)
    pulse = 0.5 + 0.5 * math.sin(hold * 0.5)
    color = (40, 220, 255) if ratio >= 1.0 else (255, 235, 40)
    cv2.circle(frame, center, radius, color, 2, cv2.LINE_AA)
    cv2.circle(frame, center, 5, (255, 255, 255), -1, cv2.LINE_AA)
    if ratio < 1.0:
        cv2.ellipse(
            frame,
            center,
            (radius, radius),
            -90,
            0,
            360 * ratio,
            (40, 220, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "CHARGING JUTSU",
            (center[0] - 70, center[1] - radius - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 235, 40),
            1,
            cv2.LINE_AA,
        )


def draw_hud(
    cv2: object,
    frame: np.ndarray,
    status: str,
    active: bool,
    clones_left: int,
    detail: str,
) -> None:
    color = (40, 220, 255) if active else (255, 235, 40)
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (620, 108), (10, 18, 22), -1)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)
    cv2.rectangle(frame, (12, 12), (620, 108), color, 1, cv2.LINE_AA)
    title = "影分身の術  KAGE BUNSHIN NO JUTSU"
    cv2.putText(frame, title, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"{status}   |   clones: {clones_left}",
        (24, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{detail}   |   cross hands into a '+' to summon | d: dispel | q/esc: quit",
        (24, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )


def run_app(
    camera: int, max_hands: int, clone_count: int, hold_frames: int
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
    mp_selfie = mp.solutions.selfie_segmentation

    clone_count = max(1, min(6, clone_count))
    clones_layout = CLONE_PRESETS[:clone_count]
    clone_tint = (150, 175, 210)

    state = CloneState()
    plus_hold = 0
    plus_history: deque[bool] = deque(maxlen=6)
    cooldown = 0
    last_region: tuple[int, int, int, int] | None = None
    last_mask: np.ndarray | None = None
    rng = random.Random(7)
    frame_index = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.55,
    ) as hands, mp_selfie.SelfieSegmentation(model_selection=1) as segmentor:
        while True:
            frame_index += 1
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hands_result = hands.process(rgb)

            mask: np.ndarray | None = None
            # Run segmentation every frame (throttled) regardless of state.active
            # so live_region is populated even before the first summon.
            if (frame_index % 2 == 0) or (last_mask is None):
                last_mask = (
                    segmentor.process(rgb).segmentation_mask.astype(np.float32)
                )
            mask = last_mask
            region = person_region(mask)
            if region is not None:
                last_region = region
            live_region = last_region

            hand_geoms: list[HandGeom] = []
            if hands_result.multi_hand_landmarks:
                for hand_landmarks in hands_result.multi_hand_landmarks:
                    landmarks = hand_landmarks.landmark
                    hand_geoms.append(hand_geometry(landmarks, width, height))
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
            plus_center = detect_plus(hand_geoms)

            plus_history.append(plus_center is not None)
            if sum(plus_history) >= 2:
                plus_hold += 1
            else:
                plus_hold = max(0, plus_hold - 1)

            if cooldown > 0:
                cooldown -= 1

            if (
                plus_hold >= hold_frames
                and cooldown == 0
                and live_region is not None
            ):
                sprite = extract_sprite(frame, mask)
                if sprite is not None:
                    state.active = True
                    state.sprite = sprite
                    state.spawn_frame = frame_index
                    state.puffs.clear()
                    feet_y = live_region[3]
                    feet_x = (live_region[0] + live_region[2]) / 2
                    positions: list[Point] = [(int(feet_x), int(feet_y - sprite.h * 0.5))]
                    for ox, foot_oy, _scale, _op, _phase in clones_layout:
                        cx = int(feet_x + ox * sprite.w)
                        cy = int(feet_y + foot_oy * sprite.h)
                        positions.append((cx, cy))
                    spawn_puffs(state.puffs, positions, rng)
                    cooldown = 40
                    plus_hold = 0

            clones_visible = (
                state.active
                and state.sprite is not None
                and live_region is not None
                and mask is not None
            )
            if clones_visible:
                m3 = mask[..., None]
                inv3 = 1.0 - m3
                frame_f = frame.astype(np.float32)
                layer_f = frame_f * inv3
                sprite = state.sprite
                feet_y = live_region[3]
                feet_x = (live_region[0] + live_region[2]) / 2
                spawn_age = frame_index - state.spawn_frame
                appear = min(1.0, spawn_age / 6.0)
                for ox, foot_oy, scale, _op, phase in clones_layout:
                    bob = 0.02 * math.sin((frame_index + phase * 6) * 0.08)
                    cy = feet_y + foot_oy * sprite.h + bob * sprite.h
                    cx = feet_x + ox * sprite.w
                    center = (int(cx), int(cy - sprite.h * scale * 0.5))
                    blit_sprite(
                        cv2,
                        layer_f,
                        sprite,
                        center,
                        scale * (0.92 + 0.08 * appear),
                        appear,
                        clone_tint,
                    )
                composited = layer_f * inv3 + frame_f * m3
                frame_out = np.clip(composited, 0, 255).astype(np.uint8)
            else:
                frame_out = frame

            draw_smoke_puffs(cv2, frame_out, state.puffs)
            draw_plus_guide(cv2, frame_out, plus_center, plus_hold, hold_frames)

            if state.active:
                status = "BUNSHIN ACTIVE"
                detail = "re-cross hands to re-summon"
            elif plus_center is not None:
                status = "PLUS DETECTED - hold to charge"
                detail = "summon incoming"
            elif hand_geoms:
                status = "HANDS SEEN"
                detail = "form a '+' with both hands"
            else:
                status = "SHOW 2 HANDS"
                detail = "make a plus (+) to begin"

            draw_hud(
                cv2,
                frame_out,
                status,
                state.active,
                clone_count if state.active else 0,
                detail,
            )
            cv2.imshow("Kage Bunshin - Shadow Clone Jutsu", frame_out)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                state.active = False
                state.sprite = None
                state.puffs.clear()
                plus_hold = 0

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> None:
    args = parse_args()
    raise SystemExit(
        run_app(args.camera, args.max_hands, args.clones, args.hold_frames)
    )


if __name__ == "__main__":
    main()
