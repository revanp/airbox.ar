# Airbox AR

An experimental webcam app for controlling a live 3D AR-style box with hand gestures.

This is a weekend / for-fun project built with Python, OpenCV, and MediaPipe. It is not a production AR system, but it is designed to be a clean and hackable starting point for gesture-based visual experiments.

## What It Does

Airbox AR tracks both hands from a webcam feed and renders a neon 3D cuboid that reacts in real time:

- Two index fingers define the live box position and size.
- Moving hands apart stretches the cuboid.
- Tilting the line between hands rotates the cuboid.
- Moving one hand closer to the camera creates yaw-like 3D rotation.
- Moving hands closer/farther from the camera affects depth and scale.
- Pinching both hands temporarily boosts the depth effect.
- Finger trails and HUD overlays give it an AR-style visual feel.

The box is intentionally live-only. It is not saved as an object; it appears and changes as your hands move.

## Demo Controls

- Show two hands to activate the live 3D box.
- Use both index fingertips as the control points.
- Move hands apart or closer together to stretch the box.
- Rotate your hands around each other to roll the box.
- Move one hand closer to the camera to rotate the box in depth.
- Pinch `thumb + index` on both hands to boost depth temporarily.
- Press `c` to clear finger trails.
- Press `q` or `Esc` to quit.

## Requirements

- Python `>=3.11,<3.13`
- Webcam access
- [`uv`](https://docs.astral.sh/uv/) for dependency management

MediaPipe wheel support can be sensitive to Python and platform versions, so this project pins compatible dependencies in `pyproject.toml` and `uv.lock`.

## Setup

```bash
uv sync
```

If `uv` does not already have a compatible Python version available:

```bash
uv python install 3.12
uv sync --python 3.12
```

## Run

```bash
uv run gesture-box
```

If your default webcam is not device `0`:

```bash
uv run gesture-box --camera 1
```

## Options

```bash
uv run gesture-box --help
```

Useful flags:

- `--camera`: Webcam device index.
- `--max-hands`: Maximum hands tracked by MediaPipe. Default is `2`.
- `--pinch-threshold`: Thumb-index distance threshold for pinch detection.
- `--trail-length`: Number of fingertip positions kept for the neon trail.

Example:

```bash
uv run gesture-box --pinch-threshold 0.06 --trail-length 28
```

## How Distance Is Estimated

The app approximates distance to the camera from palm width in pixels. It compares the detected distance between the index and pinky MCP landmarks against a rough real-world palm width prior.

This is intentionally lightweight and camera-agnostic. It is good enough for visual interaction, but it is not metric-grade depth estimation.

## Troubleshooting

- Make sure the terminal or app running Python has camera permission.
- Use good lighting and keep both hands visible.
- If pinch detection is too sensitive, increase `--pinch-threshold`.
- If pinch detection is hard to trigger, decrease `--pinch-threshold`.
- If tracking feels jittery, reduce fast hand motion or improve lighting.
- On macOS, if the wrong camera opens, try `--camera 1` or another index.

## Tech Stack

- Python
- OpenCV
- MediaPipe Hands
- NumPy
- uv

## Status

Experimental. Built for fun, learning, and visual prototyping.

## Bonus: Kage Bunshin no Jutsu (Shadow Clone Mode)

A for-fun mode inspired by Naruto's "Jutsu Seribu Bayangan". Cross both hands into
a **`+`** (one hand horizontal, one vertical, overlapping) and hold for a moment to
summon translucent shadow clones of yourself that appear in a smoke puff behind you,
crowded around like a real bunshin squad.

- Selfie segmentation lifts your silhouette off the background so clones get layered
  *behind* the live you.
- Cross hands again to re-summon (clears and respawns the squad).
- Press `d` to dispel clones. `q` / `Esc` to quit.

```bash
uv run shadow-clone
uv run shadow-clone --clones 6 --hold-frames 4
```

Flags:

- `--clones`: number of clones (1-6). Default `5`.
- `--hold-frames`: how many frames the `+` must be held before summoning. Default `6`.
- `--camera`: webcam device index.
- `--max-hands`: hands tracked by MediaPipe.

Ideas that would be interesting to add next:

- Screenshot / recording mode
- Better calibrated camera depth
- Hand-specific gesture modes
- Physics-like smoothing
- Multiple AR primitives besides boxes
