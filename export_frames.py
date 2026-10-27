#!/usr/bin/env python3
"""Export the first frame of each clip onto a green 9:16 canvas for outpainting.

Takes a folder of clips (or one video), grabs the first frame of each, scales it
to fit the canvas width, centers it, and fills the rest with solid green so an
outpainting model (ComfyUI + edit model) can fill those regions.

Usage:
  python export_frames.py clips_folder/
  python export_frames.py clips_folder/ -o frames/ --size 1088x1920 --green 6CBE45
  python export_frames.py clip.mp4 --frame 5
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".ts"}


def parse_size(value: str) -> tuple[int, int]:
    try:
        w, h = value.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Size must look like 1088x1920, got: {value}")


def parse_color(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise argparse.ArgumentTypeError(f"Color must be 6-digit hex like 6CBE45, got: {value}")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return b, g, r  # OpenCV uses BGR


def grab_frame(video: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok and frame_index > 0:  # clip shorter than requested offset — fall back to first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def compose(frame, canvas_w: int, canvas_h: int, color_bgr) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(canvas_w / w, canvas_h / h)  # fit inside canvas, preserve aspect
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4)

    canvas = np.full((canvas_h, canvas_w, 3), color_bgr, dtype=np.uint8)
    x = (canvas_w - new_w) // 2
    y = (canvas_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Export first frame of each clip onto a green 9:16 canvas.")
    parser.add_argument("input", type=Path, help="Folder of clips, or a single video file")
    parser.add_argument("-o", "--output", help="Output folder (default: <input>_frames)")
    parser.add_argument("--size", type=parse_size, default=(1088, 1920), help="Canvas WxH (default 1088x1920)")
    parser.add_argument("--green", type=parse_color, default=(69, 190, 108), metavar="HEX",
                        help="Fill color as hex RGB (default 6CBE45)")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to grab (default 0 = first)")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}")
    if args.input.is_file():
        videos = [args.input]
        default_out = args.input.parent / f"{args.input.stem}_frames"
    else:
        videos = sorted(p for p in args.input.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        default_out = args.input.parent / f"{args.input.name}_frames"
    if not videos:
        sys.exit(f"No video files found in {args.input}")

    out_dir = Path(args.output) if args.output else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    canvas_w, canvas_h = args.size
    done, failed = 0, 0
    for video in videos:
        frame = grab_frame(video, args.frame)
        if frame is None:
            print(f"FAILED to read frame: {video.name}", file=sys.stderr)
            failed += 1
            continue
        out_path = out_dir / f"{video.stem}.png"
        cv2.imwrite(str(out_path), compose(frame, canvas_w, canvas_h, args.green))
        done += 1
        print(f"{video.name} -> {out_path.name}")

    print(f"\nDone. {done} frame(s) -> {out_dir}" + (f"  ({failed} failed)" if failed else ""))


if __name__ == "__main__":
    main()
