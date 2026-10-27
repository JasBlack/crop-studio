#!/usr/bin/env python3
"""Split videos into clips at every cut / scene change / camera angle change.

Usage:
  python split_scenes.py video.mp4
  python split_scenes.py video.mp4 -o clips/ --detector adaptive --min-len 0.5
  python split_scenes.py footage_folder/ --copy
  python split_scenes.py video.mp4 --list-only
"""

import argparse
import csv
import sys
from pathlib import Path

from scenedetect import AdaptiveDetector, ContentDetector, detect, split_video_ffmpeg

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".ts"}

# Stream copy: fast, no quality loss, but cuts snap to keyframes (can be off by a second or more).
COPY_ARGS = "-map 0:v:0 -map 0:a? -c copy -avoid_negative_ts make_zero"


def build_detector(name: str, threshold: float | None, min_len_seconds: float):
    min_len = f"{min_len_seconds}s"
    if name == "content":
        return ContentDetector(threshold=threshold if threshold is not None else 27.0, min_scene_len=min_len)
    # Adaptive compares each frame's change score to a rolling average of its neighbours,
    # so it stays reliable during fast camera motion and still catches hard cuts / angle changes.
    return AdaptiveDetector(
        adaptive_threshold=threshold if threshold is not None else 3.0,
        min_scene_len=min_len,
    )


def find_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.iterdir() if p.suffix.lower() in VIDEO_EXTS)


def write_csv(csv_path: Path, scenes) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "start_timecode", "end_timecode", "start_seconds", "end_seconds", "duration_seconds"])
        for i, (start, end) in enumerate(scenes, 1):
            writer.writerow([
                i,
                start.get_timecode(),
                end.get_timecode(),
                f"{start.seconds:.3f}",
                f"{end.seconds:.3f}",
                f"{end.seconds - start.seconds:.3f}",
            ])


def process_video(video: Path, args) -> int:
    print(f"\n=== {video.name} ===")
    detector = build_detector(args.detector, args.threshold, args.min_len)
    scenes = detect(str(video), detector, show_progress=True)

    if not scenes:
        print("No frames decoded (unsupported/corrupt file?) — skipped.")
        return 0

    print(f"{len(scenes)} scene(s) detected:")
    for i, (start, end) in enumerate(scenes, 1):
        print(f"  {i:03d}  {start.get_timecode()} -> {end.get_timecode()}  ({end.seconds - start.seconds:.2f}s)")

    out_dir = Path(args.output) if args.output else video.parent / f"{video.stem}_clips"

    if args.csv or args.list_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{video.stem}_scenes.csv"
        write_csv(csv_path, scenes)
        print(f"Scene list -> {csv_path}")

    if args.list_only:
        return len(scenes)

    if len(scenes) == 1 and not args.force:
        print("Only one scene (no cuts found) — not splitting. Use --force to export anyway.")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if args.copy:
        kwargs["arg_override"] = COPY_ARGS
    elif args.crf is not None:
        kwargs["arg_override"] = f"-map 0:v:0 -map 0:a? -c:v libx264 -preset veryfast -crf {args.crf} -c:a aac"

    rc = split_video_ffmpeg(str(video), scenes, output_dir=str(out_dir), show_progress=True, **kwargs)
    if rc != 0:
        print(f"ffmpeg failed (exit {rc}) for {video.name}", file=sys.stderr)
        return 0
    print(f"{len(scenes)} clips -> {out_dir}")
    return len(scenes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split video(s) into clips at every cut / scene change.")
    parser.add_argument("input", type=Path, help="Video file or folder of videos")
    parser.add_argument("-o", "--output", help="Output folder (default: <video>_clips next to input)")
    parser.add_argument("--detector", choices=["adaptive", "content"], default="adaptive",
                        help="adaptive: robust to camera motion (default). content: plain hard-cut detection.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Sensitivity. adaptive default 3.0, content default 27.0. Lower = more cuts.")
    parser.add_argument("--min-len", type=float, default=0.5, help="Minimum clip length in seconds (default 0.5)")
    parser.add_argument("--copy", action="store_true",
                        help="Stream-copy instead of re-encoding: much faster, but cuts snap to keyframes")
    parser.add_argument("--crf", type=int, default=None, help="x264 quality for re-encode (default 22, lower = better)")
    parser.add_argument("--csv", action="store_true", help="Also write a CSV of scene timecodes")
    parser.add_argument("--list-only", action="store_true", help="Detect and list scenes only, no splitting")
    parser.add_argument("--force", action="store_true", help="Export even if only one scene is found")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}")
    videos = find_videos(args.input)
    if not videos:
        sys.exit(f"No video files found in {args.input}")

    total = 0
    for video in videos:
        total += process_video(video, args)
    print(f"\nDone. {total} clip(s) from {len(videos)} video(s).")


if __name__ == "__main__":
    main()
