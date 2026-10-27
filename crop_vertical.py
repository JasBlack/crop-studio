#!/usr/bin/env python3
"""Auto-crop horizontal clips to vertical 9:16, tracking the main subject, then re-stitch.

Priority order for framing:
  1. Faces (SCRFD/insightface ONNX detector, if the model file is present)
  2. Fine-detail energy (sharp regions beat soft sky/clouds)

The crop window follows the subject smoothly over time (median + gaussian
smoothed track, subpixel crop). Shots where the subject barely moves get a
locked static window automatically. Use --static to force one fixed window
per clip.

Usage:
  python crop_vertical.py clips_folder/ --audio-from original.mp4
  python crop_vertical.py clips_folder/ --preview
  python crop_vertical.py clip.mp4 --center 0.7 --no-stitch
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".ts"}
_MODEL_CANDIDATES = [
    Path(__file__).parent / "models" / "det_10g.onnx",  # bundled copy
    Path.home() / ".insightface" / "models" / "buffalo_l" / "det_10g.onnx",
]
DEFAULT_FACE_MODEL = next((p for p in _MODEL_CANDIDATES if p.exists()), _MODEL_CANDIDATES[0])


# --------------------------------------------------------------------------- faces

class FaceDetector:
    """Minimal SCRFD (insightface det_10g) wrapper on onnxruntime."""

    INPUT = 640
    STRIDES = (8, 16, 32)

    def __init__(self, model_path: Path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self._centers_cache = {}

    def _anchor_centers(self, stride: int) -> np.ndarray:
        if stride not in self._centers_cache:
            n = self.INPUT // stride
            grid = np.stack(np.mgrid[:n, :n][::-1], axis=-1).reshape(-1, 2).astype(np.float32) * stride
            self._centers_cache[stride] = np.repeat(grid, 2, axis=0)  # 2 anchors per cell
        return self._centers_cache[stride]

    def detect(self, frame_bgr: np.ndarray, threshold: float = 0.4) -> list[tuple[float, float, float, float, float]]:
        """Return faces as (x1, y1, x2, y2, score) in frame coordinates."""
        h, w = frame_bgr.shape[:2]
        scale = self.INPUT / max(h, w)
        rw, rh = round(w * scale), round(h * scale)
        canvas = np.zeros((self.INPUT, self.INPUT, 3), dtype=np.uint8)
        canvas[:rh, :rw] = cv2.resize(frame_bgr, (rw, rh))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (self.INPUT, self.INPUT),
                                     (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.session.run(None, {self.input_name: blob})

        boxes, scores = [], []
        for i, stride in enumerate(self.STRIDES):
            s = outputs[i].reshape(-1)
            keep = s >= threshold
            if not keep.any():
                continue
            centers = self._anchor_centers(stride)[keep]
            d = outputs[i + 3].reshape(-1, 4)[keep] * stride
            x1 = centers[:, 0] - d[:, 0]
            y1 = centers[:, 1] - d[:, 1]
            x2 = centers[:, 0] + d[:, 2]
            y2 = centers[:, 1] + d[:, 3]
            boxes.append(np.stack([x1, y1, x2, y2], axis=1))
            scores.append(s[keep])
        if not boxes:
            return []
        boxes = np.concatenate(boxes) / scale
        scores = np.concatenate(scores)
        keep = cv2.dnn.NMSBoxes(
            [(float(x1), float(y1), float(x2 - x1), float(y2 - y1)) for x1, y1, x2, y2 in boxes],
            scores.tolist(), threshold, 0.4)
        return [(*boxes[i], float(scores[i])) for i in np.asarray(keep).flatten()]


def load_face_detector(model_path: Path | None):
    if model_path is None or not model_path.exists():
        return None
    try:
        return FaceDetector(model_path)
    except Exception as e:  # missing onnxruntime, bad model, CUDA issues, ...
        print(f"Face model unavailable ({e}); falling back to detail-based framing.", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- analysis

def detail_center(frame_bgr: np.ndarray, aspect: float) -> float:
    """Horizontal center (0..1) of the most detailed region that fits a window of given aspect."""
    work_w = 480
    h, w = frame_bgr.shape[:2]
    work_h = max(1, round(h * work_w / w))
    gray = cv2.cvtColor(cv2.resize(frame_bgr, (work_w, work_h)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    energy = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=1))
    col = energy.sum(axis=0)
    win = max(1, min(work_w - 1, round(work_h * aspect)))
    rolling = np.convolve(col, np.ones(win), mode="valid")
    positions = (np.arange(rolling.size) + win / 2) / work_w
    rolling *= 1.0 - 0.15 * np.abs(positions - 0.5) * 2  # mild center prior
    return float((int(np.argmax(rolling)) + win / 2) / work_w)


def analyze_clip(video: Path, faces: FaceDetector | None, aspect: float, stride: int):
    """Return (per-frame smoothed-ready raw centers, fps, (w, h), total_frames, face_coverage)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0 or src_w <= 0:
        cap.release()
        return None

    idxs, face_centers, fallback_centers = [], [], []
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i % stride == 0:
            fc = None
            if faces is not None:
                dets = faces.detect(frame)
                if dets:
                    # Weighted mean of face centers; big/confident faces dominate.
                    weights = np.array([(x2 - x1) * (y2 - y1) * s for x1, y1, x2, y2, s in dets])
                    centers = np.array([(x1 + x2) / 2 for x1, y1, x2, y2, s in dets])
                    fc = float((centers * weights).sum() / weights.sum() / src_w)
            idxs.append(frame_i)
            face_centers.append(fc)
            fallback_centers.append(None if fc is not None else detail_center(frame, aspect))
        frame_i += 1
    cap.release()
    if not idxs:
        return None

    total = frame_i  # trust decoded count over metadata
    face_mask = np.array([c is not None for c in face_centers])
    coverage = float(face_mask.mean())

    idxs = np.array(idxs, dtype=np.float64)
    if coverage >= 0.2:
        # Face track: interpolate across face-less gaps instead of letting the
        # detail signal yank the camera somewhere else mid-shot.
        known = idxs[face_mask]
        vals = np.array([c for c in face_centers if c is not None])
        raw = np.interp(idxs, known, vals)
    else:
        raw = np.array([f if f is not None else d
                        for f, d in zip(face_centers, fallback_centers)], dtype=np.float64)

    return raw, idxs, fps, (src_w, src_h), total, coverage


def smooth_track(raw: np.ndarray, idxs: np.ndarray, total: int, fps: float,
                 smooth_seconds: float) -> np.ndarray:
    """Raw sampled centers -> one smooth center per frame."""
    if raw.size >= 5:
        k = min(9, raw.size // 2 * 2 - 1)
        pad = k // 2
        padded = np.pad(raw, pad, mode="edge")
        raw = np.array([np.median(padded[i:i + k]) for i in range(raw.size)])

    per_frame = np.interp(np.arange(total), idxs, raw)

    sigma = max(1.0, fps * smooth_seconds)
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(per_frame, radius, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")

    # Static shot? Lock the window instead of micro-drifting.
    if smoothed.size and (np.percentile(smoothed, 95) - np.percentile(smoothed, 5)) < 0.025:
        smoothed[:] = np.median(smoothed)
    return smoothed


# --------------------------------------------------------------------------- encode

def crop_size(src_w: int, src_h: int, target_w: int, target_h: int) -> tuple[int, int]:
    crop_w = min(src_w, 2 * ((src_h * target_w // target_h) // 2))
    crop_h = min(src_h, 2 * ((crop_w * target_h // target_w) // 2))
    return crop_w, crop_h


def encode_tracked(video: Path, out_path: Path, track: np.ndarray, target_w: int, target_h: int,
                   crf: int, fps: float) -> bool:
    cap = cv2.VideoCapture(str(video))
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_w, crop_h = crop_size(src_w, src_h, target_w, target_h)

    proc = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{target_w}x{target_h}", "-r", f"{fps}", "-i", "-",
        "-i", str(video),
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000",
        "-shortest", str(out_path),
    ], stdin=subprocess.PIPE)

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        center = track[min(i, track.size - 1)] * src_w
        cx = float(np.clip(center, crop_w / 2, src_w - crop_w / 2))
        cy = src_h / 2
        patch = cv2.getRectSubPix(frame, (crop_w, crop_h), (cx, cy))
        out = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        try:
            proc.stdin.write(out.tobytes())
        except (BrokenPipeError, OSError):
            break
        i += 1
    cap.release()
    proc.stdin.close()
    return proc.wait() == 0


# --------------------------------------------------------------------------- main

def parse_size(value: str) -> tuple[int, int]:
    try:
        w, h = value.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Size must look like 1080x1920, got: {value}")


def run_ffmpeg(args_list: list[str]) -> int:
    return subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args_list]).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-crop clips to vertical 9:16 tracking the subject, then stitch.")
    parser.add_argument("input", type=Path, help="Folder of clips, or a single video file")
    parser.add_argument("-o", "--output", help="Output folder (default: <input>_vertical)")
    parser.add_argument("--size", type=parse_size, default=(1080, 1920), help="Output WxH (default 1080x1920)")
    parser.add_argument("--center", type=float, default=None,
                        help="Manual fixed horizontal center 0..1 (skips detection, applies to all inputs)")
    parser.add_argument("--static", action="store_true", help="One fixed window per clip (no tracking)")
    parser.add_argument("--smooth", type=float, default=0.6,
                        help="Tracking smoothness in seconds — higher = lazier camera (default 0.6)")
    parser.add_argument("--detect-stride", type=int, default=2, help="Analyze every Nth frame (default 2)")
    parser.add_argument("--face-model", type=Path, default=DEFAULT_FACE_MODEL,
                        help="SCRFD ONNX face model path (default: insightface buffalo_l det_10g)")
    parser.add_argument("--no-faces", action="store_true", help="Disable face detection, detail energy only")
    parser.add_argument("--crf", type=int, default=20, help="x264 quality (default 20)")
    parser.add_argument("--preview", action="store_true",
                        help="Only write preview JPGs (crop box at mid-frame + travel range) — no encode")
    parser.add_argument("--no-stitch", action="store_true", help="Crop clips but skip final concatenation")
    parser.add_argument("--audio-from", type=Path, default=None,
                        help="Take stitched audio from this file (e.g. the original video)")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}")
    videos = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        sys.exit(f"No video files found in {args.input}")

    base = args.input if args.input.is_dir() else args.input.parent
    out_dir = Path(args.output) if args.output else base.parent / f"{base.name}_vertical"
    out_dir.mkdir(parents=True, exist_ok=True)
    target_w, target_h = args.size
    aspect = target_w / target_h

    faces = None if args.no_faces else load_face_detector(args.face_model)
    if faces is not None:
        print(f"Face detector: {args.face_model.name}")

    cropped = []
    for video in videos:
        analysis = analyze_clip(video, faces, aspect, args.detect_stride) if args.center is None else None
        if args.center is not None:
            cap = cv2.VideoCapture(str(video))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            track = np.full(max(total, 1), args.center)
            coverage = -1.0
        elif analysis is None:
            print(f"FAILED to analyze: {video.name}", file=sys.stderr)
            continue
        else:
            raw, idxs, fps, (src_w, src_h), total, coverage = analysis
            track = smooth_track(raw, idxs, total, fps, args.smooth)
            if args.static:
                track[:] = np.median(track)

        crop_w, crop_h = crop_size(src_w, src_h, target_w, target_h)
        lo, hi = float(track.min()), float(track.max())
        moving = (hi - lo) > 0.001
        tag = f"faces {coverage:.0%}" if coverage > 0 else ("manual" if coverage < 0 else "detail")
        print(f"{video.name}: {tag}, center {lo:.2f}" + (f" -> {hi:.2f} (tracking)" if moving else " (locked)"))

        if args.preview:
            cap = cv2.VideoCapture(str(video))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
            ok, frame = cap.read()
            cap.release()
            if ok:
                cx = track[min(total // 2, track.size - 1)] * src_w
                x = int(np.clip(cx - crop_w / 2, 0, src_w - crop_w))
                y = (src_h - crop_h) // 2
                cv2.rectangle(frame, (x, y), (x + crop_w, y + crop_h), (0, 255, 0), 3)
                for frac in (lo, hi):  # travel range markers
                    px = int(np.clip(frac * src_w, 0, src_w - 1))
                    cv2.line(frame, (px, 0), (px, 40), (0, 200, 255), 3)
                cv2.imwrite(str(out_dir / f"{video.stem}_preview.jpg"), frame)
            continue

        out_path = out_dir / f"{video.stem}_916.mp4"
        if not encode_tracked(video, out_path, track, target_w, target_h, args.crf, fps):
            print(f"ffmpeg failed on {video.name}", file=sys.stderr)
            continue
        cropped.append(out_path)

    if args.preview:
        print(f"\nPreviews -> {out_dir}. Happy? Re-run without --preview.")
        return
    if args.no_stitch or len(cropped) < 2:
        print(f"\nDone. {len(cropped)} vertical clip(s) -> {out_dir}")
        return

    concat_list = out_dir / "concat.txt"
    concat_list.write_text("".join(f"file '{p.name}'\n" for p in cropped), encoding="utf-8")
    stitched = out_dir / "stitched_9x16.mp4"
    if args.audio_from:
        rc = run_ffmpeg([
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-i", str(args.audio_from),
            "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(stitched),
        ])
    else:
        rc = run_ffmpeg([
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c", "copy",
            str(stitched),
        ])
    if rc != 0:
        sys.exit("Concat failed")
    print(f"\nDone. {len(cropped)} clips + stitched video -> {stitched}")


if __name__ == "__main__":
    main()
