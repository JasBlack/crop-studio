#!/usr/bin/env python3
"""Click-to-track subject for Crop Studio v0.5.

User clicks a point on the subject; we produce smooth crop keyframes that keep
it in frame for the whole segment.

Engine 1 (preferred): SAM 3.1 on ComfyUI Server A — point prompt -> mask ->
SAM3_VideoTrack across the segment -> per-frame mask centroids.
Engine 2 (fallback, offline): pyramidal Lucas-Kanade optical flow on a point
grid around the click.

Either way the raw path goes through median + gaussian smoothing before being
thinned to keyframes, so the result never jitters.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from comfy_client import ComfyError
from outpaint import pick_server, store_dir

TRACK_W = 640          # analysis resolution for both engines
KEYFRAME_EVERY_S = 0.4


# ------------------------------------------------------------------ shared

def _smooth(xs: np.ndarray, fps: float, smooth_seconds: float = 0.5) -> np.ndarray:
    if xs.size >= 5:
        k = min(9, xs.size // 2 * 2 - 1)
        pad = k // 2
        padded = np.pad(xs, pad, mode="edge")
        xs = np.array([np.median(padded[i:i + k]) for i in range(xs.size)])
    sigma = max(1.0, fps * smooth_seconds)
    radius = int(3 * sigma)
    kern = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kern /= kern.sum()
    return np.convolve(np.pad(xs, radius, mode="edge"), kern, mode="valid")


def _to_keyframes(ts, cxs, cys, fps: float, seg_start: float, base_y: float, base_z: float,
                  y_active: bool):
    """Smooth raw per-frame centers and thin to keyframes."""
    cxs = _smooth(np.asarray(cxs, dtype=np.float64), fps)
    cys = _smooth(np.asarray(cys, dtype=np.float64), fps)
    if (np.percentile(cxs, 95) - np.percentile(cxs, 5)) < 0.02:
        cxs[:] = np.median(cxs)
    if (np.percentile(cys, 95) - np.percentile(cys, 5)) < 0.02:
        cys[:] = np.median(cys)
    step = max(1, int(KEYFRAME_EVERY_S * fps))
    idxs = list(range(0, len(ts), step))
    if idxs[-1] != len(ts) - 1:
        idxs.append(len(ts) - 1)
    kfs = []
    for i in idxs:
        kf = {"t": round(float(ts[i]), 3), "c": round(float(cxs[i]), 4), "z": base_z}
        kf["y"] = round(float(cys[i]), 4) if y_active else base_y
        kfs.append(kf)
    return kfs


def _extract_frames(video_path: str, seg_start: float, seg_end: float):
    """Segment frames downscaled to TRACK_W via ffmpeg pipe (cv2 seek is
    unreliable on some codec/platform combos). Returns (frames, fps)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap.release()
    th = max(2, round(src_h * TRACK_W / src_w) // 2 * 2)
    frame_bytes = TRACK_W * th * 3
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-ss", f"{seg_start:.3f}", "-to", f"{seg_end:.3f}", "-i", video_path,
         "-vf", f"scale={TRACK_W}:{th},fps={fps:.6f}",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, bufsize=frame_bytes * 4)
    frames = []
    n_max = int((seg_end - seg_start) * fps) + 1
    while len(frames) < n_max:
        buf = proc.stdout.read(frame_bytes)
        if buf is None or len(buf) < frame_bytes:
            break
        frames.append(np.frombuffer(buf, np.uint8).reshape(th, TRACK_W, 3).copy())
    proc.stdout.close()
    proc.wait()
    if not frames:
        raise RuntimeError("No frames in segment")
    return frames, fps


# ------------------------------------------------------------------ SAM3 engine

def _sam3_track(client, frames, fps: float, click_x: float, click_y: float, on_progress):
    """Track via SAM 3.1 on the ComfyUI server. Returns (cxs, cys) fractions per frame."""
    h, w = frames[0].shape[:2]
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "track_clip.mp4"
        writer = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(clip),
        ], stdin=subprocess.PIPE)
        for f in frames:
            writer.stdin.write(f.tobytes())
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError("track clip encode failed")
        on_progress(0.1, "Uploading to SAM 3.1...")
        remote = client.upload(clip.read_bytes(), "cropstudio_track.mp4")

    px, py = int(click_x * w), int(click_y * h)
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"}},
        "2": {"class_type": "VHS_LoadVideo",
              "inputs": {"video": remote, "force_rate": 0, "custom_width": 0, "custom_height": 0,
                         "frame_load_cap": 0, "skip_first_frames": 0, "select_every_nth": 1,
                         "format": "AnimateDiff"}},
        "3": {"class_type": "GetImageRangeFromBatch",
              "inputs": {"start_index": 0, "num_frames": 1, "images": ["2", 0]}},
        "4": {"class_type": "SAM3_Detect",
              "inputs": {"model": ["1", 0], "image": ["3", 0], "threshold": 0.5,
                         "refine_iterations": 1, "individual_masks": False,
                         "positive_coords": json.dumps([{"x": px, "y": py}])}},
        "5": {"class_type": "SAM3_VideoTrack",
              "inputs": {"images": ["2", 0], "model": ["1", 0], "detection_threshold": 0.95,
                         "max_objects": 1, "detect_interval": 100, "initial_mask": ["4", 0]}},
        "6": {"class_type": "SAM3_TrackToMask",
              "inputs": {"track_data": ["5", 0], "object_indices": ""}},
        "7": {"class_type": "MaskToImage", "inputs": {"mask": ["6", 0]}},
        "8": {"class_type": "VHS_VideoCombine",
              "inputs": {"frame_rate": float(fps), "loop_count": 0,
                         "filename_prefix": "cropstudio/track_mask", "format": "video/h264-mp4",
                         "pix_fmt": "yuv420p", "crf": 12, "save_metadata": False,
                         "trim_to_audio": False, "pingpong": False, "save_output": True,
                         "images": ["7", 0]}},
    }
    pid = client.queue(wf)

    def tick(pos, elapsed):
        on_progress(min(0.8, 0.15 + elapsed / 90),
                    f"SAM 3.1 tracking... {int(elapsed)}s" if pos is None else f"Queued (position {pos + 1})...")

    outputs = client.wait(pid, timeout=600, on_tick=tick)
    files = client.collect_files(outputs, "gifs") or client.collect_files(outputs, "images")
    mp4s = [f for f in files if f["filename"].endswith(".mp4")]
    if not mp4s:
        raise ComfyError("SAM3 track returned no mask video")
    on_progress(0.85, "Reading mask track...")
    data = client.fetch(mp4s[-1])
    tmp = Path(tempfile.gettempdir()) / "cropstudio_mask_track.mp4"
    tmp.write_bytes(data)

    cap = cv2.VideoCapture(str(tmp))
    cxs, cys = [], []
    last = (click_x, click_y)
    while True:
        ok, mframe = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(mframe, cv2.COLOR_BGR2GRAY)
        m = gray > 96
        if m.sum() > 16:
            ys, xs = np.nonzero(m)
            last = (float(xs.mean()) / gray.shape[1], float(ys.mean()) / gray.shape[0])
        cxs.append(last[0])
        cys.append(last[1])
    cap.release()
    if not cxs:
        raise ComfyError("Empty mask track")
    return cxs, cys


# ------------------------------------------------------------------ optical-flow engine

def _lk_track(frames, click_x: float, click_y: float, on_progress):
    """Local fallback: LK optical flow on a point grid around the click."""
    h, w = frames[0].shape[:2]
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def seed(cx_px, cy_px):
        off = np.linspace(-0.04, 0.04, 5)
        pts = [[cx_px + dx * w, cy_px + dy * h] for dx in off for dy in off]
        return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    pts = seed(click_x * w, click_y * h)
    cxs, cys = [click_x], [click_y]
    for i in range(1, len(grays)):
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(grays[i - 1], grays[i], pts, None, **lk)
        good = nxt[st.flatten() == 1]
        if len(good) < 6:  # track collapsed — reseed at last known center
            pts = seed(cxs[-1] * w, cys[-1] * h)
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(grays[i - 1], grays[i], pts, None, **lk)
            good = nxt[st.flatten() == 1]
            if len(good) == 0:
                cxs.append(cxs[-1]); cys.append(cys[-1])
                continue
        cx = float(np.median(good[:, 0, 0])) / w
        cy = float(np.median(good[:, 0, 1])) / h
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        cxs.append(cx); cys.append(cy)
        pts = nxt[st.flatten() == 1].reshape(-1, 1, 2)
        if i % 24 == 0:
            on_progress(0.1 + 0.8 * i / len(grays), f"Optical-flow tracking... {i}/{len(grays)}")
    return cxs, cys


# ------------------------------------------------------------------ entry

def track_point(video_path: str, seg_start: float, seg_end: float,
                click_x: float, click_y: float, base_y: float, base_z: float,
                y_active: bool, on_progress) -> dict:
    """Click at (click_x, click_y) fractions of the source frame at seg start.

    Returns {"keyframes": [...], "engine": "sam3"|"flow"}.
    """
    on_progress(0.02, "Reading segment frames...")
    frames, fps = _extract_frames(video_path, seg_start, seg_end)
    ts = [seg_start + i / fps for i in range(len(frames))]

    client = pick_server()
    engine = "flow"
    cxs = cys = None
    if client is not None:
        try:
            cxs, cys = _sam3_track(client, frames, fps, click_x, click_y, on_progress)
            engine = "sam3"
        except Exception as e:
            on_progress(0.1, f"SAM3 failed ({e}); falling back to optical flow")
    if cxs is None:
        cxs, cys = _lk_track(frames, click_x, click_y, on_progress)

    n = min(len(ts), len(cxs))
    kfs = _to_keyframes(ts[:n], cxs[:n], cys[:n], fps, seg_start, base_y, base_z, y_active)
    on_progress(1.0, f"Tracked with {engine}: {len(kfs)} keyframes")
    return {"keyframes": kfs, "engine": engine}
