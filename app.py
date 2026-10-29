#!/usr/bin/env python3
"""Crop Studio — browser app for scene-split + subject-tracked 9:16 cropping.

Run:  python app.py   (serves http://127.0.0.1:5123)

Open a video (server-side path), cuts are auto-detected, each cut gets an
auto-framed crop position (faces first). Adjust positions / keyframes in the
browser, then export a stitched 9:16 video.
"""

import base64
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file

def _load_dotenv():
    """Tiny .env loader (no dependency) — existing environment wins."""
    import os
    p = Path(__file__).parent / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()  # before outpaint import — it reads COMFY_URLS at import time

from crop_vertical import DEFAULT_FACE_MODEL, detail_center, load_face_detector
import outpaint
import tracker

VERSION = "0.7.0"  # AI shorts suggestions + hook-structure exports
PORT = 5124

app = Flask(__name__, static_folder="static", static_url_path="")

STATE_LOCK = threading.Lock()
state = {
    "phase": "idle",  # idle | detecting | analyzing | ready | exporting | done | error
    "progress": 0.0,
    "message": "",
    "error": None,
    "video": None,     # path str
    "meta": None,      # {duration, fps, width, height}
    "segments": None,  # [{id, start, end, keyframes: [{t, c}], thumb}]
    "ranges": None,    # [{start, end, name}] — manual shorts; empty = export whole video
    "export_paths": None,
    "proxy": None,     # browser-safe playback file when the source isn't (HEVC / 10-bit / 4K)
    "suggest": None,   # AI short-suggestion job {status, message}
}
faces = load_face_detector(DEFAULT_FACE_MODEL)
ASPECT = 9 / 16
ASPECTS = {"9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350)}


def output_size(aspect_key: str, src_w: int, src_h: int):
    """At least the 1080-wide preset; native crop size when the source has more pixels
    (e.g. 4K -> 1216x2160 at 9:16) so high-res footage isn't downscaled on export."""
    pw, ph = ASPECTS.get(aspect_key, ASPECTS["9:16"])
    asp = pw / ph
    crop_h = min(src_w, src_h * asp) / asp
    if crop_h <= ph:
        return pw, ph
    return 2 * round(crop_h * asp / 2), 2 * round(crop_h / 2)


def set_state(**kw):
    with STATE_LOCK:
        state.update(kw)


def video_meta(path: str):
    # ffprobe first: OpenCV's fps/frame-count metadata is unreliable for
    # VFR / HEVC / phone footage and silently truncates the timeline.
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,codec_name,pix_fmt",
             "-show_entries", "format=duration", "-of", "json", path],
            capture_output=True, text=True, timeout=30)
        data = json.loads(out.stdout)
        st = data["streams"][0]
        num, den = (st.get("avg_frame_rate") or "24/1").split("/")
        fps = float(num) / float(den) if float(den or 0) else 24.0
        duration = float(data["format"]["duration"])
        frames = int(st.get("nb_frames") or 0) or int(round(duration * fps))
        if duration > 0 and fps > 0:
            return {"fps": fps, "width": int(st["width"]), "height": int(st["height"]),
                    "frames": frames, "duration": duration,
                    "codec": st.get("codec_name", ""), "pix_fmt": st.get("pix_fmt", "")}
    except Exception:
        pass
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 24.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    meta["duration"] = meta["frames"] / meta["fps"]
    cap.release()
    return meta


def frame_center(frame) -> float:
    """Face-weighted center of one frame, detail fallback. Returns 0..1."""
    if faces is not None:
        dets = faces.detect(frame)
        if dets:
            w = frame.shape[1]
            weights = np.array([(x2 - x1) * (y2 - y1) * s for x1, y1, x2, y2, s in dets])
            centers = np.array([(x1 + x2) / 2 for x1, y1, x2, y2, s in dets])
            return float((centers * weights).sum() / weights.sum() / w)
    return detail_center(frame, ASPECT)


def sidecar(path: str) -> Path:
    return Path(path).with_suffix(Path(path).suffix + ".crop.json")


def ensure_proxy(path: str, meta: dict):
    """Browser-safe playback proxy when the source isn't directly playable.

    Preview/scrubbing use it; export and all ComfyUI stages keep the original.
    """
    safe = (meta.get("codec") in ("h264", "vp9") and meta.get("pix_fmt") == "yuv420p"
            and meta["width"] <= 1920)
    if safe:
        return None
    proxy = outpaint.store_dir(path) / "proxy.mp4"
    src_mtime = Path(path).stat().st_mtime
    if proxy.exists() and proxy.stat().st_mtime >= src_mtime and proxy.stat().st_size > 0:
        return str(proxy)
    set_state(message="Building playback proxy (one-time, source isn't browser-playable)...",
              progress=0.05)
    rc = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path,
        "-vf", "scale='min(1280,iw)':-2", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "2", str(proxy),
    ]).returncode
    if rc != 0 or not proxy.exists():
        return None  # play the original and hope; analysis is unaffected either way
    return str(proxy)


def analyze_worker(path: str):
    try:
        meta = video_meta(path)
        if meta is None:
            set_state(phase="error", error=f"Cannot open video: {path}")
            return
        set_state(proxy=ensure_proxy(path, meta))

        saved = None
        sc = sidecar(path)
        if sc.exists():
            try:
                saved = json.loads(sc.read_text(encoding="utf-8"))
                if abs(saved.get("duration", -1) - meta["duration"]) > 1.0:
                    saved = None  # different file, same name
            except Exception:
                saved = None

        set_state(phase="detecting", message="Detecting cuts...", progress=0.05)
        if saved:
            boundaries = [(s["start"], s["end"]) for s in saved["segments"]]
        else:
            from scenedetect import AdaptiveDetector, detect
            scenes = detect(path, AdaptiveDetector(min_scene_len="0.5s"), show_progress=False)
            boundaries = [(s.seconds, e.seconds) for s, e in scenes] or [(0.0, meta["duration"])]
            # Decoder gave up early (odd container)? Don't lose the tail of the video.
            if meta["duration"] - boundaries[-1][1] > 1.0:
                boundaries[-1] = (boundaries[-1][0], meta["duration"])

        set_state(phase="analyzing", message="Framing segments...", progress=0.1)
        cap = cv2.VideoCapture(path)
        segments = []
        for i, (t0, t1) in enumerate(boundaries):
            mid = (t0 + t1) / 2
            cap.set(cv2.CAP_PROP_POS_MSEC, mid * 1000)
            ok, frame = cap.read()
            thumb = ""
            center = 0.5
            if ok:
                th = cv2.resize(frame, (128, 72))
                thumb = "data:image/jpeg;base64," + base64.b64encode(
                    cv2.imencode(".jpg", th, [cv2.IMWRITE_JPEG_QUALITY, 60])[1]).decode()
                if not saved:
                    # sample 3 frames for the default position
                    cs = [frame_center(frame)]
                    for frac in (0.25, 0.75):
                        cap.set(cv2.CAP_PROP_POS_MSEC, (t0 + (t1 - t0) * frac) * 1000)
                        ok2, f2 = cap.read()
                        if ok2:
                            cs.append(frame_center(f2))
                    center = float(np.median(cs))
            extra = {}
            if saved:
                s_old = saved["segments"][i]
                kfs = s_old["keyframes"]
                if "auto_c" in s_old:
                    extra["auto_c"] = s_old["auto_c"]
                if s_old.get("outpaint"):
                    op = s_old["outpaint"]
                    # a job can't survive an app restart — mark it interrupted
                    if op.get("status") in ("generating", "video_generating"):
                        op["status"] = "error" if not op.get("options") else "ready"
                        op["message"] = "Interrupted by restart — run again"
                    extra["outpaint"] = op
            else:
                kfs = [{"t": round(t0, 3), "c": round(center, 4)}]
                extra["auto_c"] = round(center, 4)  # lets "clear all" return to auto-framing
            segments.append({"id": i, "start": round(t0, 3), "end": round(t1, 3),
                             "keyframes": kfs, "thumb": thumb, **extra})
            set_state(progress=0.1 + 0.9 * (i + 1) / len(boundaries),
                      message=f"Framing segments... {i + 1}/{len(boundaries)}")
        cap.release()
        set_state(phase="ready", video=path, meta=meta, segments=segments,
                  ranges=(saved or {}).get("ranges", []),
                  message="", progress=1.0, export_paths=None)
        save_sidecar()
    except Exception as e:
        set_state(phase="error", error=f"{type(e).__name__}: {e}")


def save_sidecar():
    with STATE_LOCK:
        if not state["video"] or not state["segments"]:
            return
        data = {"duration": state["meta"]["duration"], "segments": state["segments"],
                "ranges": state["ranges"] or []}
        path = sidecar(state["video"])
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def build_tracks(segments, meta):
    """Per-frame (x-center, y-center, zoom) arrays from segment keyframes."""
    fps, total = meta["fps"], meta["frames"]
    cx = np.full(total, 0.5)
    cy = np.full(total, 0.5)
    cz = np.ones(total)
    for seg in segments:
        f0 = int(round(seg["start"] * fps))
        f1 = min(total, int(round(seg["end"] * fps))) if seg is not segments[-1] else total
        if f1 <= f0:
            continue
        kfs = sorted(seg["keyframes"], key=lambda k: k["t"])
        ts = np.array([k["t"] for k in kfs])
        frame_ts = (np.arange(f0, f1) + 0.5) / fps
        cx[f0:f1] = np.interp(frame_ts, ts, np.array([k["c"] for k in kfs]))
        cy[f0:f1] = np.interp(frame_ts, ts, np.array([k.get("y", 0.5) for k in kfs]))
        cz[f0:f1] = np.interp(frame_ts, ts, np.array([max(0.35, k.get("z", 1.0)) for k in kfs]))
    return cx, cy, cz


# ------------------------------------------------------------------ captions

def words_path(path: str) -> Path:
    return Path(path).with_suffix(Path(path).suffix + ".words.json")


# Runs in a separate process: ctranslate2 imports torch, whose CUDA DLLs clash
# with the onnxruntime-GPU DLLs already loaded in the app process.
TRANSCRIBE_SCRIPT = """
import json, sys
from faster_whisper import WhisperModel
path, out = sys.argv[1], sys.argv[2]
model = WhisperModel("small", device="cpu", compute_type="int8")
seg_iter, info = model.transcribe(path, word_timestamps=True)
words = []
for seg in seg_iter:
    for w in seg.words or []:
        words.append({"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)})
    if info.duration:
        print(f"PROG {min(1.0, seg.end / info.duration):.3f}", flush=True)
open(out, "w", encoding="utf-8").write(json.dumps(words, ensure_ascii=False))
"""


def ensure_words(path: str):
    """Transcribe once with faster-whisper (word timestamps), cache next to video."""
    wp = words_path(path)
    if wp.exists():
        return json.loads(wp.read_text(encoding="utf-8"))
    set_state(message="Transcribing audio (one-time, cached)...", progress=0.0)
    proc = subprocess.Popen([sys.executable, "-c", TRANSCRIBE_SCRIPT, path, str(wp)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in proc.stdout:
        if line.startswith("PROG"):
            frac = float(line.split()[1])
            set_state(progress=frac, message=f"Transcribing... {int(frac * 100)}%")
    proc.wait()
    if proc.returncode != 0 or not wp.exists():
        err = (proc.stderr.read() or "")[-400:]
        raise RuntimeError(f"Transcription failed: {err}")
    return json.loads(wp.read_text(encoding="utf-8"))


def ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_ass(words, parts, w: int, h: int, out_file: Path):
    """Chunked lower-third captions across export parts (hook tease + playthrough)."""
    chunks = []  # (short_time_start, short_time_end, text-words)
    offset = 0.0
    for t0, t1 in parts:
        sel = [x for x in words if x["s"] < t1 and x["e"] > t0 and x["w"]]
        cur = []
        for x in sel:
            if cur and (len(cur) >= 4 or x["s"] - cur[-1]["e"] > 0.8):
                chunks.append((max(0.0, cur[0]["s"] - t0) + offset,
                               min(t1 - t0, cur[-1]["e"] - t0 + 0.15) + offset, cur))
                cur = []
            cur.append(x)
        if cur:
            chunks.append((max(0.0, cur[0]["s"] - t0) + offset,
                           min(t1 - t0, cur[-1]["e"] - t0 + 0.15) + offset, cur))
        offset += t1 - t0
    fontsize = round(h * 0.045)
    header = f"""[Script Info]
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Outline, Shadow, Alignment, MarginL, MarginR, MarginV
Style: Cap,Arial,{fontsize},&H00FFFFFF,&H00000000,&H80000000,-1,{max(2, h // 400)},1,2,40,40,{round(h * 0.16)}

[Events]
Format: Layer, Start, End, Style, Text
"""
    lines = []
    for start, end, ch in chunks:
        if end <= start:
            continue
        text = " ".join(x["w"] for x in ch).replace("{", "").replace("}", "")
        lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Cap,{text}")
    out_file.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_ " else "" for ch in name).strip().replace(" ", "_") or "short"


def sample_frame_window(frame, cfrac, yfrac, z, target_w, target_h, crop_w0, crop_h0):
    """One output frame. z>=1: subpixel crop. z<1: window extends beyond frame -> green fill."""
    src_h, src_w = frame.shape[:2]
    if z >= 0.999:
        cw, ch = crop_w0 / z, crop_h0 / z
        cx = float(np.clip(cfrac * src_w, cw / 2, src_w - cw / 2))
        cy = float(np.clip(yfrac * src_h, ch / 2, src_h - ch / 2))
        patch = cv2.getRectSubPix(frame, (max(2, round(cw)), max(2, round(ch))), (cx, cy))
        return cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    x0, y0, win_w, win_h = outpaint.window_geometry(
        src_w, src_h, {"c": cfrac, "y": yfrac, "z": z}, aspect_wh=target_w / target_h)
    canvas = np.full((target_h, target_w, 3), outpaint.GREEN, dtype=np.uint8)
    scale = target_w / win_w
    vx0, vy0 = max(0.0, x0), max(0.0, y0)
    vx1, vy1 = min(src_w, x0 + win_w), min(src_h, y0 + win_h)
    if vx1 > vx0 and vy1 > vy0:
        band = frame[int(vy0):int(round(vy1)), int(vx0):int(round(vx1))]
        dx, dy = round((vx0 - x0) * scale), round((vy0 - y0) * scale)
        dw = min(max(2, round((vx1 - vx0) * scale)), target_w - dx)
        dh = min(max(2, round((vy1 - vy0) * scale)), target_h - dy)
        canvas[dy:dy + dh, dx:dx + dw] = cv2.resize(band, (dw, dh), interpolation=cv2.INTER_LANCZOS4)
    return canvas


class SegmentSplicer:
    """Feeds frames from a segment's generated outpaint video during export."""

    def __init__(self, segments, fps):
        self.jobs = []
        for seg in segments:
            op = seg.get("outpaint") or {}
            if op.get("status") == "video_ready" and op.get("video") and Path(op["video"]).exists():
                self.jobs.append((op.get("seg_start", seg["start"]),
                                  op.get("seg_end", seg["end"]), op["video"]))
        self.fps = fps
        self.cap = None
        self.active = None

    def frame_for(self, t, target_w, target_h):
        """Outpainted frame at global time t, or None to use the normal crop path."""
        job = next(((s, e, v) for s, e, v in self.jobs if s - 1e-6 <= t < e), None)
        if job is None:
            self._close()
            return None
        if self.active != job:
            self._close()
            self.cap = cv2.VideoCapture(job[2])
            self.active = job
            self.vfps = self.cap.get(cv2.CAP_PROP_FPS) or self.fps
            self.vframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.next_idx = 0
        want = int(round((t - job[0]) * self.vfps))
        if want >= self.vframes:
            return None  # generated clip shorter than segment -> crop path takes over
        frame = None
        while self.next_idx <= want:
            ok, frame = self.cap.read()
            if not ok:
                return None
            self.next_idx += 1
        if frame is None:
            return None
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    def _close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.active = None


def export_worker(segments, ranges, default_aspect, crf, captions):
    try:
        with STATE_LOCK:
            path, meta = state["video"], dict(state["meta"])
        fps, total = meta["fps"], meta["frames"]
        src_w, src_h = meta["width"], meta["height"]
        tcx, tcy, tcz = build_tracks(segments, meta)
        splicer = SegmentSplicer(segments, fps)

        words = ensure_words(path) if captions else None

        stem = Path(path).stem
        # Unique names (time range + export time) so editors never see two different
        # exports with the same file name, and re-exports don't overwrite earlier ones.
        stamp = time.strftime("%H%M%S")
        tc = lambda t: f"{int(t // 60)}m{int(t % 60):02d}s"
        if ranges:
            jobs = []
            for i, r in enumerate(ranges):
                parts = []
                hs, he = r.get("hook_start"), r.get("hook_end")
                if hs is not None and he is not None and float(he) > float(hs):
                    parts.append((float(hs), float(he)))  # hook tease first
                parts.append((r["start"], r["end"]))      # then the full playthrough
                jobs.append({"parts": parts,
                             "aspect": r.get("aspect") or default_aspect,
                             "out": str(Path(path).with_name(f"{stem}_{safe_name(r.get('name', f'short-{i+1}'))}_{tc(r['start'])}-{tc(r['end'])}_{stamp}.mp4"))})
        else:
            jobs = [{"parts": [(0.0, meta["duration"])], "aspect": default_aspect,
                     "out": str(Path(path).with_name(f"{stem}_crop_edit_{stamp}.mp4"))}]

        frames_all = sum(max(1, int((t1 - t0) * fps)) for j in jobs for t0, t1 in j["parts"])
        done_frames = 0
        outs = []
        for ji, job in enumerate(jobs):
            target_w, target_h = output_size(job["aspect"], src_w, src_h)
            crop_w0 = min(src_w, src_h * target_w / target_h)
            crop_h0 = crop_w0 * target_h / target_w
            parts = [(t0, t1) for t0, t1 in job["parts"] if t1 > t0]
            if not parts:
                continue

            vf = []
            if words is not None:
                ass = Path(path).with_name(f"_captions_{ji}.ass")
                n = build_ass(words, parts, target_w, target_h, ass)
                if n:
                    escaped = str(ass).replace("\\", "/").replace(":", "\\:")
                    vf = ["-vf", f"subtitles=filename='{escaped}'"]

            # Audio assembled in part order (hook first) via concat of trims.
            atmp = Path(path).with_name(f"_audio_{ji}.m4a")
            trims = "".join(
                f"[0:a]atrim=start={t0:.3f}:end={t1:.3f},asetpts=PTS-STARTPTS[a{k}];"
                for k, (t0, t1) in enumerate(parts))
            concat_in = "".join(f"[a{k}]" for k in range(len(parts)))
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path,
                "-filter_complex", f"{trims}{concat_in}concat=n={len(parts)}:v=0:a=1[out]",
                "-map", "[out]", "-c:a", "aac", "-ar", "48000", str(atmp),
            ])

            proc = subprocess.Popen([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{target_w}x{target_h}",
                "-r", f"{fps}", "-i", "-",
                "-i", str(atmp),
                "-map", "0:v", "-map", "1:a?", *vf,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-c:a", "copy", "-shortest", job["out"],
            ], stdin=subprocess.PIPE)

            # Decode each part via ffmpeg pipe (cv2 seeking is unreliable cross-platform).
            frame_bytes = src_w * src_h * 3
            written = 0
            broken = False
            for t0, t1 in parts:
                if broken:
                    break
                reader = subprocess.Popen([
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", path,
                    "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
                ], stdout=subprocess.PIPE, bufsize=frame_bytes * 4)
                n_part = max(1, int(round((t1 - t0) * fps)))
                for i in range(n_part):
                    buf = reader.stdout.read(frame_bytes)
                    if buf is None or len(buf) < frame_bytes:
                        break
                    frame = np.frombuffer(buf, np.uint8).reshape(src_h, src_w, 3)
                    t_global = t0 + i / fps
                    idx = min(int(t_global * fps), total - 1)
                    out = splicer.frame_for(t_global, target_w, target_h)
                    if out is None:
                        out = sample_frame_window(frame, tcx[idx], tcy[idx], tcz[idx],
                                                  target_w, target_h, crop_w0, crop_h0)
                    try:
                        proc.stdin.write(out.tobytes())
                    except (BrokenPipeError, OSError):
                        broken = True
                        break
                    written += 1
                    done_frames += 1
                    if done_frames % 48 == 0:
                        set_state(progress=done_frames / frames_all,
                                  message=f"Rendering {ji + 1}/{len(jobs)}... {done_frames}/{frames_all}")
                reader.stdout.close()
                reader.wait()
            proc.stdin.close()
            rc = proc.wait()
            if atmp.exists():
                atmp.unlink()
            if written == 0:
                set_state(phase="error",
                          error=f"Decoded 0 frames for {Path(job['out']).name} — source codec issue")
                return
            if words is not None and ass.exists():
                ass.unlink()
            if rc != 0:
                set_state(phase="error", error=f"ffmpeg failed on {Path(job['out']).name}")
                return
            outs.append(job["out"])
        set_state(phase="done", progress=1.0,
                  message=f"Exported {len(outs)} file(s)", export_paths=outs)
    except Exception as e:
        set_state(phase="error", error=f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------ outpaint jobs

def _get_segment(seg_index: int):
    with STATE_LOCK:
        segs = state["segments"]
        if not segs or seg_index < 0 or seg_index >= len(segs):
            return None, None
        return segs[seg_index], state["video"]


def _update_outpaint(seg_index: int, **kw):
    with STATE_LOCK:
        segs = state["segments"]
        if segs and 0 <= seg_index < len(segs):
            op = segs[seg_index].setdefault("outpaint", {})
            op.update(kw)
    save_sidecar()


def outpaint_stills_worker(seg_index: int, t: float, crop_state: dict, prompt: str, n: int):
    seg, video = _get_segment(seg_index)
    try:
        def prog(frac, msg):
            _update_outpaint(seg_index, progress=round(frac, 3), message=msg)
        result = outpaint.flux_outpaint_options(video, seg_index, t, crop_state, prompt, n, prog)
        with STATE_LOCK:
            op = state["segments"][seg_index].setdefault("outpaint", {})
            existing = op.get("options") or []
            uploads = [p for p in existing if "_upload" in Path(p).name]  # keep user uploads
        _update_outpaint(seg_index, status="ready", options=uploads + result["options"],
                         mock=result["mock"], chosen=None, progress=1.0,
                         message=f"Pick one of the {len(uploads) + len(result['options'])} options"
                                 + (" (MOCK)" if result["mock"] else ""))
    except Exception as e:
        _update_outpaint(seg_index, status="error", message=f"{type(e).__name__}: {e}")


def outpaint_video_worker(seg_index: int):
    seg, video = _get_segment(seg_index)
    try:
        op = seg["outpaint"]
        chosen_png = op["options"][op["chosen"]]
        with STATE_LOCK:
            fps = state["meta"]["fps"]
            k = len(op.get("videos") or [])

        def prog(frac, msg):
            _update_outpaint(seg_index, progress=round(frac, 3), message=msg)
        # Times frozen when the options were generated — segment indices can
        # shift via cuts/merges/undo, the outpainted content must not.
        result = outpaint.ltx_outpaint_video(
            video, seg_index,
            op.get("seg_start", seg["start"]), op.get("seg_end", seg["end"]),
            op["crop_state"], chosen_png, fps, op.get("prompt", ""), prog,
            render_tag=f"_r{k}")
        with STATE_LOCK:
            op2 = state["segments"][seg_index]["outpaint"]
            videos = op2.get("videos") or []
            videos.append({"file": result["video"], "still": op["chosen"], "mock": result["mock"]})
            op2.update(videos=videos, active_video=len(videos) - 1)
        _update_outpaint(seg_index, status="video_ready", video=result["video"],
                         mock=result["mock"], progress=1.0,
                         message=f"Render {k + 1} ready — active at export" + (" (MOCK)" if result["mock"] else ""))
    except Exception as e:
        _update_outpaint(seg_index, status="error", message=f"{type(e).__name__}: {e}")


def _set_track_job(seg_index: int, **kw):
    with STATE_LOCK:
        segs = state["segments"]
        if segs and 0 <= seg_index < len(segs):
            job = segs[seg_index].setdefault("track_job", {})
            job.update(kw)


def track_click_worker(seg_index: int, x: float, y: float):
    seg, video = _get_segment(seg_index)
    try:
        kf0 = seg["keyframes"][0]
        base_y, base_z = kf0.get("y", 0.5), max(0.35, kf0.get("z", 1.0))
        y_active = base_z > 1.01  # vertical position only matters when punched in

        def prog(frac, msg):
            _set_track_job(seg_index, status="running", progress=round(frac, 3), message=msg)
        result = tracker.track_point(video, seg["start"], seg["end"], x, y,
                                     base_y, base_z, y_active, prog)
        with STATE_LOCK:
            state["segments"][seg_index]["keyframes"] = result["keyframes"]
            state["segments"][seg_index]["track_job"] = {
                "status": "done", "progress": 1.0,
                "message": f"Tracked ({result['engine']}) — {len(result['keyframes'])} keyframes"}
        save_sidecar()
    except Exception as e:
        _set_track_job(seg_index, status="error", message=f"{type(e).__name__}: {e}")


@app.post("/api/track_click")
def api_track_click():
    data = request.json or {}
    seg_index = int(data.get("seg", -1))
    seg, video = _get_segment(seg_index)
    if seg is None:
        return jsonify({"error": "bad segment"}), 400
    x = min(max(float(data.get("x", 0.5)), 0.0), 1.0)
    y = min(max(float(data.get("y", 0.5)), 0.0), 1.0)
    _set_track_job(seg_index, status="running", progress=0.0, message="Starting...")
    threading.Thread(target=track_click_worker, args=(seg_index, x, y), daemon=True).start()
    return jsonify({"ok": True})


@app.post("/api/outpaint")
def api_outpaint():
    data = request.json or {}
    seg_index = int(data.get("seg", -1))
    seg, video = _get_segment(seg_index)
    if seg is None:
        return jsonify({"error": "bad segment"}), 400
    crop_state = data.get("state") or {"c": 0.5, "y": 0.5, "z": 0.6}
    # Always the segment's FIRST frame: the LTX stage conditions the still onto
    # frame 0 of the band clip — a mid-segment still would contradict the guide
    # video there and lose most of its steering power (verified by A/B 2026-07-27).
    t = seg["start"] + 0.04
    prompt = (data.get("prompt") or "").strip()
    n = max(1, min(8, int(data.get("n", 4))))
    _update_outpaint(seg_index, status="generating", progress=0.0, message="Starting...",
                     crop_state=crop_state, t=t, prompt=prompt,
                     chosen=None, video=None,
                     seg_start=seg["start"], seg_end=seg["end"])
    threading.Thread(target=outpaint_stills_worker,
                     args=(seg_index, t, crop_state, prompt, n), daemon=True).start()
    return jsonify({"ok": True})


@app.post("/api/outpaint/upload")
def api_outpaint_upload():
    """User-provided outpainted still becomes a selectable option."""
    seg_index = request.form.get("seg", -1, type=int)
    seg, video = _get_segment(seg_index)
    f = request.files.get("file")
    if seg is None or f is None:
        return jsonify({"error": "bad request"}), 400
    try:
        crop_state = json.loads(request.form.get("state") or "")
    except (json.JSONDecodeError, TypeError):
        crop_state = {"c": 0.5, "y": 0.5, "z": 0.6}
    out_dir = outpaint.store_dir(video)
    existing = (seg.get("outpaint") or {}).get("options") or []
    k = sum(1 for p in existing if "_upload" in Path(p).name)
    p = out_dir / f"seg{seg_index}_upload{k}.png"
    f.save(str(p))
    with STATE_LOCK:
        op = state["segments"][seg_index].setdefault("outpaint", {})
        opts = op.get("options") or []
        opts.insert(0, str(p))
        op.update(options=opts, status="ready", chosen=0,
                  message="Uploaded image selected",
                  crop_state=op.get("crop_state", crop_state),
                  seg_start=op.get("seg_start", seg["start"]),
                  seg_end=op.get("seg_end", seg["end"]),
                  prompt=op.get("prompt", ""))
    save_sidecar()
    return jsonify({"ok": True, "index": 0})


@app.post("/api/outpaint/usevideo")
def api_outpaint_usevideo():
    data = request.json or {}
    seg_index, idx = int(data.get("seg", -1)), int(data.get("index", -1))
    seg, _ = _get_segment(seg_index)
    op = (seg or {}).get("outpaint") or {}
    videos = op.get("videos") or []
    if not 0 <= idx < len(videos):
        return jsonify({"error": "bad index"}), 400
    _update_outpaint(seg_index, video=videos[idx]["file"], active_video=idx)
    return jsonify({"ok": True})


@app.post("/api/outpaint/choose")
def api_outpaint_choose():
    data = request.json or {}
    seg_index, choice = int(data.get("seg", -1)), int(data.get("choice", -1))
    seg, _ = _get_segment(seg_index)
    if seg is None or not (seg.get("outpaint") or {}).get("options"):
        return jsonify({"error": "no options"}), 400
    if not 0 <= choice < len(seg["outpaint"]["options"]):
        return jsonify({"error": "bad choice"}), 400
    _update_outpaint(seg_index, chosen=choice)
    return jsonify({"ok": True})


@app.post("/api/outpaint/tovideo")
def api_outpaint_tovideo():
    data = request.json or {}
    seg_index = int(data.get("seg", -1))
    seg, _ = _get_segment(seg_index)
    op = (seg or {}).get("outpaint") or {}
    if not op.get("options") or op.get("chosen") is None:
        return jsonify({"error": "choose an option first"}), 400
    _update_outpaint(seg_index, status="video_generating", progress=0.0, message="Starting LTX...")
    threading.Thread(target=outpaint_video_worker, args=(seg_index,), daemon=True).start()
    return jsonify({"ok": True})


@app.post("/api/outpaint/discard")
def api_outpaint_discard():
    data = request.json or {}
    seg_index = int(data.get("seg", -1))
    seg, _ = _get_segment(seg_index)
    if seg is None:
        return jsonify({"error": "bad segment"}), 400
    with STATE_LOCK:
        state["segments"][seg_index].pop("outpaint", None)
    save_sidecar()
    return jsonify({"ok": True})


@app.get("/api/outpaint/file")
def api_outpaint_file():
    seg_index = request.args.get("seg", -1, type=int)
    seg, _ = _get_segment(seg_index)
    op = (seg or {}).get("outpaint") or {}
    if request.args.get("video"):
        vids = op.get("videos") or []
        vi = request.args.get("i", -1, type=int)
        p = vids[vi]["file"] if 0 <= vi < len(vids) else op.get("video")
    else:
        i = request.args.get("i", 0, type=int)
        opts = op.get("options") or []
        p = opts[i] if 0 <= i < len(opts) else None
    if not p or not Path(p).exists():
        return "Not found", 404
    return send_file(p, conditional=True)


# ------------------------------------------------------------------ AI short suggestions

SUGGEST_PROMPT = """You select clips for vertical short-form video (Reels/TikTok/Shorts) from a transcript.
Below is a transcript with [start-end] second timestamps per line.

Return STRICT JSON only — an array of at most {n} suggestions:
[{{"start": <sec>, "end": <sec>, "title": "<short catchy title>", "hook_start": <sec>, "hook_end": <sec>, "reason": "<one sentence>"}}]

Rules:
- Each short: 15-45 seconds, a self-contained idea/story/punchline with a natural end.
- hook_start/hook_end: the single most attention-grabbing 2-6 second moment INSIDE [start, end] — the climax, punchline, boldest claim, or biggest reaction. It will play FIRST as a tease, then the short plays from the beginning and the hook lands again in place as the payoff.
- If no good hook exists for a short, set hook_start and hook_end to null.
- Prefer fewer, stronger suggestions over padding to {n}.

Transcript:
{transcript}
"""


def _transcript_lines(words, max_chars=24000):
    lines, cur, t0 = [], [], None
    for w in words:
        if t0 is None:
            t0 = w["s"]
        cur.append(w["w"])
        if w["e"] - t0 > 8 or len(cur) > 28:
            lines.append(f"[{t0:.1f}-{w['e']:.1f}] " + " ".join(cur))
            cur, t0 = [], None
    if cur:
        lines.append(f"[{t0:.1f}-{words[-1]['e']:.1f}] " + " ".join(cur))
    text = "\n".join(lines)
    return text[:max_chars]


def _llm(prompt: str) -> str:
    """Try in order: Anthropic API key -> claude CLI -> local Ollama."""
    import os
    import shutil
    import urllib.request
    errors = []

    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages", method="POST",
                data=json.dumps({"model": "claude-sonnet-5-5", "max_tokens": 2000,
                                 "messages": [{"role": "user", "content": prompt}]}).encode(),
                headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["content"][0]["text"]
        except Exception as e:
            errors.append(f"API: {e}")

    claude = shutil.which("claude")
    if claude:
        try:
            proc = subprocess.run([claude, "-p"], input=prompt, capture_output=True, text=True,
                                  timeout=300, encoding="utf-8", errors="replace")
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout
            errors.append(f"claude CLI: {(proc.stderr or proc.stdout or 'no output')[:120]}")
        except Exception as e:
            errors.append(f"claude CLI: {e}")

    ollama_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    try:
        with urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=5) as r:
            models = [m["name"] for m in json.loads(r.read()).get("models", [])]
        if models:
            model = os.environ.get("OLLAMA_MODEL") or models[0]
            req = urllib.request.Request(
                f"{ollama_url}/api/chat", method="POST",
                data=json.dumps({"model": model, "stream": False,
                                 "options": {"num_ctx": 16384, "temperature": 0.4},
                                 "messages": [{"role": "user", "content": prompt}]}).encode(),
                headers={"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())["message"]["content"]
        errors.append("Ollama: running but no models pulled")
    except Exception as e:
        errors.append(f"Ollama: {e}")

    raise RuntimeError(
        "No LLM reachable. Fix any one of: set ANTHROPIC_API_KEY / run `claude` once in a "
        "terminal to refresh login / install+run Ollama (ollama.com) with a model pulled. "
        + " | ".join(errors))


def _parse_suggestions(text: str, duration: float):
    m = text[text.find("["):text.rfind("]") + 1]
    items = json.loads(m)
    out = []
    for it in items:
        try:
            s, e = float(it["start"]), float(it["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= s < e <= duration + 1) or not (8 <= e - s <= 90):
            continue
        r = {"start": round(s, 2), "end": round(min(e, duration), 2),
             "name": safe_name(str(it.get("title") or f"suggested-{len(out) + 1}"))[:40]}
        hs, he = it.get("hook_start"), it.get("hook_end")
        if hs is not None and he is not None:
            hs, he = float(hs), float(he)
            if s <= hs < he <= e and 1 <= he - hs <= 10:
                r["hook_start"], r["hook_end"] = round(hs, 2), round(he, 2)
        out.append(r)
    return out


def suggest_worker(n: int):
    try:
        with STATE_LOCK:
            path, duration = state["video"], state["meta"]["duration"]
        set_state(suggest={"status": "running", "message": "Transcribing (cached after first run)..."})
        words = ensure_words(path)
        if len(words) < 20:
            set_state(suggest={"status": "error",
                               "message": "Not enough speech found in this video for transcript-based suggestions"})
            return
        set_state(suggest={"status": "running", "message": "Asking Claude for short suggestions..."})
        text = _llm(SUGGEST_PROMPT.format(n=n, transcript=_transcript_lines(words)))
        suggestions = _parse_suggestions(text, duration)
        if not suggestions:
            set_state(suggest={"status": "error", "message": "No usable suggestions returned"})
            return
        with STATE_LOCK:
            ranges = state["ranges"] or []
            ranges.extend(suggestions)
            ranges.sort(key=lambda r: r["start"])
            state["ranges"] = ranges
        save_sidecar()
        set_state(suggest={"status": "done",
                           "message": f"Added {len(suggestions)} suggested short(s)"})
    except Exception as e:
        set_state(suggest={"status": "error", "message": f"{type(e).__name__}: {e}"})


@app.post("/api/suggest_shorts")
def api_suggest_shorts():
    with STATE_LOCK:
        if not state["video"]:
            return jsonify({"error": "no video open"}), 400
    n = max(1, min(8, int((request.json or {}).get("n", 3))))
    threading.Thread(target=suggest_worker, args=(n,), daemon=True).start()
    return jsonify({"ok": True})


# ------------------------------------------------------------------ project snapshots

def _projects_dir():
    with STATE_LOCK:
        video = state["video"]
    if not video:
        return None
    d = outpaint.store_dir(video) / "projects"
    d.mkdir(exist_ok=True)
    return d


@app.post("/api/project/save")
def api_project_save():
    name = safe_name((request.json or {}).get("name") or "")
    d = _projects_dir()
    if not d or not name:
        return jsonify({"error": "no video open / bad name"}), 400
    with STATE_LOCK:
        data = {"duration": state["meta"]["duration"],
                "segments": state["segments"], "ranges": state["ranges"] or []}
    (d / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
    return jsonify({"ok": True})


@app.get("/api/project/list")
def api_project_list():
    d = _projects_dir()
    if not d:
        return jsonify({"projects": []})
    return jsonify({"projects": sorted(p.stem for p in d.glob("*.json"))})


@app.post("/api/project/load")
def api_project_load():
    name = safe_name((request.json or {}).get("name") or "")
    d = _projects_dir()
    p = (d / f"{name}.json") if d else None
    if not p or not p.exists():
        return jsonify({"error": "not found"}), 404
    data = json.loads(p.read_text(encoding="utf-8"))
    set_state(segments=data["segments"], ranges=data.get("ranges") or [])
    save_sidecar()
    with STATE_LOCK:
        return jsonify({"segments": state["segments"], "ranges": state["ranges"]})


# ------------------------------------------------------------------ routes

@app.get("/")
def index():
    return app.send_static_file("index.html")


BROWSE_SCRIPT = """
import tkinter as tk
from tkinter import filedialog
root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
path = filedialog.askopenfilename(title="Choose a video", filetypes=[
    ("Video", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.mpg *.mpeg *.wmv *.flv *.ts"),
    ("All files", "*.*")])
print(path or "", end="")
"""
BROWSE_LOCK = threading.Lock()


@app.post("/api/browse")
def api_browse():
    """Open a native file-picker dialog on this machine, return the chosen path."""
    if not BROWSE_LOCK.acquire(blocking=False):
        return jsonify({"error": "Dialog already open"}), 409
    try:
        result = subprocess.run([sys.executable, "-c", BROWSE_SCRIPT],
                                capture_output=True, text=True, timeout=300)
        return jsonify({"path": result.stdout.strip() or None})
    except subprocess.TimeoutExpired:
        return jsonify({"path": None})
    finally:
        BROWSE_LOCK.release()


@app.post("/api/open")
def api_open():
    path = (request.json or {}).get("path", "").strip().strip('"')
    if not path or not Path(path).is_file():
        return jsonify({"error": f"File not found: {path}"}), 400
    set_state(phase="detecting", progress=0.0, message="Opening...", error=None,
              video=path, segments=None, export_path=None)
    threading.Thread(target=analyze_worker, args=(path,), daemon=True).start()
    return jsonify({"ok": True})


@app.get("/api/status")
def api_status():
    with STATE_LOCK:
        return jsonify({"version": VERSION, **{k: state[k] for k in
                        ("phase", "progress", "message", "error", "video", "meta",
                         "segments", "ranges", "export_paths", "suggest")}})


@app.get("/video")
def video():
    with STATE_LOCK:
        path = state["proxy"] or state["video"]
    if not path:
        return "No video", 404
    return send_file(path, conditional=True)


@app.post("/api/save")
def api_save():
    data = request.json or {}
    segs = data.get("segments")
    if not segs:
        return jsonify({"error": "no segments"}), 400
    set_state(segments=segs, ranges=data.get("ranges") or [])
    save_sidecar()
    return jsonify({"ok": True})


@app.post("/api/track")
def api_track():
    """Auto-track one segment: returns keyframes every ~0.4s following the subject."""
    data = request.json or {}
    t0, t1 = float(data["start"]), float(data["end"])
    with STATE_LOCK:
        path, meta = state["video"], state["meta"]
    if not path:
        return jsonify({"error": "no video"}), 400
    fps = meta["fps"]
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
    stride = 2
    ts, cs = [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = t0 + i / fps
        if t >= t1:
            break
        if i % stride == 0:
            ts.append(t)
            cs.append(frame_center(frame))
        i += 1
    cap.release()
    if not ts:
        return jsonify({"keyframes": [{"t": t0, "c": 0.5}]})
    cs = np.array(cs)
    if cs.size >= 5:  # median filter outliers
        k = min(7, cs.size // 2 * 2 - 1)
        pad = k // 2
        padded = np.pad(cs, pad, mode="edge")
        cs = np.array([np.median(padded[j:j + k]) for j in range(cs.size)])
    # gaussian smooth ~0.5s
    sigma = max(1.0, fps * 0.5 / stride)
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kern = np.exp(-0.5 * (x / sigma) ** 2)
    kern /= kern.sum()
    cs = np.convolve(np.pad(cs, radius, mode="edge"), kern, mode="valid")
    if (np.percentile(cs, 95) - np.percentile(cs, 5)) < 0.02:  # static shot
        return jsonify({"keyframes": [{"t": round(t0, 3), "c": round(float(np.median(cs)), 4)}]})
    step = max(1, int(0.4 * fps / stride))  # keyframe every ~0.4s
    kfs = [{"t": round(ts[j], 3), "c": round(float(cs[j]), 4)} for j in range(0, len(ts), step)]
    if kfs[-1]["t"] < ts[-1]:
        kfs.append({"t": round(ts[-1], 3), "c": round(float(cs[-1]), 4)})
    return jsonify({"keyframes": kfs})


@app.post("/api/export")
def api_export():
    data = request.json or {}
    segs = data.get("segments")
    if not segs:
        return jsonify({"error": "no segments"}), 400
    ranges = data.get("ranges") or []
    set_state(segments=segs, ranges=ranges, phase="exporting", progress=0.0,
              message="Rendering...", error=None, export_paths=None)
    save_sidecar()
    threading.Thread(target=export_worker, args=(
        segs, ranges, data.get("aspect", "9:16"),
        int(data.get("crf", 20)), bool(data.get("captions"))), daemon=True).start()
    return jsonify({"ok": True})


@app.get("/api/result")
def api_result():
    idx = request.args.get("i", 0, type=int)
    with STATE_LOCK:
        paths = state["export_paths"] or []
    if idx < 0 or idx >= len(paths) or not Path(paths[idx]).exists():
        return "No export", 404
    return send_file(paths[idx], conditional=True,
                     as_attachment=bool(request.args.get("dl")))


if __name__ == "__main__":
    print(f"Crop Studio v{VERSION} -> http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, threaded=True)
