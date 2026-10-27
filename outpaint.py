#!/usr/bin/env python3
"""Outpainting engine for Crop Studio v0.5.

Stage 1 — Flux Fill (Server A): the visible band of a zoomed-out crop window is
outpainted top+bottom into a 9:16 still. 4 seed variants, user picks one.
Stage 2 — LTX-2.3 v2v (Server A): the segment's band clip + the chosen still
drive the user's proven In/Out-painting IC-LoRA workflow (two-pass, Laplacian
blend, audio carried through) to outpaint the whole segment.

Both workflow templates were extracted verbatim from the metadata of the user's
own working ComfyUI outputs (outpainting_00015_.png / ltx-outpaint__00006_.mp4)
— only file names, pads, prompts and seeds get patched per run.

When the ComfyUI servers are unreachable, a local mock (blur-extend) keeps the
UI flow fully testable.
"""

import json
import os
import random
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from comfy_client import ComfyClient, ComfyError, load_template

# Candidate ComfyUI servers, first reachable wins. Override/extend with the
# COMFY_URLS env var (comma-separated, e.g. your remote GPU box) — see .env.example.
COMFY_CANDIDATES = [u.strip() for u in os.environ.get(
    "COMFY_URLS", "http://127.0.0.1:8000,http://127.0.0.1:8188").split(",") if u.strip()]
COMFY_SERVER = None  # resolved lazily by pick_server()
LTX_TARGET_W, LTX_TARGET_H = 720, 1280      # baseline target (also mock canvas ratio)
MAX_LTX_SECONDS = 10.0
# Resolution is HARD-CAPPED at 720x1280 for this workflow on the studio 4090s:
# both 1088x1920 (@128f) and 864x1536 (@127f) crashed ComfyUI outright even on a
# freshly restarted card (2026-07-29) — attention memory scales with per-frame
# resolution, not just pixels x frames. Quality above 720p comes from the
# RealESRGAN x2 post-pass instead (ESRGAN_UPSCALE below).
LTX_SIZE_LADDER = [(720, 1280)]
PIXEL_FRAME_BUDGET = 175_000_000  # length cap: ~7.9s at 720x1280
ESRGAN_UPSCALE = False  # tried 2026-07-29: ESRGAN sharpening reads as plastic on painted content — off by default


def _pick_ltx_target(n_frames: int):
    """Largest safe render size for this frame count; may cap frames at minimum size."""
    for w, h in LTX_SIZE_LADDER:
        if w * h * n_frames <= PIXEL_FRAME_BUDGET:
            return w, h, n_frames
    w, h = LTX_SIZE_LADDER[-1]
    return w, h, max(24, PIXEL_FRAME_BUDGET // (w * h))
WORKFLOW_DIR = Path(__file__).parent / "workflows"
GREEN = (69, 190, 108)  # BGR

FLUX_TEMPLATE = WORKFLOW_DIR / "flux_outpaint_extracted.api.json"
LTX_TEMPLATE = WORKFLOW_DIR / "ltx_outpaint_extracted.api.json"
ESRGAN_TEMPLATE = WORKFLOW_DIR / "esrgan_x2.api.json"
# Empirical (4 real A/B runs on Server A, 2026-07-27): the extracted template's own
# recipe — feather 40, DifferentialDiffusion, no post-compositing — gives the
# cleanest joins. Wider feathers ghost busy content (collars); re-compositing the
# pristine band back leaves a hard line where Flux had already diverged. Don't
# "improve" this again without side-by-side evidence.
FLUX_FEATHER = 40


def pick_server():
    """First reachable ComfyUI from the candidate list, else None (mock mode)."""
    for url in COMFY_CANDIDATES:
        client = ComfyClient(url)
        if client.alive():
            return client
    return None


def store_dir(video_path: str) -> Path:
    p = Path(video_path)
    d = p.parent / f"{p.stem}.cropstudio"
    d.mkdir(exist_ok=True)
    return d


# ------------------------------------------------------------------ geometry

def window_geometry(src_w: int, src_h: int, state: dict, aspect_wh: float = 9 / 16):
    """Crop window rect in source px for {c, y, z}. z<1 extends beyond the frame."""
    z = max(0.35, float(state.get("z", 1.0)))
    win_h = src_h / z
    win_w = win_h * aspect_wh
    if win_w > src_w:
        win_w = src_w
        win_h = win_w / aspect_wh
    c = 0.5 if win_w >= src_w else min(max(state.get("c", 0.5), win_w / 2 / src_w), 1 - win_w / 2 / src_w)
    y = 0.5 if win_h >= src_h else min(max(state.get("y", 0.5), win_h / 2 / src_h), 1 - win_h / 2 / src_h)
    return c * src_w - win_w / 2, y * src_h - win_h / 2, win_w, win_h


def band_rect(src_w: int, src_h: int, state: dict):
    """Visible part of the window + pads (source px). Returns (x0,y0,x1,y1, pad_top, pad_bottom)."""
    x0, y0, win_w, win_h = window_geometry(src_w, src_h, state)
    vx0, vy0 = max(0, int(round(x0))), max(0, int(round(y0)))
    vx1, vy1 = min(src_w, int(round(x0 + win_w))), min(src_h, int(round(y0 + win_h)))
    pad_top = max(0, vy0 - int(round(y0)))
    pad_bottom = max(0, int(round(y0 + win_h)) - vy1)
    return vx0, vy0, vx1, vy1, pad_top, pad_bottom


def grab_frame(video_path: str, t: float):
    """Single frame at time t. ffmpeg first — cv2's time-seek silently fails on
    some codec/platform combos (Mac AVFoundation, HEVC, external volumes)."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-ss", f"{max(0.0, t):.3f}", "-i", video_path,
         "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True)
    if proc.returncode == 0 and proc.stdout:
        img = cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame at {t:.2f}s")
    return frame


# ------------------------------------------------------------------ mock helpers

def _mock_extend(band: np.ndarray, pad_top: int, pad_bottom: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = band.shape[:2]
    canvas = np.zeros((pad_top + h + pad_bottom, w, 3), dtype=np.uint8)
    canvas[pad_top:pad_top + h] = band
    if pad_top:
        top = cv2.resize(cv2.flip(band[: max(2, min(h, pad_top))], 0), (w, pad_top))
        canvas[:pad_top] = cv2.GaussianBlur(top, (0, 0), 14)
    if pad_bottom:
        bot = cv2.resize(cv2.flip(band[-max(2, min(h, pad_bottom)):], 0), (w, pad_bottom))
        canvas[pad_top + h:] = cv2.GaussianBlur(bot, (0, 0), 14)
    tint = rng.integers(-16, 16, size=3)
    mask = np.ones((canvas.shape[0], 1), dtype=bool)
    mask[pad_top:pad_top + h] = False
    for ch in range(3):
        col = canvas[:, :, ch].astype(np.int16)
        col[mask[:, 0]] = np.clip(col[mask[:, 0]] + int(tint[ch]), 0, 255)
        canvas[:, :, ch] = col.astype(np.uint8)
    return canvas


# ------------------------------------------------------------------ flux stage

def flux_outpaint_options(video_path: str, seg_id: int, t: float, state: dict,
                          prompt: str, n: int, on_progress) -> dict:
    out_dir = store_dir(video_path)
    frame = grab_frame(video_path, t)
    src_h, src_w = frame.shape[:2]
    vx0, vy0, vx1, vy1, pad_top, pad_bottom = band_rect(src_w, src_h, state)
    band = frame[vy0:vy1, vx0:vx1]
    # Cap Flux input width — 4K bands would blow past Flux's comfortable ~4MP total
    MAX_BAND_W = 1536
    if band.shape[1] > MAX_BAND_W:
        s = MAX_BAND_W / band.shape[1]
        band = cv2.resize(band, (MAX_BAND_W, max(2, round(band.shape[0] * s))),
                          interpolation=cv2.INTER_AREA)
        pad_top, pad_bottom = round(pad_top * s), round(pad_bottom * s)
    ok, band_png = cv2.imencode(".png", band)
    if not ok:
        raise RuntimeError("PNG encode failed")

    client = pick_server()
    if client is not None:
        remote = client.upload(band_png.tobytes(), f"cropstudio_band_seg{seg_id}.png")
        template = load_template(FLUX_TEMPLATE)
        pids, wfs = [], []
        for i in range(n):
            wf = json.loads(json.dumps(template))
            wf["17"]["inputs"]["image"] = remote
            wf["46"]["inputs"]["top"] = int(pad_top)
            wf["46"]["inputs"]["bottom"] = int(pad_bottom)
            wf["46"]["inputs"]["feathering"] = FLUX_FEATHER
            if prompt:
                wf["23"]["inputs"]["text"] = prompt
            # generic prompts ("background") make Flux hallucinate poster text sometimes
            wf["7"]["inputs"]["text"] = "blurred, out of focus, text, letters, typography, writing, watermark, caption"
            wf["3"]["inputs"]["seed"] = random.randrange(2 ** 48)
            wf["9"]["inputs"]["filename_prefix"] = f"cropstudio/outpaint_seg{seg_id}"
            wfs.append(wf)
            pids.append(client.queue(wf))
        paths = []
        for i, pid in enumerate(pids):
            on_progress(i / n, f"Flux outpaint {i + 1}/{n}...")
            try:
                outputs = client.wait(pid, timeout=600)
            except ComfyError as e:
                if "vanished" not in str(e):
                    raise
                # server dropped the job (restart/eviction) — requeue once
                on_progress(i / n, f"Job {i + 1} lost by server, requeueing...")
                pid = client.queue(wfs[i])
                outputs = client.wait(pid, timeout=600)
            files = client.collect_files(outputs, "images")
            if not files:
                raise ComfyError("Flux job returned no image")
            p = out_dir / f"seg{seg_id}_opt{i}.png"
            p.write_bytes(client.fetch(files[0]))
            paths.append(str(p))
            on_progress((i + 1) / n, f"Flux outpaint {i + 1}/{n} done")
        return {"options": paths, "mock": False}

    paths = []
    for i in range(n):
        on_progress(i / n, f"Mock outpaint {i + 1}/{n} (ComfyUI unreachable)...")
        time.sleep(0.8)
        canvas = _mock_extend(band, pad_top, pad_bottom, seed=seg_id * 31 + i)
        p = out_dir / f"seg{seg_id}_opt{i}.png"
        cv2.imwrite(str(p), canvas)
        paths.append(str(p))
    on_progress(1.0, "Mock outpaint done (connect studio ComfyUI for real results)")
    return {"options": paths, "mock": True}


# ------------------------------------------------------------------ ltx stage

COLOR_MATCH_BANDS = True
COLOR_MATCH_RAMP = 48  # px feather where the correction fades out toward the source band
TRIM_GLITCH_FRAMES = 2  # LTX's first frames smear while reconciling still vs motion — hold frame N instead


def _band_zones(result_h: int, pad_top: int, band_h: int, pad_bottom: int):
    """Outpainted zone rows in result pixel space: (top_end, bottom_start)."""
    total = max(1, pad_top + band_h + pad_bottom)
    return round(pad_top / total * result_h), round((pad_top + band_h) / total * result_h)


def _zone_correction(zone_res: np.ndarray, zone_ref: np.ndarray):
    """Per-channel affine (a, b) mapping result zone stats onto reference stats."""
    a = np.ones(3, np.float32)
    b = np.zeros(3, np.float32)
    for ch in range(3):
        rs, rm = float(zone_res[..., ch].std()) + 1e-4, float(zone_res[..., ch].mean())
        fs, fm = float(zone_ref[..., ch].std()) + 1e-4, float(zone_ref[..., ch].mean())
        a[ch] = min(1.5, max(0.6, fs / rs))
        b[ch] = fm - a[ch] * rm
    return a, b


def _match_band_colors(result_path: Path, chosen_png: str, pad_top: int, band_h: int,
                       pad_bottom: int, on_progress, audio_src: Path = None):
    """Correct the outpainted zones' tone to match the chosen still, all frames alike."""
    cap = cv2.VideoCapture(str(result_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, f0 = cap.read()
    if not ok:
        cap.release()
        return
    ref = cv2.resize(cv2.imread(chosen_png), (w, h), interpolation=cv2.INTER_AREA)
    top_end, bot_start = _band_zones(h, pad_top, band_h, pad_bottom)

    corrections = []  # (row_slice, a, b, ramp_at_end: 'bottom'|'top')
    if top_end > 12 and pad_top > 0:
        a, b = _zone_correction(f0[:top_end].astype(np.float32), ref[:top_end].astype(np.float32))
        corrections.append((0, top_end, a, b, "bottom"))
    if h - bot_start > 12 and pad_bottom > 0:
        a, b = _zone_correction(f0[bot_start:].astype(np.float32), ref[bot_start:].astype(np.float32))
        corrections.append((bot_start, h, a, b, "top"))
    if not corrections:
        cap.release()
        return

    def apply(frame):
        for y0, y1, a, b, ramp_edge in corrections:
            zone = frame[y0:y1].astype(np.float32)
            corrected = zone * a[None, None, :] + b[None, None, :]
            n = y1 - y0
            r = min(COLOR_MATCH_RAMP, n // 2)
            alpha = np.ones((n, 1, 1), np.float32)
            if r > 0:
                if ramp_edge == "bottom":  # zone above the band: fade correction near its bottom
                    alpha[-r:, 0, 0] = np.linspace(1.0, 0.0, r)
                else:                       # zone below the band: fade near its top
                    alpha[:r, 0, 0] = np.linspace(0.0, 1.0, r)
            frame[y0:y1] = np.clip(alpha * corrected + (1 - alpha) * zone, 0, 255).astype(np.uint8)
        return frame

    tmp = result_path.with_suffix(".cc.mp4")
    proc = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
        "-i", str(audio_src or result_path),
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-shortest", str(tmp),
    ], stdin=subprocess.PIPE)
    # Read ahead past LTX's smeared opening frames; hold the first clean frame for those slots.
    lead = [f0]
    for _ in range(TRIM_GLITCH_FRAMES):
        ok, nxt = cap.read()
        if not ok:
            break
        lead.append(nxt)
    clean = apply(lead[-1].copy())
    for _ in range(len(lead)):
        proc.stdin.write(clean.tobytes())
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        proc.stdin.write(apply(frame).tobytes())
    cap.release()
    proc.stdin.close()
    if proc.wait() == 0:
        result_path.unlink()
        tmp.rename(result_path)
        on_progress(0.99, "Band tone matched to chosen still")
    elif tmp.exists():
        tmp.unlink()

def _render_band_clip(video_path: str, seg_start: float, seg_end: float, state: dict,
                      out_path: Path, fps: float):
    """Cut the segment and crop it to the window's visible band (audio kept).

    Forced CFR: VFR sources (phone/4K footage) otherwise carry odd timestamps
    into the LTX pipeline and the result comes back time-stretched.
    """
    cap = cv2.VideoCapture(video_path)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    vx0, vy0, vx1, vy1, _, _ = band_rect(src_w, src_h, state)
    w = (vx1 - vx0) // 2 * 2
    h = (vy1 - vy0) // 2 * 2
    end = min(seg_end, seg_start + MAX_LTX_SECONDS)
    rc = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{seg_start:.3f}", "-to", f"{end:.3f}", "-i", video_path,
        "-vf", f"crop={w}:{h}:{vx0}:{vy0},fps={fps:.6f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", str(out_path),
    ]).returncode
    if rc != 0:
        raise RuntimeError("Band clip render failed")


def ltx_outpaint_video(video_path: str, seg_id: int, seg_start: float, seg_end: float,
                       state: dict, chosen_png: str, fps: float, prompt: str, on_progress,
                       render_tag: str = "") -> dict:
    out_dir = store_dir(video_path)
    band_clip = out_dir / f"seg{seg_id}_band.mp4"
    n_frames = int(min(seg_end - seg_start, MAX_LTX_SECONDS) * fps)
    tw, th, capped_frames = _pick_ltx_target(n_frames)
    capped_end = min(seg_end, seg_start + capped_frames / fps)
    on_progress(0.02, f"Cutting segment band clip ({tw}x{th}"
                + (f", capped to {capped_frames / fps:.1f}s" if capped_frames < n_frames else "") + ")...")
    _render_band_clip(video_path, seg_start, capped_end, state, band_clip, fps)
    result_path = out_dir / f"seg{seg_id}_outpaint{render_tag}.mp4"

    client = pick_server()
    if client is not None:
        on_progress(0.05, "Uploading clip + chosen still...")
        clip_name = client.upload(band_clip.read_bytes(), f"cropstudio_band_seg{seg_id}.mp4")
        still_name = client.upload(Path(chosen_png).read_bytes(), f"cropstudio_chosen_seg{seg_id}.png")
        client.free()  # evict warm Flux weights etc. — the 22B LTX pass needs the whole card
        wf = load_template(LTX_TEMPLATE)
        wf["5363"]["inputs"]["target_width"] = tw
        wf["5363"]["inputs"]["target_height"] = th
        wf["5368"]["inputs"]["file"] = clip_name
        wf["2004"]["inputs"]["image"] = still_name
        if prompt:
            wf["2483"]["inputs"]["text"] = prompt
        seed = random.randrange(2 ** 31)
        wf["4832"]["inputs"]["noise_seed"] = seed
        wf["5210"]["inputs"]["noise_seed"] = seed
        wf["5228"]["inputs"]["filename_prefix"] = f"cropstudio/ltx_seg{seg_id}_hi"
        wf["5394"]["inputs"]["filename_prefix"] = f"cropstudio/ltx_seg{seg_id}"
        pid = client.queue(wf)

        def tick(pos, elapsed):
            msg = f"Queued on server (position {pos + 1})..." if pos is not None else f"LTX rendering... {int(elapsed)}s"
            on_progress(min(0.9, 0.1 + elapsed / 500), msg)

        outputs = client.wait(pid, timeout=2400, on_tick=tick)
        files = client.collect_files(outputs, "images") + client.collect_files(outputs, "gifs")
        mp4s = [f for f in files if f["filename"].lower().endswith((".mp4", ".webm", ".mov"))]
        if not mp4s:
            raise ComfyError(f"LTX job returned no video (files: {[f['filename'] for f in files]})")
        # native-resolution output (the '_hi' variant is just a lanczos 2x — ESRGAN below is better)
        mp4s.sort(key=lambda f: ("_hi" in f["filename"], f["filename"]))
        on_progress(0.88, "Downloading result...")
        raw_path = out_dir / f"seg{seg_id}_raw{render_tag}.mp4"
        raw_path.write_bytes(client.fetch(mp4s[0]))

        final_src = raw_path
        if ESRGAN_UPSCALE:
            try:
                on_progress(0.9, "ESRGAN 2x upscale...")
                client.free()
                up_name = client.upload(raw_path.read_bytes(), f"cropstudio_up_seg{seg_id}.mp4")
                uwf = load_template(ESRGAN_TEMPLATE)
                uwf["1"]["inputs"]["video"] = up_name
                uwf["5"]["inputs"]["frame_rate"] = float(fps)
                uwf["5"]["inputs"]["filename_prefix"] = f"cropstudio/up_seg{seg_id}"
                upid = client.queue(uwf)
                uout = client.wait(upid, timeout=1200)
                ufiles = [f for f in client.collect_files(uout, "gifs") + client.collect_files(uout, "images")
                          if f["filename"].endswith(".mp4")]
                if ufiles:
                    up_path = out_dir / f"seg{seg_id}_up.mp4"
                    up_path.write_bytes(client.fetch(ufiles[-1]))
                    final_src = up_path
            except Exception as e:
                import traceback
                traceback.print_exc()  # lands in the app server log
                on_progress(0.92, f"Upscale skipped ({type(e).__name__}: {e}) — using native resolution")

        if final_src != result_path:
            if result_path.exists():
                result_path.unlink()
            final_src.replace(result_path)
        if COLOR_MATCH_BANDS:
            on_progress(0.97, "Matching band tone to chosen still...")
            frame = grab_frame(video_path, seg_start)
            vx0, vy0, vx1, vy1, pad_top, pad_bottom = band_rect(frame.shape[1], frame.shape[0], state)
            # audio always from the band clip: exact segment timing, and some LTX
            # output variants come back without an audio stream
            _match_band_colors(result_path, chosen_png, pad_top, vy1 - vy0, pad_bottom,
                               on_progress, audio_src=band_clip)
        on_progress(1.0, "Done")
        return {"video": str(result_path), "mock": False}

    # mock: chosen still's bands held static around the moving band clip
    on_progress(0.1, "Mock video (ComfyUI unreachable)...")
    chosen = cv2.imread(chosen_png)
    cap = cv2.VideoCapture(str(band_clip))
    bw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    bh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    total_h = round(bw * LTX_TARGET_H / LTX_TARGET_W) // 2 * 2
    chosen = cv2.resize(chosen, (bw, total_h))
    frame = grab_frame(video_path, seg_start)
    _, _, _, _, pad_top, pad_bottom = band_rect(frame.shape[1], frame.shape[0], state)
    win_total = max(1, pad_top + bh + pad_bottom)
    y_band = min(max(0, round(pad_top * total_h / win_total)), total_h - bh)
    writer = cv2.VideoWriter(str(result_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (bw, total_h))
    i = 0
    while True:
        ok, band = cap.read()
        if not ok:
            break
        out = chosen.copy()
        out[y_band:y_band + bh] = band
        writer.write(out)
        i += 1
        if i % 24 == 0:
            on_progress(min(0.95, 0.1 + 0.85 * i / n_frames), f"Mock video frame {i}/{n_frames}")
    cap.release()
    writer.release()
    on_progress(1.0, "Mock video done (connect studio ComfyUI for real results)")
    return {"video": str(result_path), "mock": True}
