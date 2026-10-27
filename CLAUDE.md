# Crop Studio

Browser-based tool that turns horizontal video into vertical (9:16 / 1:1 / 4:5) shorts.
It auto-detects cuts, auto-frames each cut on faces/subjects, lets you adjust crop keyframes
per scene, and adds AI outpainting (Flux Fill stills + LTX-2.3 video via ComfyUI), SAM-3
click-to-track, Whisper captions, and LLM-suggested shorts with hook-structure exports.

## Tech stack

- **Backend:** Python 3.10+ / Flask (single process, worker threads for long jobs), OpenCV,
  numpy, ffmpeg (must be on PATH), faster-whisper (captions + transcripts),
  onnxruntime (SCRFD face detection, model bundled in `models/`).
- **Frontend:** one vanilla-JS single page, no build step (`static/index.html`).
- **External compute (optional):** any ComfyUI server carrying `flux1-fill-dev`,
  the LTX-2.3 22B checkpoint + `ltx-2.3-22b-ic-lora-in-outpainting-0.9`, and
  `sam3.1_multiplex_fp16` — configured via `COMFY_URLS`. Without one the app still
  works: outpainting falls back to a labeled mock, tracking to local optical flow.
- **LLM (optional, for Suggest shorts):** `ANTHROPIC_API_KEY` → `claude` CLI → local Ollama,
  first available wins.

## Folder structure

- `app.py` — Flask app: scene detection, state + sidecar persistence, export pipeline
  (hook-structure assembly, caption burn, outpaint splicing), project snapshots,
  AI suggest, playback-proxy builder. Port 5124.
- `outpaint.py` — outpaint engine: window geometry, Flux Fill options, LTX video renders,
  band-tone color matching, mock fallbacks. ComfyUI server picker.
- `tracker.py` — click-to-track: SAM-3 via ComfyUI, optical-flow fallback, anti-jitter
  smoothing → keyframes.
- `comfy_client.py` — minimal ComfyUI HTTP client (upload / queue / poll / fetch /free),
  with vanished-job detection.
- `crop_vertical.py` — standalone CLI: face-tracked auto-crop + stitch (also imported by
  the app for the face detector + detail saliency).
- `split_scenes.py`, `export_frames.py` — standalone CLIs: cut-splitting; green-padded
  9:16 frame export for outpaint workflows.
- `static/index.html` — the whole UI.
- `docs/` — browser-only web version (GitHub Pages, crop.jas.black): a port of
  `static/index.html` where Flask calls are replaced by Mediabunny (WebCodecs decode/encode,
  `docs/vendor/`) and MediaPipe BlazeFace. Ports of `detail_center`, `frame_center`,
  `/api/track` smoothing and an adaptive cut detector live in its script. Desktop-only
  features (outpaint, SAM tracking, captions, suggest, zoom < 1) are removed there and
  listed in a "Desktop app" card. Keep both UIs in sync when changing shared editor logic.
- `workflows/*.api.json` — ComfyUI graphs. The two `*_extracted*` files were extracted
  verbatim from the metadata of known-good renders — treat as authoritative; the app only
  patches filenames/pads/prompts/seeds at runtime.
- `models/det_10g.onnx` — bundled SCRFD face detector (insightface buffalo_l).

Per-video working data lives NEXT TO each source video, not in the repo:
`<video>.crop.json` (auto-saved project), `<video>.words.json` (transcript cache),
`<video>.cropstudio/` (outpaint stills/renders, playback proxy, snapshots).

## Install / run

```sh
# ffmpeg required on PATH (winget install ffmpeg / brew install ffmpeg)
pip install -r requirements.txt
cp .env.example .env   # optional — fill in what you have
python app.py          # → http://127.0.0.1:5124
```

No build step. No deployment target — it's a local tool (Flask dev server, binds 127.0.0.1).

## Environment variables (all optional — names only)

- `COMFY_URLS` — comma-separated ComfyUI endpoints, first reachable wins
- `ANTHROPIC_API_KEY` — for Suggest shorts via Anthropic API
- `OLLAMA_URL`, `OLLAMA_MODEL` — for Suggest shorts via local Ollama

## Status

Works (all validated end-to-end on real footage):
- Cut detection w/ instant reopen, face/detail auto-framing, keyframes, 2D crop + zoom
  (incl. zoom-out beyond frame), subject click-tracking, undo/redo, snapshots (Save as/Load),
  shorts with In/Out + per-range aspect, AI-suggested shorts with hooks, hook-structure
  export (tease → playthrough → payoff), Whisper caption burn, outpaint stills (N options +
  own-image upload), multiple LTX renders with compare/select, export splicing of outpainted
  segments, playback proxy for non-browser codecs (HEVC/10-bit/4K).

Known limits / unfinished:
- LTX outpaint renders max 720x1280 (hard VRAM limit on 24 GB cards — see gotchas) and
  ~7.9 s per segment; longer segments fall back to plain crop for the tail.
- LTX's first ~2 frames smear (reconciling still vs motion) — app holds frame 3 instead;
  occasionally still visible.
- Suggest quality depends on speech; music-only content is refused.
- Single-user, single open video at a time (global server state by design).
- Flask dev server only; no auth — do not expose beyond localhost.

## Gotchas (hard-won — read before "improving")

- **opencv-python wheel 4.13.0.92 actually ships OpenCV 5.0**: `cv2.CascadeClassifier` /
  haar are REMOVED. Face detection is SCRFD via onnxruntime with a custom decoder
  (`crop_vertical.py`).
- **Never trust cv2 metadata or cv2 seeking.** Durations come from ffprobe; ALL frame
  reads (export, still grabs, tracking) go through ffmpeg pipes. cv2 seek silently fails
  on Mac/HEVC/VFR and once produced 261-byte exports.
- **Outpaint seam quality**: the extracted Flux workflow's own recipe (feather 40,
  DifferentialDiffusion, NO post-compositing) is optimal. Wider feathers ghost busy
  content; re-compositing the source band back creates hard lines. A/B-tested.
- **LTX resolution is capped by per-frame attention memory, not pixels x frames**:
  864x1536 and 1088x1920 both OOM-crashed ComfyUI on a clean 24 GB card. Don't retry
  without side-by-side evidence on bigger hardware. ESRGAN post-upscale was tried and
  looks worse on painted content (off by default, code kept).
- **The LTX guide still MUST come from the segment's first frame** — a mid-segment still
  contradicts the frame-0 condition and loses its steering (A/B-verified). The app
  enforces this; outpaint jobs freeze their time range at generation time so later
  cuts/merges can't shift them.
- **Call `/free` on ComfyUI before the LTX pass** (evicts warm Flux weights; 22B needs
  the whole card). Already wired in.
- **PowerShell writes UTF-8 BOM** — workflow JSONs load with `utf-8-sig`.
- Negative prompt for Flux outpaint must include text/typography terms or it hallucinates
  poster text on vague prompts.
- Band tone of LTX renders drifts lighter — fixed-per-render color match against the
  chosen still runs automatically (`_match_band_colors`).
- claude CLI headless (`claude -p`, prompt on stdin) fails if its OAuth is stale; an
  interactive `claude` run refreshes it. Ollama is the zero-auth fallback.
