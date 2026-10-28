# Crop Studio

Turn horizontal video into vertical shorts, in your browser. Auto-detects cuts,
auto-frames every scene on faces/subjects, and gives you per-scene crop keyframes,
click-to-track (SAM 3), AI outpainting beyond the frame (Flux + LTX-2.3 via ComfyUI),
Whisper captions, and transcript-based AI short suggestions with hook-structure exports
(tease → playthrough → payoff).

## Online version

`docs/` is a browser-only build of the core editor, served by GitHub Pages at
https://crop.jas.black. Cut detection, face auto-framing, crop keyframes, shorts and
export all run locally in the browser (WebCodecs), so videos are never uploaded.
AI outpainting, click-to-track, captions and AI-suggested shorts need this desktop app.
Project files (`<video>.crop.json`) work in both.

To publish: repo Settings → Pages → Deploy from branch → `main` / `/docs`, then point a
`CNAME` DNS record for `crop` at `<your-username>.github.io`. Set `REPO_URL` near the
top of the script in `docs/index.html` to this repo's URL.

## Run the desktop app

```sh
# needs Python 3.10+ and ffmpeg on PATH (winget install ffmpeg / brew install ffmpeg)
pip install -r requirements.txt
cp .env.example .env    # optional: ComfyUI / LLM endpoints
python app.py           # → http://127.0.0.1:5124
```

Open a video, adjust the green crop window per scene, mark shorts with the I/O keys,
hit Export. Everything auto-saves next to the source video (`<video>.crop.json`,
`<video>.cropstudio/`); reopening is instant.

Without a ComfyUI server or an LLM configured the app still fully works — AI outpainting
runs as a labeled mock, subject tracking falls back to optical flow, and Suggest explains
what it needs. Face detection is bundled (`models/det_10g.onnx`), no downloads.

**Setting up the AI features** (captions, Suggest shorts, ComfyUI outpainting and SAM tracking,
with links to every model): see [`SETUP.md`](SETUP.md).

See `CLAUDE.md` for architecture, current status, and hard-won gotchas. Standalone CLI
versions of the core steps: `split_scenes.py` (cut splitting), `crop_vertical.py`
(auto-crop + stitch), `export_frames.py` (green 9:16 frames for outpaint workflows).

Mac note: if the Browse dialog doesn't appear (Homebrew Python lacks tkinter:
`brew install python-tk`), paste the file path instead.
