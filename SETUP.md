# Desktop setup guide

The desktop app has everything the [online version](https://crop.jas.black) has, plus AI
features that need extra software or a GPU. Every extra is optional: set up only the ones
you want. The app runs fine without them; it just skips or falls back.

| Feature | What it needs | Setup |
|---|---|---|
| Editor, cut detection, face framing, export | Python + ffmpeg | [1. Basic install](#1-basic-install) |
| Burned-in captions | nothing extra (downloads a model on first use) | [2. Captions](#2-captions-whisper) |
| Suggest shorts | an Anthropic API key, the `claude` CLI, or Ollama | [3. Suggest shorts](#3-suggest-shorts-llm) |
| Click-to-track subjects | ComfyUI + SAM 3.1 (falls back to optical flow without it) | [4. ComfyUI](#4-comfyui-for-ai-outpainting-and-tracking) |
| AI outpainting (stills) | ComfyUI + FLUX.1 Fill | [4. ComfyUI](#4-comfyui-for-ai-outpainting-and-tracking) |
| AI outpainting (video) | ComfyUI + LTX-2.3, **24 GB GPU** | [4. ComfyUI](#4-comfyui-for-ai-outpainting-and-tracking) |

---

## 1. Basic install

You need **Python 3.10+** and **ffmpeg**.

| | Python | ffmpeg |
|---|---|---|
| macOS | `brew install python python-tk` | `brew install ffmpeg` |
| Windows | [python.org/downloads](https://www.python.org/downloads/) (tick "Add to PATH") | `winget install ffmpeg` |
| Linux | `sudo apt install python3 python3-venv python3-tk` | `sudo apt install ffmpeg` |

No Homebrew on Mac? Install it from [brew.sh](https://brew.sh).

Then, in the `crop-studio` folder:

```sh
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
python app.py
```

Open **http://127.0.0.1:5124** in your browser. Face detection is bundled in
`models/det_10g.onnx`, so nothing else to download.

> **Mac:** if the **Browse** button does nothing, `python-tk` is missing. Install it (above) or paste
> the video's full path into the box instead.

All the settings below go in the `.env` file you just copied. Restart `python app.py` after editing it.

---

## 2. Captions (Whisper)

Nothing to set up. The first time you tick **Burn auto-captions** and export, the app downloads
the Whisper `small` model (about 500 MB) via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
and transcribes the video. The transcript is cached next to the video as `<video>.words.json`,
so later exports are instant. It runs on the CPU; expect roughly real-time speed.

---

## 3. Suggest shorts (LLM)

**Suggest** reads the transcript and proposes shorts with hook intros. It needs one of these;
the app tries them in this order and uses the first that works:

1. **Anthropic API key** (best results, paid per use). Create a key at
   [console.anthropic.com](https://console.anthropic.com/) and add it to `.env`:
   ```sh
   ANTHROPIC_API_KEY=sk-ant-...
   ```
2. **Claude Code CLI** (uses your Claude subscription). Install it from
   [code.claude.com/docs](https://code.claude.com/docs), then run `claude` once to sign in.
   If Suggest later says the CLI failed, run `claude` again to refresh the sign-in.
3. **Ollama** (free, runs locally, no account). Install from [ollama.com/download](https://ollama.com/download), then:
   ```sh
   ollama pull gemma3:12b
   ```
   The defaults in `.env` (`OLLAMA_URL=http://127.0.0.1:11434`, `OLLAMA_MODEL=gemma3:12b`) already
   point at it. A different model works too: change `OLLAMA_MODEL`.

Suggest needs speech. Music-only videos have no transcript to work from, so it will decline.

---

## 4. ComfyUI (for AI outpainting and tracking)

Outpainting and SAM tracking run on a [ComfyUI](https://github.com/comfyanonymous/ComfyUI) server.
It can be on the same computer or another machine with a big GPU.

### 4.1 Install ComfyUI

- Easiest: **ComfyUI Desktop** from [comfy.org/download](https://www.comfy.org/download), or
- Manual: follow the install steps in the [ComfyUI README](https://github.com/comfyanonymous/ComfyUI#installing).

**Update ComfyUI to the latest version.** The SAM 3.1 and LTX audio/video nodes are built in,
but only in recent releases.

### 4.2 Install the custom nodes

Install these three node packs. The easiest way is [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager)
(**Manager → Custom Nodes Manager**, search by name). Or clone each one into `ComfyUI/custom_nodes/`
and run `pip install -r requirements.txt` inside it:

| Node pack | Used for |
|---|---|
| [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) | LTX video outpainting (IC-LoRA guide, blending, tiled decode) |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | outpaint padding, batch helpers |
| [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | loading and saving video for tracking |

Restart ComfyUI after installing.

### 4.3 Download the models

The workflows refer to these **exact file names**, so keep the names as they are. Paths are inside
your `ComfyUI/models/` folder. Some Hugging Face pages require you to sign in and accept the
model's license before the download button appears.

**Click-to-track (SAM 3.1)**

| File | Download | Put in |
|---|---|---|
| `sam3.1_multiplex_fp16.safetensors` | [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1/blob/main/checkpoints/sam3.1_multiplex_fp16.safetensors) | `checkpoints/` |

**Outpainting stills (FLUX.1 Fill)**

| File | Download | Put in |
|---|---|---|
| `flux1-fill-dev.safetensors` | [black-forest-labs/FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/blob/main/flux1-fill-dev.safetensors) (accept license) | `diffusion_models/` |
| `ae.safetensors` | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/ae.safetensors) | `vae/FLUX1/` ← note the subfolder |
| `clip_l.safetensors` | [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors) | `text_encoders/` |
| `t5xxl_fp16.safetensors` | [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors) | `text_encoders/` |

**Outpainting video (LTX-2.3)**: needs a **24 GB GPU** (e.g. RTX 3090/4090)

| File | Download | Put in |
|---|---|---|
| `ltx-2.3-22b-dev.safetensors` | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-dev.safetensors) | `checkpoints/` |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-distilled-lora-384-1.1.safetensors) | `loras/` |
| `ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors` | [Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting/blob/main/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors) | `loras/` |
| `gemma_3_12B_it_fp4_mixed.safetensors` | [Comfy-Org/ltx-2](https://huggingface.co/Comfy-Org/ltx-2/blob/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors) | `text_encoders/` |

**Optional: ESRGAN upscale.** Off by default (`ESRGAN_UPSCALE` in `outpaint.py`) because it looks
plastic on painted content. Only needed if you turn it on:

| File | Download | Put in |
|---|---|---|
| `RealESRGAN_x2.pth` | [ai-forever/Real-ESRGAN](https://huggingface.co/ai-forever/Real-ESRGAN/blob/main/RealESRGAN_x2.pth) | `upscale_models/` |

These are big downloads (tens of GB in total, most of it LTX). Download only the groups you
need: tracking alone needs just the SAM file.

### 4.4 Connect Crop Studio to ComfyUI

Start ComfyUI, note its address (usually `http://127.0.0.1:8188`), and set it in `.env`:

```sh
COMFY_URLS=http://127.0.0.1:8188
```

Several servers can be listed, comma-separated; the first one that responds is used.

**ComfyUI on another machine** (e.g. a GPU PC on your network): start it with
`python main.py --listen 0.0.0.0` (Desktop: Settings → Server Config → enable listening on the
network), then use that machine's address, e.g. `COMFY_URLS=http://192.168.1.50:8188`.

> ComfyUI has **no password**. Only expose it on your home network or over a private VPN such as
> [Tailscale](https://tailscale.com/), never directly to the internet.

### 4.5 Check it works

- **Outpaint:** zoom the crop below 1.0x and click **Outpaint**. If the card shows **MOCK**, the
  app couldn't reach ComfyUI. Check `COMFY_URLS` and that ComfyUI is running.
- **Tracking:** click **Track subject**, then click a person. The status shows
  "SAM 3.1 tracking..." when the server is used, or "falling back to optical flow" with the reason
  when it isn't.
- If a job fails with a "missing node" or "value not in list" error, a node pack or model file
  is missing or misnamed. Compare against the tables above. ComfyUI's own console window shows
  exactly which one.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ffmpeg` not found | Install it (section 1) and open a new terminal so PATH updates. |
| Browse button does nothing (Mac) | `brew install python-tk`, or paste the file path. |
| LTX render crashes ComfyUI (out of memory) | Needs a 24 GB card with nothing else loaded. Close other GPU apps. Output is capped at 720x1280 on purpose; higher resolutions run out of memory. |
| Outpaint shows MOCK | ComfyUI not reachable. Check `COMFY_URLS`, firewall, and `--listen` on remote machines. |
| Suggest: "claude CLI" error | Run `claude` once interactively to refresh its sign-in, or use an API key / Ollama. |
| HEVC / iPhone video won't preview | The app builds a playback copy automatically the first time; wait for it to finish. |

## Licenses of the models

Check each model's license before commercial use. In particular:

- **FLUX.1 Fill [dev]** is released under the FLUX.1 [dev] Non-Commercial License (see its model page).
- **LTX-2.3** and its LoRAs have their own license on the [Lightricks model page](https://huggingface.co/Lightricks/LTX-2.3).
- The bundled face detector `models/det_10g.onnx` comes from [insightface](https://github.com/deepinsight/insightface),
  whose pretrained models are for non-commercial research use.

The Crop Studio code itself is MIT licensed (see `LICENSE`).
