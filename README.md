# QuirkFlow

An autonomous, human-in-the-loop video production pipeline with an integrated browser studio WebUI, FastAPI orchestrator, HyperFrames motion-graphics bridge, and embedded Antigravity (`agy`) Co-Pilot.

## Features

- **8-Stage Video Pipeline**
  1. `ingest` — Local FFmpeg 16 kHz mono extraction, media probing, luminance contact sheet, and `faster-whisper` word-level transcription
  2. `script` — Multi-option script generation and adversarial retention critic pass + **Gate A** human approval
  3. `edl` — Edit Decision List generation with automated adjacent-segment overlap validation
  4. `base_cut` — Local cut assembly, stutter detection, script-diff QA, and loudness metrics
  5. `bgm_broll` — Background music selection, sidechain ducking setup, and **Gate B** audio review
  6. `graphics` — Procedural GSAP + SVG motion graphics compiled via HyperFrames (`index.html`, `cues.json`, overlay keyframes)
  7. `composite` — Overlay compositing, RNNoise/compressor/loudnorm mastering, and EBU R128 loudness check
  8. `deliverables` — Frame-extracted cover thumbnails (Portrait Vision + Landscape Pillow) and 5 social copy options
- **Embedded Single-Window Co-Pilot:** Split-workspace WebUI drawer running `agy --output-format stream-json --dangerously-skip-permissions` with multi-turn session continuity (`--conversation`), live tool/token progress streaming, workspace file-change detection, and one-click UI/stage actions.
- **Per-Job & Per-Message Model Routing:** Quota-safe defaults (`gemini-3.8-flash-high`) with live `PATCH /api/jobs/{job_id}/models` hot-switching across `agy` model tiers.

## Architecture

- [`server.py`](server.py) — FastAPI HTTP + SSE server and static WebUI host
- [`orchestrator.py`](orchestrator.py) — State machine (`JobManager`), `StudioManager` preview bridge, and `copilot_history.json` persistence
- [`adapters.py`](adapters.py) — Async subprocess runner (`_run_proc`), `agy` JSON & stream-json adapters, media/HyperFrames tooling
- [`agent_prompts.py`](agent_prompts.py) — Structured JSON schemas, prompt builders, and EDL overlap repair
- [`presets.py`](presets.py) — Vertical (`1080x1920`), landscape (`1920x1080`), retention presets, and `AGY_MODELS` catalog
- [`web/`](web/) — Vanilla HTML/CSS/ES-module studio frontend (zero build step)
- [`tests/test_pipeline.py`](tests/test_pipeline.py) — Pytest unit and API integration suite

## Prerequisites

- Python 3.11+ & [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg` / `ffprobe`
- Node.js & `@heygen/hyperframes` (`hyperframes`)
- Antigravity CLI (`agy`)

## Running Locally

```bash
./run.sh
```

Then open `http://127.0.0.1:8400` in your browser.

## Running Tests

```bash
uv run --with pytest --with pytest-asyncio --with httpx --with fastapi --with uvicorn \
  --with python-multipart --with pillow --with numpy \
  pytest -v tests/test_pipeline.py
```
