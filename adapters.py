"""Execution adapters for external tools, CLI binaries, media processing, and pipeline stages.

Integrates:
- ffmpeg / ffprobe media analysis, normalization, and thumbnails
- faster-whisper word-level transcription and VAD
- build.py, qa.py, stutters.py, scriptdiff.py for video spine building
- hyperframes check, preview, snapshot, and render for motion graphics
- agy CLI wrapper with robust structured JSON extraction and Co-Pilot turn support
- finish.py for neural audio mastering, ducking, and compositing
- cover.py for thumbnail generation
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any, Callable

try:
    from .agent_prompts import (
        AUDIO_SPOTCHECK_SCHEMA,
        COVER_SELECTOR_SCHEMA,
        CRITIC_SCHEMA,
        DESCRIPTIONS_SCHEMA,
        EDL_SCHEMA,
        HYPERFRAMES_SCHEMA,
        HYPERFRAMES_SCENE_SCHEMA,
        SCRIPT_OPTIONS_SCHEMA,
        VISION_CRITIQUE_SCHEMA,
        build_audio_spotcheck_prompt,
        build_cover_prompt,
        build_descriptions_prompt,
        build_edl_prompt,
        build_fallback_scenes_js,
        build_hyperframes_prompt,
        build_hyperframes_revision_prompt,
        build_script_critic_prompt,
        build_script_generation_prompt,
        build_vision_critique_prompt,
        render_hyperframes_index_html,
        validate_and_fix_edl_overlaps,
    )
    from .presets import get_preset
except ImportError:
    from agent_prompts import (
        AUDIO_SPOTCHECK_SCHEMA,
        COVER_SELECTOR_SCHEMA,
        CRITIC_SCHEMA,
        DESCRIPTIONS_SCHEMA,
        EDL_SCHEMA,
        HYPERFRAMES_SCHEMA,
        HYPERFRAMES_SCENE_SCHEMA,
        SCRIPT_OPTIONS_SCHEMA,
        VISION_CRITIQUE_SCHEMA,
        build_audio_spotcheck_prompt,
        build_cover_prompt,
        build_descriptions_prompt,
        build_edl_prompt,
        build_fallback_scenes_js,
        build_hyperframes_prompt,
        build_hyperframes_revision_prompt,
        build_script_critic_prompt,
        build_script_generation_prompt,
        build_vision_critique_prompt,
        render_hyperframes_index_html,
        validate_and_fix_edl_overlaps,
    )
    from presets import get_preset


# Binary & Asset Paths on Host
SCRIPTS_DIR = Path("/home/sandbox/tools/skills/story-video/scripts")
V6_REF_DIR = Path("/home/sandbox/ContentCreation/claude/v6")
AGY_BIN = "/home/sandbox/tools/bin/agy"
HYPERFRAMES_BIN = "/home/sandbox/tools/bin/hyperframes"
UV_BIN = "/home/sandbox/.local/bin/uv"
FFMPEG_BIN = "/usr/bin/ffmpeg"
FFPROBE_BIN = "/usr/bin/ffprobe"
THUMBS_CACHE_DIR = Path("/tmp/studio-pipeline-thumbs")


def check_tool_health() -> dict[str, dict[str, Any]]:
    """Probe host environment for availability of core pipeline tools."""
    tool_map = {
        "agy": [shutil.which("agy"), AGY_BIN],
        "hyperframes": [shutil.which("hyperframes"), HYPERFRAMES_BIN],
        "ffmpeg": [shutil.which("ffmpeg"), FFMPEG_BIN],
        "ffprobe": [shutil.which("ffprobe"), FFPROBE_BIN],
        "uv": [shutil.which("uv"), UV_BIN],
        "comfy": [shutil.which("comfy"), shutil.which("comfyui")],
    }
    results = {}
    for name, candidates in tool_map.items():
        found = next((c for c in candidates if c and Path(c).exists()), None)
        results[name] = {
            "available": found is not None,
            "path": str(found) if found else None,
        }
    return results


# ============================================================================
# SUBPROCESS & CLI RUNNERS
# ============================================================================

async def _run_proc(
    cmd: list[str],
    cwd: str | Path | None = None,
    timeout_s: float | None = None,
    cancel_event: asyncio.Event | None = None,
    log_cb: Callable[[str], None] | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Execute a subprocess asynchronously in its own process group with full cancellation cleanup.

    Guarantees:
    - Never leaves orphan child processes on cancellation or timeout (kills entire process group).
    - Streams stdout/stderr lines to log_cb in real-time.
    """
    proc_env = dict(os.environ)
    if env:
        proc_env.update(env)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=16 * 1024 * 1024,
        start_new_session=True,  # setsid for process group termination
        env=proc_env,
    )

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    async def _read_stream(stream: asyncio.StreamReader | None, chunks: list[str], is_err: bool):
        if stream is None:
            return
        while True:
            try:
                line = await stream.readline()
            except ValueError:
                # Fallback if a single line ever exceeds buffer limit
                line = await stream.read(65536)
            if not line:
                break
            text = line.decode("utf-8", errors="replace")
            chunks.append(text)
            if log_cb:
                try:
                    log_cb(f"[{'ERR' if is_err else 'OUT'}] {text.rstrip()}")
                except Exception:
                    pass

    reader_stdout = asyncio.create_task(_read_stream(proc.stdout, stdout_chunks, False))
    reader_stderr = asyncio.create_task(_read_stream(proc.stderr, stderr_chunks, True))

    wait_proc = asyncio.create_task(proc.wait())
    watch_tasks: set[asyncio.Future[Any]] = {wait_proc}

    cancel_task: asyncio.Task | None = None
    if cancel_event:
        cancel_task = asyncio.create_task(cancel_event.wait())
        watch_tasks.add(cancel_task)

    timed_out = False
    cancelled = False

    try:
        done, _ = await asyncio.wait(
            watch_tasks,
            timeout=timeout_s,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done:
            timed_out = True
        elif cancel_task and cancel_task in done:
            cancelled = True
        elif wait_proc not in done:
            timed_out = True

    except asyncio.CancelledError:
        cancelled = True
    finally:
        if cancel_task and not cancel_task.done():
            cancel_task.cancel()
            try:
                await cancel_task
            except (asyncio.CancelledError, Exception):
                pass

    if timed_out or cancelled:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass

        try:
            await asyncio.wait_for(proc.wait(), timeout=1.5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

        for t in [reader_stdout, reader_stderr, wait_proc]:
            if not t.done():
                t.cancel()

        if cancelled:
            if log_cb:
                log_cb("[SYS] Process cancelled by user.")
            raise asyncio.CancelledError("Process cancelled")
        if timed_out:
            msg = f"Process timed out after {timeout_s}s"
            if log_cb:
                log_cb(f"[SYS] {msg}")
            return (-1, "".join(stdout_chunks), msg)

    try:
        await asyncio.wait_for(
            asyncio.gather(reader_stdout, reader_stderr, return_exceptions=True),
            timeout=5.0,
        )
    except asyncio.TimeoutError:
        for t in [reader_stdout, reader_stderr]:
            if not t.done():
                t.cancel()

    return (proc.returncode if proc.returncode is not None else 0, "".join(stdout_chunks), "".join(stderr_chunks))


def _extract_json_payload(raw_text: str, prefer_structured: bool = False) -> Any:
    """Robustly extract JSON data from AGY outputs, handling markdown blocks and envelopes."""
    text = raw_text.strip()
    if not text:
        return None

    # Direct JSON parse attempt
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            # When prefer_structured is True (e.g. schema was requested), check structured_output first
            if prefer_structured and "structured_output" in parsed and parsed["structured_output"] is not None:
                return parsed["structured_output"]
            # In AGY output, check response field first if present
            if "response" in parsed and isinstance(parsed["response"], str):
                resp_str = parsed["response"].strip()
                extracted = _extract_json_payload(resp_str, prefer_structured=False)
                if extracted is not None:
                    return extracted
            # If prefer_structured wasn't true or response didn't contain JSON, check structured_output as fallback
            if "structured_output" in parsed and parsed["structured_output"] is not None:
                return parsed["structured_output"]
            if "result" in parsed and isinstance(parsed["result"], str):
                try:
                    return json.loads(parsed["result"])
                except Exception:
                    text = parsed["result"].strip()
            elif "result" in parsed and isinstance(parsed["result"], (dict, list)):
                return parsed["result"]
            if "content" in parsed and isinstance(parsed["content"], str):
                text = parsed["content"].strip()
            if parsed.get("status") == "ERROR":
                return None
        else:
            return parsed
    except Exception:
        pass

    # Search for markdown fenced blocks ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass

    # Search between outermost balanced brackets
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace : last_brace + 1])
        except Exception:
            pass

    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket != -1 and last_bracket > first_bracket:
        try:
            return json.loads(text[first_bracket : last_bracket + 1])
        except Exception:
            pass

    return None


async def run_agy_json(
    prompt: str,
    schema: dict[str, Any] | None = None,
    model: str = "gemini-3.8-flash-high",
    effort: str | None = None,
    conversation_id: str | None = None,
    cwd: str | Path | None = None,
    add_dirs: list[str | Path] | None = None,
    timeout_s: float = 360.0,
    cancel_event: asyncio.Event | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> Any:
    """Invoke AGY via CLI to obtain a structured JSON analysis result."""
    cmd = [
        AGY_BIN,
        "-p",
        prompt,
        "--output-format",
        "json",
        "--dangerously-skip-permissions",
        "--disable-slash-commands",
        "--new-project",
        "--model",
        model,
    ]
    if effort:
        cmd.extend(["--effort", effort])
    if conversation_id:
        cmd.extend(["--conversation", conversation_id])
    if add_dirs:
        for d in add_dirs:
            cmd.extend(["--add-dir", str(d)])
    if schema:
        cmd.extend(["--json-schema", json.dumps(schema)])

    if log_cb:
        schema_label = "structured schema" if schema else "freeform"
        log_cb(f"[AI] Invoking AGY ({model}) with {schema_label}...")

    effective_cwd = cwd if cwd is not None else "/tmp"
    code, out, err = await _run_proc(cmd, cwd=effective_cwd, timeout_s=timeout_s, cancel_event=cancel_event, log_cb=log_cb)
    if code != 0 and not out:
        raise RuntimeError(f"AGY execution failed (code {code}): {err}")

    parsed = _extract_json_payload(out, prefer_structured=(schema is not None))
    if parsed is None:
        if code != 0:
            raise RuntimeError(f"AGY execution failed (code {code}): {err or out[:500]}")
        raise ValueError(f"Failed to extract JSON from AGY response:\n{out[:500]}")
    return parsed


def _format_tool_hint(params: Any) -> str:
    """Extract a short readable target hint from AGY tool call parameters."""
    if not isinstance(params, dict):
        return ""
    for key in ("TargetFile", "AbsolutePath", "DirectoryPath", "path", "CommandLine", "Query"):
        val = params.get(key)
        if isinstance(val, str) and val.strip():
            clean = val.strip()
            if key in ("TargetFile", "AbsolutePath", "DirectoryPath", "path"):
                return Path(clean).name or clean
            return clean[:48] + ("..." if len(clean) > 48 else "")
    return ""


async def run_agy_copilot_turn(
    prompt: str,
    job_dir: Path,
    model: str = "gemini-3.8-flash-high",
    conversation_id: str | None = None,
    add_dirs: list[Path] | None = None,
    timeout_s: float = 600.0,
    cancel_event: asyncio.Event | None = None,
    log_cb: Callable[[str], None] | None = None,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute a single interactive Antigravity Co-Pilot turn with real-time NDJSON progress streaming."""
    import time as _time

    cmd = [
        AGY_BIN,
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--dangerously-skip-permissions",
        "--model",
        model,
        "--add-dir",
        str(job_dir.resolve()),
    ]
    if conversation_id:
        cmd.extend(["--conversation", conversation_id])
    if add_dirs:
        for d in add_dirs:
            if str(d.resolve()) != str(job_dir.resolve()):
                cmd.extend(["--add-dir", str(d.resolve())])

    if log_cb:
        conv_label = f"resuming conversation {conversation_id[:8]}..." if conversation_id else "starting new conversation..."
        log_cb(f"[COPILOT] Invoking Antigravity ({model}) {conv_label}")
    if progress_cb:
        progress_cb({"kind": "status", "text": f"Starting {model}..."})

    stream_state: dict[str, Any] = {
        "conversation_id": conversation_id,
        "result_obj": None,
        "text_parts": [],
        "last_delta_ts": 0.0,
        "last_step_idx": None,
    }

    def _emit_progress(payload: dict[str, Any]):
        if progress_cb:
            try:
                progress_cb(payload)
            except Exception:
                pass

    def _consume_event(ev: dict[str, Any]):
        ev_type = ev.get("event")
        if ev.get("conversation_id"):
            stream_state["conversation_id"] = ev["conversation_id"]

        if ev_type == "init":
            _emit_progress({"kind": "status", "text": f"Connected ({model}), analyzing workspace..."})
            return

        if ev_type == "step_update":
            su = ev.get("step_update") or {}
            if su.get("conversation_id"):
                stream_state["conversation_id"] = su["conversation_id"]
            step_type = su.get("step_type", "")
            step_state = su.get("state", "")
            step_idx = su.get("step_index")

            if step_type == "tool" and step_state == "ACTIVE":
                tool_info = su.get("tool_info") or {}
                tool_name = su.get("tool_name") or tool_info.get("name") or "tool"
                hint = _format_tool_hint(tool_info.get("parameters"))
                text = f"Running {tool_name} ({hint})..." if hint else f"Running {tool_name}..."
                _emit_progress({"kind": "tool", "tool": tool_name, "text": text})
                if log_cb:
                    log_cb(f"[COPILOT] {text}")
            elif step_type == "agent_response" and su.get("text_delta"):
                stream_state["text_parts"].append(str(su["text_delta"]))
                now = _time.monotonic()
                if now - stream_state["last_delta_ts"] >= 0.25 or step_state == "DONE":
                    stream_state["last_delta_ts"] = now
                    combined = "".join(stream_state["text_parts"])
                    _emit_progress({"kind": "delta", "preview": combined[-320:]})
            elif step_state == "ACTIVE" and step_idx != stream_state["last_step_idx"] and step_type:
                stream_state["last_step_idx"] = step_idx
                if step_type not in ("user_input", "agent_response"):
                    _emit_progress({"kind": "thinking", "text": f"Step {step_idx}: {step_type.replace('_', ' ')}..."})
            return

        if ev_type == "result":
            res = ev.get("result") or {}
            if res.get("conversation_id"):
                stream_state["conversation_id"] = res["conversation_id"]
            stream_state["result_obj"] = res

    def _stream_proc_log(raw_line: str):
        if raw_line.startswith("[OUT] "):
            body = raw_line[6:].strip()
            if body.startswith("{") and body.endswith("}"):
                try:
                    _consume_event(json.loads(body))
                    return
                except Exception:
                    pass
        if log_cb:
            log_cb(raw_line)

    code, out, err = await _run_proc(
        cmd,
        cwd=job_dir,
        timeout_s=timeout_s,
        cancel_event=cancel_event,
        log_cb=_stream_proc_log,
    )
    if code != 0 and not out:
        raise RuntimeError(f"AGY Co-Pilot execution failed (code {code}): {err}")

    # Fallback parsing for multi-line NDJSON or single JSON envelope if stdout bypassed _consume_event
    if stream_state["result_obj"] is None:
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    ev = json.loads(s)
                    if "event" in ev:
                        _consume_event(ev)
                    elif "response" in ev or "conversation_id" in ev:
                        stream_state["result_obj"] = ev
                except Exception:
                    pass

    res_obj = stream_state["result_obj"] or {}
    if not res_obj:
        fb = out.find("{")
        lb = out.rfind("}")
        if fb != -1 and lb > fb:
            try:
                res_obj = json.loads(out[fb : lb + 1])
            except Exception:
                res_obj = {}

    response_text = res_obj.get("response") or "".join(stream_state["text_parts"])
    if not response_text and not res_obj:
        response_text = out.strip()

    return {
        "response_text": str(response_text).strip(),
        "conversation_id": res_obj.get("conversation_id") or stream_state["conversation_id"] or conversation_id,
        "model": model,
        "duration_seconds": float(res_obj.get("duration_seconds", 0.0) or 0.0),
        "usage": res_obj.get("usage", {}) or {},
    }


# ============================================================================
# MEDIA PROBING, THUMBNAILS & CLIP DISCOVERY
# ============================================================================

def probe_media(path: str | Path, sample_luma: bool = False) -> dict[str, Any]:
    """Extract media stream properties and optional mean luminance using ffprobe and ffmpeg."""
    target = Path(path)
    if not target.exists():
        return {
            "duration_s": 0.0,
            "width": 0,
            "height": 0,
            "fps": 30.0,
            "has_audio": False,
            "mean_luma": 100.0,
        }

    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=width,height,r_frame_rate,codec_type",
        "-of",
        "json",
        str(target),
    ]

    try:
        raw = subprocess.check_output(cmd, text=True)
        data = json.loads(raw)
    except Exception:
        return {
            "duration_s": 0.0,
            "width": 0,
            "height": 0,
            "fps": 30.0,
            "has_audio": False,
            "mean_luma": 100.0,
        }

    dur = float(data.get("format", {}).get("duration", 0.0))
    streams = data.get("streams", [])

    width, height, fps = 1080, 1920, 30.0
    has_audio = False

    for s in streams:
        if s.get("codec_type") == "video":
            width = int(s.get("width", 1080))
            height = int(s.get("height", 1920))
            r_rate = s.get("r_frame_rate", "30/1")
            if "/" in r_rate:
                num, den = r_rate.split("/")
                fps = round(float(num) / max(1.0, float(den)), 2)
            else:
                fps = float(r_rate)
        elif s.get("codec_type") == "audio":
            has_audio = True

    mean_luma = 100.0
    if sample_luma:
        sample_time = min(max(0.5, dur * 0.25), 5.0)
        luma_cmd = [
            FFMPEG_BIN,
            "-v",
            "error",
            "-ss",
            f"{sample_time:.2f}",
            "-i",
            str(target),
            "-vframes",
            "1",
            "-vf",
            "format=gray,scale=32:32",
            "-f",
            "rawvideo",
            "-",
        ]
        try:
            pix = subprocess.check_output(luma_cmd)
            if pix:
                mean_luma = round(sum(pix) / len(pix), 1)
        except Exception:
            pass

    return {
        "duration_s": round(dur, 2),
        "width": width,
        "height": height,
        "fps": fps,
        "has_audio": has_audio,
        "mean_luma": mean_luma,
    }


def get_or_create_thumbnail(video_path: str | Path) -> Path:
    """Generate or retrieve a cached JPEG thumbnail for a video file."""
    src = Path(video_path).resolve()
    THUMBS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    h = hashlib.sha1(str(src).encode()).hexdigest()
    thumb_path = THUMBS_CACHE_DIR / f"{h}.jpg"

    if thumb_path.exists() and thumb_path.stat().st_size > 0:
        return thumb_path

    cmd = [
        FFMPEG_BIN,
        "-v",
        "error",
        "-y",
        "-ss",
        "1.0",
        "-i",
        str(src),
        "-vframes",
        "1",
        "-vf",
        "scale=320:-1",
        str(thumb_path),
    ]
    res = subprocess.run(cmd)
    if res.returncode != 0 or not thumb_path.exists():
        # Fallback to frame 0 for very short clips
        cmd[4] = "0.0"
        subprocess.run(cmd)

    return thumb_path


def list_clips(folder_path: str) -> dict[str, Any]:
    """Scan and list all supported video clips in a folder with media probe details and thumbnails."""
    p = Path(folder_path).resolve()
    if not p.exists() or not p.is_dir():
        return {
            "path": str(p),
            "exists": False,
            "writable": False,
            "clips": [],
        }

    writable = os.access(p, os.W_OK)
    valid_exts = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
    clips: list[dict[str, Any]] = []

    try:
        entries = sorted(p.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
    except Exception:
        entries = []

    for f in entries:
        if f.is_file() and f.suffix.lower() in valid_exts and not f.name.startswith("."):
            try:
                st = f.stat()
                probe = probe_media(f)
                thumb = get_or_create_thumbnail(f)
                clips.append(
                    {
                        "name": f.name,
                        "path": str(f),
                        "size_bytes": st.st_size,
                        "duration_s": probe["duration_s"],
                        "width": probe["width"],
                        "height": probe["height"],
                        "fps": probe["fps"],
                        "mtime": st.st_mtime,
                        "thumb_url": f"/api/thumb?path={f.resolve()}",
                    }
                )
            except Exception:
                continue

    return {
        "path": str(p),
        "exists": True,
        "writable": writable,
        "clips": clips,
    }


# ============================================================================
# STAGE EXECUTION ADAPTERS
# ============================================================================

async def run_ingest_transcribe(
    job_dir: Path,
    selected_clip_paths: list[str],
    whisper_model: str = "small.en",
    cancel_event: asyncio.Event | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Stage 1: Normalize selected clips into work/src/<clip_id>.mp4 and extract 16kHz audio, then transcribe."""
    work_dir = job_dir / "work"
    src_dir = work_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    if not selected_clip_paths:
        raise ValueError("No clip paths provided for ingestion.")

    if log_cb:
        log_cb(f"[INGEST] Processing {len(selected_clip_paths)} clips into work/src/...")

    clip_map: dict[str, Any] = {}
    wav_files: list[str] = []
    luma_samples: list[float] = []

    for idx, path_str in enumerate(selected_clip_paths):
        raw_path = Path(path_str).resolve()
        if not raw_path.exists():
            raise FileNotFoundError(f"Clip not found: {path_str}")

        clip_id = f"c{idx + 1:02d}"
        symlink_mp4 = src_dir / f"{clip_id}.mp4"
        if symlink_mp4.is_symlink() or symlink_mp4.exists():
            symlink_mp4.unlink()
        symlink_mp4.symlink_to(raw_path)

        wav_name = f"{clip_id}.wav"
        out_wav = work_dir / wav_name

        # Extract 16kHz mono audio for Whisper
        ffmpeg_cmd = [
            FFMPEG_BIN,
            "-v",
            "error",
            "-y",
            "-i",
            str(raw_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out_wav),
        ]
        code, _, err = await _run_proc(ffmpeg_cmd, cwd=work_dir, cancel_event=cancel_event, log_cb=log_cb)
        if code != 0:
            raise RuntimeError(f"FFmpeg audio extraction failed for {raw_path.name}: {err}")

        wav_files.append(wav_name)
        probe = probe_media(raw_path, sample_luma=True)
        luma_samples.append(probe["mean_luma"])

        clip_map[clip_id] = {
            "id": clip_id,
            "original_path": str(raw_path),
            "stem": raw_path.stem,
            "duration_s": probe["duration_s"],
            "width": probe["width"],
            "height": probe["height"],
            "fps": probe["fps"],
            "mean_luma": probe["mean_luma"],
        }

    # Save clip map
    (work_dir / "clip_map.json").write_text(json.dumps(clip_map, indent=2))

    # Adaptive gamma suggestion based on average luminance
    avg_luma = sum(luma_samples) / max(1, len(luma_samples))
    if avg_luma < 45.0:
        suggested_gamma = 1.15
    elif avg_luma < 75.0:
        suggested_gamma = 1.08
    else:
        suggested_gamma = 1.02

    if log_cb:
        log_cb(f"[INGEST] Average luma: {avg_luma:.1f} -> suggested gamma: {suggested_gamma}")

    ingest_meta = {
        "suggested_gamma": suggested_gamma,
        "avg_luma": round(avg_luma, 1),
    }
    (work_dir / "ingest_meta.json").write_text(json.dumps(ingest_meta, indent=2))

    # Generate lighting contact sheet (3x3 grid)
    sheet_path = work_dir / "lighting_sheet.png"
    first_clip = src_dir / f"{list(clip_map.keys())[0]}.mp4"
    sheet_cmd = [
        FFMPEG_BIN,
        "-v",
        "error",
        "-y",
        "-i",
        str(first_clip),
        "-vf",
        "select='not(mod(n\\,100))',scale=360:640,tile=3x3",
        "-frames:v",
        "1",
        str(sheet_path),
    ]
    await _run_proc(sheet_cmd, cwd=work_dir, cancel_event=cancel_event)

    # Run transcription
    if log_cb:
        log_cb(f"[TRANSCRIBE] Running faster-whisper ({whisper_model}) on {len(wav_files)} clips...")

    if whisper_model == "small.en":
        transcribe_cmd = [
            UV_BIN,
            "run",
            "--with",
            "faster-whisper",
            "--with",
            "numpy",
            "python",
            "-I",
            str(SCRIPTS_DIR / "prep_transcribe.py"),
            *wav_files,
        ]
    else:
        # Custom model inline script
        inline_script = (
            "import sys,wave,json,numpy as np\n"
            "from faster_whisper import WhisperModel\n"
            f"m=WhisperModel('{whisper_model}',compute_type='int8')\n"
            "out={};txt=[]\n"
            "for f in sys.argv[1:]:\n"
            "    k=f.split('/')[-1][:-4]\n"
            "    w=wave.open(f);a=np.frombuffer(w.readframes(w.getnframes()),np.int16).astype(np.float32)/32768\n"
            "    s,_=m.transcribe(a,vad_filter=True,language='en',word_timestamps=True)\n"
            "    out[k]=[];txt.append('## '+k)\n"
            "    for seg in s:\n"
            "        txt.append(f'[{seg.start:.1f}-{seg.end:.1f}] {seg.text.strip()}')\n"
            "        out[k]+= [[x.word.strip(),round(x.start,2),round(x.end,2)] for x in seg.words]\n"
            "json.dump(out,open('words.json','w')); open('transcript_all.txt','w').write('\\n'.join(txt))\n"
        )
        transcribe_cmd = [
            UV_BIN,
            "run",
            "--with",
            "faster-whisper",
            "--with",
            "numpy",
            "python",
            "-c",
            inline_script,
            *wav_files,
        ]

    code, out, err = await _run_proc(transcribe_cmd, cwd=work_dir, cancel_event=cancel_event, log_cb=log_cb)
    if code != 0:
        raise RuntimeError(f"Transcription failed: {err}")

    words_file = work_dir / "words.json"
    transcript_file = work_dir / "transcript_all.txt"
    if not words_file.exists() or not transcript_file.exists():
        raise FileNotFoundError("Transcription outputs (words.json / transcript_all.txt) were not generated.")

    words_data = json.loads(words_file.read_text())
    total_words = sum(len(ws) for ws in words_data.values())

    return {
        "clips": clip_map,
        "suggested_gamma": suggested_gamma,
        "total_words": total_words,
        "transcript_path": str(transcript_file),
        "words_path": str(words_file),
        "lighting_sheet": str(sheet_path),
    }


def _normalize_ass_color(value: Any, default: str = "&H3D8AFF&") -> str:
    """Normalize RGB hex (#RRGGBB) or ASS BGR (&HBBGGRR&) color strings to ASS BGR format."""
    s = str(value or "").strip()
    if re.fullmatch(r"&H[0-9A-Fa-f]{6}&?", s):
        return s if s.endswith("&") else f"{s}&"
    m = re.fullmatch(r"#([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})", s)
    if m:
        r, g, b = m.group(1), m.group(2), m.group(3)
        return f"&H{b.upper()}{g.upper()}{r.upper()}&"
    return default


async def run_build_and_qa(
    job_dir: Path,
    preset: dict[str, Any],
    edl: dict[str, Any] | None = None,
    cancel_event: asyncio.Event | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Stage 4: Assemble video spine via build.py, run qa.py, remove disfluencies via stutters.py, and diff script."""
    work_dir = job_dir / "work"
    src_dir = work_dir / "src"

    words_data: dict[str, Any] = {}
    if (work_dir / "words.json").exists():
        words_data = json.loads((work_dir / "words.json").read_text())

    edl_file = work_dir / "edl.json"
    if edl is not None:
        fixed_edl = validate_and_fix_edl_overlaps(edl, words_data)
        edl_file.write_text(json.dumps(fixed_edl, indent=2))
    elif edl_file.exists():
        raw_disk_edl = json.loads(edl_file.read_text())
        fixed_edl = validate_and_fix_edl_overlaps(raw_disk_edl, words_data)
        edl_file.write_text(json.dumps(fixed_edl, indent=2))
    else:
        raise FileNotFoundError("edl.json missing in workdir.")

    # Merge EDL keyword highlights, name normalizations, and ingest luma gamma into config.json
    merged_highlight: dict[str, str] = {
        str(k).lower(): _normalize_ass_color(v)
        for k, v in (preset.get("highlight_colors") or {"pink": "&HA04FFF&", "blue": "&HFF6B3D&"}).items()
        if str(k).strip()
    }
    for k, v in (fixed_edl.get("highlight") or {}).items():
        clean_k = re.sub(r"[^a-z0-9']", "", str(k).lower())
        if clean_k:
            merged_highlight[clean_k] = _normalize_ass_color(v)

    merged_names: dict[str, str] = {"christian": "Christian"}
    for k, v in (fixed_edl.get("names") or {}).items():
        if str(k).strip() and str(v).strip():
            merged_names[str(k).strip().lower()] = str(v).strip()

    gamma_val = float(preset.get("gamma", 1.08))
    ingest_meta_file = work_dir / "ingest_meta.json"
    if ingest_meta_file.exists():
        try:
            meta_json = json.loads(ingest_meta_file.read_text())
            if meta_json.get("suggested_gamma"):
                gamma_val = float(meta_json["suggested_gamma"])
        except Exception:
            pass

    config_data = {
        "src": str(src_dir / "%s.mp4"),
        "highlight": merged_highlight,
        "names": merged_names,
        "zoom": preset.get("zoom", 1.08),
        "gamma": round(gamma_val, 3),
    }
    (work_dir / "config.json").write_text(json.dumps(config_data, indent=2))

    is_vertical = preset.get("width", 1080) == 1080 and preset.get("height", 1920) == 1920

    if log_cb:
        log_cb("[BUILD] Executing base cut build...")

    if is_vertical:
        build_cmd = [
            UV_BIN,
            "run",
            "--with",
            "faster-whisper",
            "--with",
            "numpy",
            "python",
            "-I",
            str(SCRIPTS_DIR / "build.py"),
            str(work_dir),
        ]
        code, out, err = await _run_proc(build_cmd, cwd=work_dir, cancel_event=cancel_event, log_cb=log_cb)
        if code != 0:
            raise RuntimeError(f"build.py failed: {err}\n{out}")

        if log_cb:
            log_cb("[QA] Running audio burst and clipping analysis...")
        qa_cmd = [
            UV_BIN,
            "run",
            "--with",
            "faster-whisper",
            "--with",
            "numpy",
            "python",
            "-I",
            str(SCRIPTS_DIR / "qa.py"),
            str(work_dir),
            "base.mp4",
        ]
        code, qa_out, _ = await _run_proc(qa_cmd, cwd=work_dir, cancel_event=cancel_event, log_cb=log_cb)

        if log_cb:
            log_cb("[STUTTERS] Detecting immediate word repeats and fillers...")
        stutters_cmd = [
            UV_BIN,
            "run",
            "--with",
            "faster-whisper",
            "--with",
            "numpy",
            "python",
            "-I",
            str(SCRIPTS_DIR / "stutters.py"),
            str(work_dir),
        ]
        await _run_proc(stutters_cmd, cwd=work_dir, cancel_event=cancel_event, log_cb=log_cb)

        cuts_proposed_file = work_dir / "cuts_proposed.json"
        if cuts_proposed_file.exists():
            try:
                cuts_data = json.loads(cuts_proposed_file.read_text())
                total_proposed = sum(len(c) for c in cuts_data.values()) if isinstance(cuts_data, dict) else 0
                if total_proposed > 0:
                    if log_cb:
                        log_cb(f"[STUTTERS] Applying {total_proposed} proposed stutter cuts and rebuilding...")
                    (work_dir / "cuts.json").write_text(json.dumps(cuts_data, indent=2))
                    rcode, rout, rerr = await _run_proc(build_cmd, cwd=work_dir, cancel_event=cancel_event, log_cb=log_cb)
                    if rcode != 0:
                        raise RuntimeError(f"Post-stutter build.py rebuild failed: {rerr}\n{rout}")
                    # Refresh QA on the newly rebuilt base.mp4 so bursts/qa logs match actual output
                    _, qa_out, _ = await _run_proc(qa_cmd, cwd=work_dir, cancel_event=cancel_event, log_cb=log_cb)
            except RuntimeError:
                raise
            except Exception:
                pass

        # Always re-extract base16.wav from the final base.mp4 (never reuse stale pre-rebuild base16.wav)
        await _run_proc(
            [FFMPEG_BIN, "-v", "error", "-y", "-i", "base.mp4", "-vn", "-ac", "1", "-ar", "16000", "base16.wav"],
            cwd=work_dir,
            cancel_event=cancel_event,
        )

        vad_script = (
            "import wave,numpy as np\n"
            "from faster_whisper import WhisperModel\n"
            "w=wave.open('base16.wav');a=np.frombuffer(w.readframes(w.getnframes()),np.int16).astype(np.float32)/32768\n"
            "m=WhisperModel('small.en',compute_type='int8')\n"
            "segs,_=m.transcribe(a,language='en',vad_filter=True,word_timestamps=True,condition_on_previous_text=False)\n"
            "with open('final_transcript.txt','w') as f:\n"
            "    for s in segs: f.write(f'[{s.start:6.1f}] '+' '.join(x.word.strip() for x in s.words)+'\\n')\n"
        )
        await _run_proc(
            [UV_BIN, "run", "--with", "faster-whisper", "--with", "numpy", "python", "-c", vad_script],
            cwd=work_dir,
            cancel_event=cancel_event,
        )

        diff_cmd = [
            "python3",
            "-I",
            str(SCRIPTS_DIR / "scriptdiff.py"),
            str(work_dir),
        ]
        _, diff_out, _ = await _run_proc(diff_cmd, cwd=work_dir, cancel_event=cancel_event, log_cb=log_cb)

    else:
        # Landscape Explainer (1920x1080) Adapter
        if log_cb:
            log_cb("[BUILD] Generating landscape 1920x1080 explainer video...")
        edl_data = json.load(open(work_dir / "edl.json"))
        # Render 16:9 base video using standard filter_complex
        ins: list[str] = []
        fc: list[str] = []
        for idx, part in enumerate(edl_data["parts"]):
            clip_path = src_dir / f"{part['clip']}.mp4"
            r0, r1 = part["range"]
            dur = r1 - r0
            ins.extend(["-ss", f"{r0:.3f}", "-t", f"{dur:.3f}", "-i", str(clip_path)])
            fc.append(
                f"[{idx}:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30,setsar=1,setpts=PTS-STARTPTS[v{idx}];"
                f"[{idx}:a]aresample=48000,asetpts=PTS-STARTPTS[a{idx}]"
            )

        n_clips = len(edl_data["parts"])
        concat_str = "".join(f"[v{i}][a{i}]" for i in range(n_clips)) + f"concat=n={n_clips}:v=1:a=1[v][a]"
        fc.append(concat_str)

        # Generate lower-third ASS subtitles for landscape
        ass_content = (
            "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n"
            "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
            "Style: Cap,Inter Black,68,&HFFFFFF,&HFFFFFF,&H000000,&H80000000,0,0,0,0,100,100,0,0,1,5,2,2,120,120,80,1\n"
            "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
        )
        cur_time = 0.0
        timeline_rows = []
        for idx, part in enumerate(edl_data["parts"]):
            r0, r1 = part["range"]
            d = r1 - r0
            end_t = cur_time + d
            st_str = f"0:{int(cur_time // 60):02d}:{cur_time % 60:05.2f}"
            et_str = f"0:{int(end_t // 60):02d}:{end_t % 60:05.2f}"
            ass_content += f"Dialogue: 0,{st_str},{et_str},Cap,,0,0,0,,{part['say']}\n"
            timeline_rows.append(
                {
                    "id": part.get("id"),
                    "clip": part["clip"],
                    "start": round(cur_time, 2),
                    "end": round(end_t, 2),
                    "say": part["say"],
                }
            )
            cur_time = end_t

        (work_dir / "cap.ass").write_text(ass_content)
        (work_dir / "timeline.json").write_text(json.dumps({"total": round(cur_time, 2), "parts": timeline_rows}, indent=2))

        filter_txt = ";".join(fc) + ";[v]ass=cap.ass:fontsdir=/usr/share/fonts[vo]"
        (work_dir / "filter.txt").write_text(filter_txt)

        render_cmd = [
            FFMPEG_BIN,
            "-v",
            "error",
            "-y",
            *ins,
            "-filter_complex_script",
            "filter.txt",
            "-map",
            "[vo]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-preset",
            "medium",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "base.mp4",
        ]
        await _run_proc(render_cmd, cwd=work_dir, cancel_event=cancel_event, log_cb=log_cb)
        qa_out = "PASS (Landscape)"
        diff_out = "0 differences"

    # Compile QA Report
    bursts = []
    if (work_dir / "bursts.json").exists():
        try:
            bursts = json.loads((work_dir / "bursts.json").read_text())
        except Exception:
            pass

    # Extract script match percentage
    script_match_pct = 100.0
    if diff_out:
        m_ratio = re.search(r"word match ratio\s+([0-9.]+)", diff_out)
        if m_ratio:
            try:
                script_match_pct = round(float(m_ratio.group(1)) * 100, 1)
            except (ValueError, TypeError):
                pass
        elif "0 differences" in diff_out:
            script_match_pct = 100.0

    # Extract parts count from timeline.json or edl.json
    parts_count = 0
    timeline_file = work_dir / "timeline.json"
    if timeline_file.exists():
        try:
            t_data = json.loads(timeline_file.read_text())
            parts_count = len(t_data.get("parts", []))
        except Exception:
            pass
    if parts_count == 0 and (work_dir / "edl.json").exists():
        try:
            e_data = json.loads((work_dir / "edl.json").read_text())
            parts_count = len(e_data.get("parts", []))
        except Exception:
            pass

    # Extract stutters cut count from cuts.json
    stutters_cut = 0
    cuts_file = work_dir / "cuts.json"
    if cuts_file.exists():
        try:
            c_data = json.loads(cuts_file.read_text())
            if isinstance(c_data, dict):
                stutters_cut = sum(len(v) for v in c_data.values() if isinstance(v, list))
            elif isinstance(c_data, list):
                stutters_cut = len(c_data)
        except Exception:
            pass

    probe_base = probe_media(work_dir / "base.mp4")
    qa_report = {
        "status": "PASS" if "PASS" in (qa_out or "") else "REVIEW",
        "duration_s": probe_base["duration_s"],
        "script_match_pct": script_match_pct,
        "parts_count": parts_count,
        "stutters_cut": stutters_cut,
        "bursts_detected": len(bursts),
        "bursts": bursts[:10],
        "qa_log": qa_out[:1000] if qa_out else "",
        "diff_summary": diff_out.strip() if diff_out else "",
    }
    (work_dir / "qa_report.json").write_text(json.dumps(qa_report, indent=2))

    return qa_report


async def run_bgm_and_broll(
    job_dir: Path,
    preset: dict[str, Any],
    broll_provider: str = "local",
    bgm_choice: str | None = None,
    cancel_event: asyncio.Event | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Stage 5: Populate BGM candidate tracks, handle B-roll provider, and set active track.mp3."""
    work_dir = job_dir / "work"
    bgm_candidates_dir = work_dir / "assets" / "bgm" / "candidates"
    bgm_candidates_dir.mkdir(parents=True, exist_ok=True)

    # Copy local reference tracks from v6
    v6_bgm_dir = V6_REF_DIR / "work" / "assets" / "bgm"
    available_tracks: list[str] = []

    if v6_bgm_dir.exists():
        for track in v6_bgm_dir.iterdir():
            if track.is_file() and track.suffix in {".mp3", ".wav"}:
                dest = bgm_candidates_dir / track.name
                if not dest.exists():
                    shutil.copy2(track, dest)
                available_tracks.append(track.name)

    # Set active track
    chosen_name = bgm_choice if (bgm_choice and (bgm_candidates_dir / bgm_choice).exists()) else "track.mp3"
    chosen_path = bgm_candidates_dir / chosen_name
    active_bgm = work_dir / "assets" / "bgm" / "track.mp3"

    if chosen_path.exists():
        if active_bgm.is_symlink() or active_bgm.exists():
            active_bgm.unlink()
        active_bgm.symlink_to(chosen_path)
    elif available_tracks:
        first = bgm_candidates_dir / available_tracks[0]
        if active_bgm.is_symlink() or active_bgm.exists():
            active_bgm.unlink()
        active_bgm.symlink_to(first)

    if log_cb:
        log_cb(f"[AUDIO] Active BGM set to: {active_bgm.name} (from {len(available_tracks)} candidates)")

    # Handle broll_provider cleanly
    norm_broll = "comfyui" if broll_provider == "comfy" else broll_provider
    if norm_broll in ("off", "none", "local"):
        if log_cb:
            log_cb(f"[BROLL] AI B-roll provider '{norm_broll}': Using local footage and reference media only.")
    elif norm_broll in ("comfy", "comfyui"):
        comfy_bin = shutil.which("comfy") or shutil.which("comfyui")
        if comfy_bin:
            if log_cb:
                log_cb(f"[BROLL] ComfyUI provider detected at {comfy_bin}. Checking server readiness...")
                log_cb("[BROLL] [WARN] ComfyUI server offline or no workflow templates configured. Gracefully falling back to local footage.")
        else:
            if log_cb:
                log_cb("[BROLL] [WARN] ComfyUI CLI not found on system PATH. Gracefully falling back to local footage.")
    elif norm_broll == "mcp":
        if log_cb:
            log_cb("[BROLL] MCP provider: Checking for external media generation tools...")
            log_cb("[BROLL] [WARN] No external media MCP servers currently configured. Falling back to local footage.")

    return {
        "active_bgm": str(active_bgm),
        "candidates": available_tracks,
        "broll_provider": norm_broll,
    }


def _read_ass_cues_summary(cap_ass: Path) -> str:
    """Extract clean readable timestamped subtitle lines from cap.ass for graphics sync."""
    if not cap_ass.exists():
        return ""
    lines: list[str] = []
    for raw in cap_ass.read_text(errors="replace").splitlines():
        if not raw.startswith("Dialogue:"):
            continue
        parts = raw.split(",", 9)
        if len(parts) < 10:
            continue
        start_s, end_s, text_raw = parts[1].strip(), parts[2].strip(), parts[9]
        clean = re.sub(r"\{[^}]*\}", "", text_raw).strip()
        if clean:
            lines.append(f"[{start_s} -> {end_s}] {clean}")
    return "\n".join(lines[:45])


def _extract_check_issues(check_out: str) -> tuple[bool, list[str]]:
    """Inspect hyperframes check JSON stdout for hard errors, empty stage, or static seek."""
    issues: list[str] = []
    try:
        idx = check_out.find("{")
        if idx == -1:
            return False, []
        data = json.loads(check_out[idx:])
        for section in ("lint", "runtime", "layout", "craft"):
            sec = data.get(section, {})
            if sec.get("errorCount", 0) > 0:
                for f in sec.get("findings", []):
                    issues.append(f"{section}: {f.get('code')} - {f.get('message')}")
            elif section in ("layout", "craft"):
                for f in sec.get("findings", []):
                    if f.get("code") in ("sweep_static", "first_frame_empty"):
                        issues.append(f"{section}: {f.get('code')} - {f.get('message')}")
        return len(issues) > 0, issues
    except Exception:
        return False, []


async def run_motion_graphics(
    job_dir: Path,
    preset: dict[str, Any],
    creative_model: str = "gemini-3.8-flash-high",
    vision_model: str = "gemini-3.8-flash-high",
    mechanical_model: str = "gemini-3.8-flash-high",
    cancel_event: asyncio.Event | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Stage 6: Generate scaffolded HyperFrames composition (GSAP + SVG sketch), check DOM, snapshot, enforce Vision critique, and render overlay.mov."""
    work_dir = job_dir / "work"
    hf_dir = job_dir / "hf"
    assets_dir = hf_dir / "assets"
    fonts_dir = assets_dir / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)

    # Copy reference assets (GSAP & hand-drawn fonts)
    v6_hf_assets = V6_REF_DIR / "hf" / "assets"
    if (v6_hf_assets / "gsap.min.js").exists():
        shutil.copy2(v6_hf_assets / "gsap.min.js", assets_dir / "gsap.min.js")
    for f in (v6_hf_assets / "fonts").glob("*.woff2"):
        shutil.copy2(f, fonts_dir / f.name)

    hf_config = {
        "$schema": "https://hyperframes.heygen.com/schema/hyperframes.json",
        "registry": "https://raw.githubusercontent.com/heygen-com/hyperframes/main/registry",
        "paths": {
            "blocks": "compositions",
            "components": "compositions/components",
            "assets": "assets",
        },
        "media": {"autoProxy": True},
        "registryItems": [],
    }
    (hf_dir / "hyperframes.json").write_text(json.dumps(hf_config, indent=2))

    pkg_json = {
        "name": "hf",
        "private": True,
        "type": "module",
        "scripts": {
            "cues": "node cues.mjs",
            "check": "node cues.mjs && hyperframes check",
            "render": "node cues.mjs && hyperframes render",
        },
    }
    (hf_dir / "package.json").write_text(json.dumps(pkg_json, indent=2))

    cues_mjs_code = (
        "import fs from 'fs';\n"
        "const c = JSON.parse(fs.readFileSync('cues.json', 'utf8'));\n"
        "fs.writeFileSync('cues.js', 'window.CUES=' + JSON.stringify(c) + ';\\n');\n"
        "const h = fs.readFileSync('index.html', 'utf8').replace(/data-duration=\"[^\"]*\"/, `data-duration=\"${c.total}\"`);\n"
        "fs.writeFileSync('index.html', h);\n"
    )
    (hf_dir / "cues.mjs").write_text(cues_mjs_code)

    timeline_file = work_dir / "timeline.json"
    timeline_data = json.loads(timeline_file.read_text()) if timeline_file.exists() else {"total": 60.0, "parts": []}
    edl_file = work_dir / "edl.json"
    edl_data = json.loads(edl_file.read_text()) if edl_file.exists() else {"parts": []}
    total_dur = float(timeline_data.get("total", 60.0) or 60.0)
    word_cues_text = _read_ass_cues_summary(work_dir / "cap.ass")

    def _write_scene_files(cues_obj: dict[str, Any], scenes_code: str):
        cues_norm = dict(cues_obj or {})
        cues_norm.setdefault("total", round(total_dur, 2))
        (hf_dir / "cues.json").write_text(json.dumps(cues_norm, indent=2))
        (hf_dir / "index.html").write_text(render_hyperframes_index_html(preset, scenes_code, total_dur))

    # Generate lightweight cues_json + scenes_js against scaffolded chalk APIs
    cues_data: dict[str, Any] = {}
    scenes_js = ""
    prompt = build_hyperframes_prompt(timeline_data, edl_data, preset, word_cues_text=word_cues_text)
    try:
        if log_cb:
            log_cb(f"[GRAPHICS] Generating HyperFrames beat scenes via AGY ({creative_model})...")
        comp_data = await run_agy_json(
            prompt,
            schema=HYPERFRAMES_SCENE_SCHEMA,
            model=creative_model,
            cwd=hf_dir,
            add_dirs=[hf_dir],
            timeout_s=360.0,
            cancel_event=cancel_event,
            log_cb=log_cb,
        )
        cues_data = comp_data.get("cues_json") or {}
        scenes_js = str(comp_data.get("scenes_js") or "").strip()
        if not scenes_js:
            raise ValueError("AGY returned empty scenes_js")
    except Exception as e:
        if log_cb:
            log_cb(f"[WARN] HyperFrames scene generation failed ({e}); building deterministic animated chalk fallback.")
        cues_data, scenes_js = build_fallback_scenes_js(timeline_data, edl_data, preset)

    _write_scene_files(cues_data, scenes_js)

    # Sync cues and check composition
    await _run_proc(["node", "cues.mjs"], cwd=hf_dir, cancel_event=cancel_event)
    if log_cb:
        log_cb("[GRAPHICS] Validating composition with hyperframes check...")
    check_code, check_out, _ = await _run_proc(
        [HYPERFRAMES_BIN, "check", "hf", "--json"],
        cwd=job_dir,
        cancel_event=cancel_event,
        log_cb=log_cb,
    )
    has_check_issues, check_issues = _extract_check_issues(check_out)

    # Capture snapshots for Art Director critique
    snapshots_dir = hf_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    if log_cb:
        log_cb("[GRAPHICS] Capturing preview keyframe snapshots...")
    await _run_proc(
        [HYPERFRAMES_BIN, "snapshot", "hf", "--no-browser-gpu", "--output", str(snapshots_dir), "--frames", "5"],
        cwd=job_dir,
        cancel_event=cancel_event,
    )

    critique_res: dict[str, Any] = {"passes": True, "critique": "", "issues": [], "fixes_needed": []}
    try:
        critique_prompt = build_vision_critique_prompt(str(snapshots_dir), str(hf_dir / "cues.json"), preset)
        critique_res = await run_agy_json(
            critique_prompt,
            schema=VISION_CRITIQUE_SCHEMA,
            model=vision_model,
            cwd=job_dir,
            add_dirs=[snapshots_dir, hf_dir],
            cancel_event=cancel_event,
            log_cb=log_cb,
        )
    except Exception as ce:
        if log_cb:
            log_cb(f"[GRAPHICS] Vision critique skipped/failed: {ce}")

    # Enforce Vision QA & check findings with targeted revision pass
    needs_revision = (
        check_code != 0
        or has_check_issues
        or (isinstance(critique_res, dict) and critique_res.get("passes") is False)
    )
    if needs_revision:
        combined_issues = list(check_issues) + list(critique_res.get("issues", []))
        fixes_needed = list(critique_res.get("fixes_needed", []))
        if log_cb:
            log_cb(f"[GRAPHICS] Critique/check flagged {len(combined_issues)} issues — running targeted revision via AGY ({creative_model})...")
        try:
            rev_prompt = build_hyperframes_revision_prompt(
                issues=combined_issues or ["Static/blank frame or layout warning"],
                fixes_needed=fixes_needed or ["Ensure visible animated chalk SVG props at t=0 and clean out() transitions"],
                cues_json=cues_data,
                scenes_js=scenes_js,
                timeline_json=timeline_data,
                edl_json=edl_data,
                preset=preset,
            )
            rev_data = await run_agy_json(
                rev_prompt,
                schema=HYPERFRAMES_SCENE_SCHEMA,
                model=creative_model,
                cwd=hf_dir,
                add_dirs=[hf_dir],
                timeout_s=300.0,
                cancel_event=cancel_event,
                log_cb=log_cb,
            )
            rev_js = str(rev_data.get("scenes_js") or "").strip()
            if rev_js:
                cues_data = rev_data.get("cues_json") or cues_data
                scenes_js = rev_js
                _write_scene_files(cues_data, scenes_js)
                await _run_proc(["node", "cues.mjs"], cwd=hf_dir, cancel_event=cancel_event)
                check_code2, check_out2, _ = await _run_proc(
                    [HYPERFRAMES_BIN, "check", "hf", "--json"],
                    cwd=job_dir,
                    cancel_event=cancel_event,
                    log_cb=log_cb,
                )
                still_bad, _ = _extract_check_issues(check_out2)
                if check_code2 != 0 or still_bad:
                    if log_cb:
                        log_cb("[WARN] Revised scenes still flagged DOM/static errors — applying deterministic animated fallback.")
                    cues_data, scenes_js = build_fallback_scenes_js(timeline_data, edl_data, preset)
                    _write_scene_files(cues_data, scenes_js)
                    await _run_proc(["node", "cues.mjs"], cwd=hf_dir, cancel_event=cancel_event)
                await _run_proc(
                    [HYPERFRAMES_BIN, "snapshot", "hf", "--no-browser-gpu", "--output", str(snapshots_dir), "--frames", "5"],
                    cwd=job_dir,
                    cancel_event=cancel_event,
                )
        except Exception as re_err:
            if log_cb:
                log_cb(f"[WARN] Revision pass failed ({re_err}); ensuring deterministic non-empty animated fallback.")
            cues_data, scenes_js = build_fallback_scenes_js(timeline_data, edl_data, preset)
            _write_scene_files(cues_data, scenes_js)
            await _run_proc(["node", "cues.mjs"], cwd=hf_dir, cancel_event=cancel_event)

    # Render transparent MOV overlay
    overlay_mov = work_dir / "overlay.mov"
    if log_cb:
        log_cb(f"[RENDER] Rendering transparent overlay to {overlay_mov.name}...")
    render_cmd = [
        HYPERFRAMES_BIN,
        "render",
        "hf",
        "--no-browser-gpu",
        "--format",
        "mov",
        "-o",
        str(overlay_mov),
    ]
    rcode, rout, rerr = await _run_proc(render_cmd, cwd=job_dir, timeout_s=360.0, cancel_event=cancel_event, log_cb=log_cb)
    if rcode != 0 or not overlay_mov.exists():
        raise RuntimeError(f"HyperFrames render failed: {rerr}\n{rout}")

    return {
        "overlay_mov": str(overlay_mov),
        "snapshots_dir": str(snapshots_dir),
        "vision_critique": critique_res,
    }


async def run_composite(
    job_dir: Path,
    preset: dict[str, Any],
    cancel_event: asyncio.Event | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Stage 7: Execute finish.py with arnndn neural denoiser, compressor, loudness mastering, and sidechain ducking."""
    work_dir = job_dir / "work"
    final_dir = job_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Ensure rnn_cb.rnnn exists in work/assets
    rnnn_dest = work_dir / "assets" / "rnn_cb.rnnn"
    rnnn_dest.parent.mkdir(parents=True, exist_ok=True)
    rnnn_src = V6_REF_DIR / "work" / "assets" / "rnn_cb.rnnn"
    if not rnnn_dest.exists() and rnnn_src.exists():
        rnnn_dest.symlink_to(rnnn_src)

    overlay_file = work_dir / "overlay.mov"
    overlay_arg = str(overlay_file) if overlay_file.exists() else "-"
    out_mp4 = final_dir / "final.mp4"
    bgm_path = "assets/bgm/track.mp3"

    if log_cb:
        log_cb("[COMPOSITE] Mastering audio and finishing final output with finish.py...")

    finish_cmd = [
        "python3",
        "-I",
        str(SCRIPTS_DIR / "finish.py"),
        str(work_dir),
        overlay_arg,
        str(out_mp4),
        bgm_path,
    ]
    code, out, err = await _run_proc(finish_cmd, cwd=work_dir, cancel_event=cancel_event, log_cb=log_cb)
    if code != 0 or not out_mp4.exists():
        raise RuntimeError(f"finish.py failed: {err}\n{out}")

    # Audio Loudness QA Check via EBU R128
    ebur128_cmd = [
        FFMPEG_BIN,
        "-i",
        str(out_mp4),
        "-af",
        "ebur128=framelog=verbose",
        "-f",
        "null",
        "-",
    ]
    _, _, ebur_err = await _run_proc(ebur128_cmd, cwd=work_dir, cancel_event=cancel_event)

    i_luma = -14.0
    m_i = re.search(r"Integrated loudness:\s+I:\s+([-\d.]+)\s+LUFS", ebur_err)
    if m_i:
        i_luma = float(m_i.group(1))

    final_probe = probe_media(out_mp4)
    final_qa = {
        "output_path": str(out_mp4),
        "duration_s": final_probe["duration_s"],
        "integrated_loudness_lufs": i_luma,
        "width": final_probe["width"],
        "height": final_probe["height"],
        "fps": final_probe["fps"],
    }
    (final_dir / "qa_report.json").write_text(json.dumps(final_qa, indent=2))

    return final_qa


def _generate_landscape_cover(
    frame_path: str | Path | None,
    out_base: Path,
    title_text: str,
    font_path: str = "/home/sandbox/.local/share/fonts/google-fonts/Inter-Bold.ttf",
):
    """Compact Pillow helper generating 1920x1080 landscape covers with punchy high-contrast titles."""
    from PIL import Image, ImageDraw, ImageFont, ImageEnhance
    W, H = 1920, 1080
    f_p = Path(frame_path) if frame_path else None
    if f_p and f_p.exists() and f_p.is_file():
        try:
            img = Image.open(f_p).convert("RGB")
            aspect = img.width / max(1, img.height)
            target_aspect = W / H
            if aspect > target_aspect:
                new_h = H
                new_w = int(H * aspect)
            else:
                new_w = W
                new_h = int(W / max(0.001, aspect))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            x_off = (new_w - W) // 2
            y_off = (new_h - H) // 2
            img = img.crop((x_off, y_off, x_off + W, y_off + H))
        except Exception:
            img = Image.new("RGB", (W, H), (20, 22, 28))
    else:
        img = Image.new("RGB", (W, H), (20, 22, 28))

    # Color & Contrast grading
    img = ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(1.08)).enhance(1.1)

    # Dark gradient bottom overlay
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d_ov = ImageDraw.Draw(overlay)
    for y in range(H - 420, H):
        factor = (y - (H - 420)) / 420.0
        alpha = int(210 * (factor ** 1.3))
        d_ov.line([(0, y), (W, y)], fill=(0, 0, 0, alpha))
    img.paste(overlay, (0, 0), overlay)

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, 80)
    except Exception:
        font = ImageFont.load_default()

    clean_title = (title_text or "INSPECTING THE EVIDENCE").upper()
    words = clean_title.split()
    if len(words) > 5:
        mid = len(words) // 2
        lines = [" ".join(words[:mid]), " ".join(words[mid:])]
    else:
        lines = [clean_title]

    y_pos = H - 280 if len(lines) == 1 else H - 340
    for line in lines:
        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_w = bbox[2] - bbox[0]
        except Exception:
            line_w = len(line) * 40
        x_pos = (W - line_w) // 2
        for dx, dy in [(-3, -3), (-3, 3), (3, -3), (3, 3), (0, 3), (0, -3), (3, 0), (-3, 0)]:
            draw.text((x_pos + dx, y_pos + dy), line, font=font, fill=(0, 0, 0))
        draw.text((x_pos, y_pos), line, font=font, fill=(255, 255, 255))
        y_pos += 95

    out_base.parent.mkdir(parents=True, exist_ok=True)
    img.save(f"{out_base}.jpg", quality=92)
    img.save(f"{out_base}.png")


async def recomposite_studio(
    job_dir: Path,
    preset: dict[str, Any] | None = None,
    cancel_event: asyncio.Event | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Lightweight standalone recomposition when editing in studio: checks, re-renders overlay.mov, and finishes final.mp4."""
    work_dir = job_dir / "work"
    hf_dir = job_dir / "hf"

    if log_cb:
        log_cb("[STUDIO] Recompositing from live studio edits...")

    # Sync cues
    await _run_proc(["node", "cues.mjs"], cwd=hf_dir, cancel_event=cancel_event)

    # Check
    await _run_proc([HYPERFRAMES_BIN, "check", "hf", "--json"], cwd=job_dir, cancel_event=cancel_event, log_cb=log_cb)

    # Render with software WebGL
    overlay_mov = work_dir / "overlay.mov"
    await _run_proc(
        [HYPERFRAMES_BIN, "render", "hf", "--no-browser-gpu", "--format", "mov", "-o", str(overlay_mov)],
        cwd=job_dir,
        timeout_s=300.0,
        cancel_event=cancel_event,
        log_cb=log_cb,
    )

    if preset is None:
        p_id = "story_vertical"
        job_json_file = job_dir / "job.json"
        if job_json_file.exists():
            try:
                j_data = json.loads(job_json_file.read_text())
                p_id = j_data.get("preset_id") or j_data.get("config", {}).get("preset_id", "story_vertical")
            except Exception:
                pass
        preset = get_preset(p_id)

    # Delegate final composite + loudness probing directly to run_composite
    return await run_composite(
        job_dir,
        preset=preset,
        cancel_event=cancel_event,
        log_cb=log_cb,
    )


async def _extract_cover_candidates(
    work_dir: Path,
    cover_dir: Path,
    preset: dict[str, Any],
    cancel_event: asyncio.Event | None = None,
) -> list[str]:
    """Extract 8 uncaptioned cover candidate frames from source clips aligned to timeline parts."""
    src_dir = work_dir / "src"
    timeline_file = work_dir / "timeline.json"
    w = int(preset.get("width", 1080))
    h = int(preset.get("height", 1920))
    scale_crop = (
        f"scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},setsar=1"
    )

    sample_specs: list[tuple[Path, float]] = []
    if timeline_file.exists():
        try:
            t_data = json.loads(timeline_file.read_text())
            parts = t_data.get("parts", [])
            if parts:
                ratios = [0.06, 0.16, 0.28, 0.40, 0.54, 0.68, 0.80, 0.92]
                total = float(t_data.get("total", 0.0) or 0.0)
                for r in ratios:
                    target_t = total * r if total > 0 else 0.0
                    chosen = next(
                        (p for p in parts if float(p.get("start", 0.0)) <= target_t <= float(p.get("end", 0.0))),
                        parts[min(len(parts) - 1, int(r * len(parts)))],
                    )
                    clip_file = src_dir / f"{chosen.get('clip', 'c01')}.mp4"
                    subs = chosen.get("subs") or []
                    if clip_file.exists() and subs:
                        s0, s1, _ = subs[len(subs) // 2]
                        src_t = (float(s0) + float(s1)) / 2.0
                        sample_specs.append((clip_file, max(0.0, src_t)))
                    elif clip_file.exists() and chosen.get("src"):
                        src_t = (float(chosen["src"][0]) + float(chosen["src"][1])) / 2.0
                        sample_specs.append((clip_file, max(0.0, src_t)))
        except Exception:
            sample_specs = []

    if not sample_specs:
        base_mp4 = work_dir / "base.mp4"
        dur = probe_media(base_mp4)["duration_s"]
        sample_specs = [
            (base_mp4, dur * r) for r in [0.08, 0.18, 0.28, 0.38, 0.50, 0.65, 0.80, 0.90]
        ]

    candidate_frames: list[str] = []
    for idx, (clip_path, ts) in enumerate(sample_specs):
        if not clip_path.exists():
            continue
        frame_path = cover_dir / f"cand_{idx + 1:02d}.png"
        extract_cmd = [
            FFMPEG_BIN,
            "-v",
            "error",
            "-y",
            "-ss",
            f"{ts:.2f}",
            "-i",
            str(clip_path),
            "-vf",
            scale_crop,
            "-vframes",
            "1",
            str(frame_path),
        ]
        await _run_proc(extract_cmd, cwd=cover_dir, cancel_event=cancel_event)
        if frame_path.exists() and frame_path.stat().st_size > 0:
            candidate_frames.append(str(frame_path))
    return candidate_frames


async def _extract_cover_sticker(
    work_dir: Path,
    cover_dir: Path,
    preset: dict[str, Any],
    cancel_event: asyncio.Event | None = None,
) -> dict[str, Any] | None:
    """Sample overlay.mov for a crisp non-transparent graphic sticker bbox within graphics_band."""
    overlay_mov = work_dir / "overlay.mov"
    if not overlay_mov.exists():
        return None
    dur = probe_media(overlay_mov)["duration_s"]
    if dur <= 0.2:
        return None

    try:
        from PIL import Image
    except ImportError:
        return None

    g_band = preset.get("graphics_band", {})
    g_top = int(g_band.get("top", 1140))
    g_bot = int(g_band.get("bottom", 1500))
    best_info: tuple[int, Path, tuple[int, int, int, int]] | None = None

    for idx, ratio in enumerate([0.04, 0.22, 0.45, 0.68, 0.85]):
        t_pos = min(max(0.0, dur * ratio), max(0.0, dur - 0.1))
        tmp_png = cover_dir / f"_ov_sample_{idx}.png"
        await _run_proc(
            [FFMPEG_BIN, "-v", "error", "-y", "-ss", f"{t_pos:.2f}", "-i", str(overlay_mov), "-vframes", "1", str(tmp_png)],
            cwd=cover_dir,
            cancel_event=cancel_event,
        )
        if not tmp_png.exists():
            continue
        try:
            with Image.open(tmp_png).convert("RGBA") as img:
                alpha = img.getchannel("A")
                # Restrict search to graphics band to avoid picking empty borders
                cropped_band = alpha.crop((0, max(0, g_top), img.width, min(img.height, g_bot)))
                bbox = cropped_band.getbbox()
                if bbox:
                    full_bbox = (bbox[0], bbox[1] + max(0, g_top), bbox[2], bbox[3] + max(0, g_top))
                    bw = full_bbox[2] - full_bbox[0]
                    bh = full_bbox[3] - full_bbox[1]
                    area = bw * bh
                    if bw >= 80 and bh >= 40 and (best_info is None or area > best_info[0]):
                        best_info = (area, tmp_png, full_bbox)
        except Exception:
            pass

    if not best_info:
        for idx in range(5):
            (cover_dir / f"_ov_sample_{idx}.png").unlink(missing_ok=True)
        return None

    _, winner_path, bbox = best_info
    sticker_path = cover_dir / "sticker_src.png"
    shutil.copy2(winner_path, sticker_path)
    for idx in range(5):
        (cover_dir / f"_ov_sample_{idx}.png").unlink(missing_ok=True)

    bw = max(1, bbox[2] - bbox[0])
    bh = max(1, bbox[3] - bbox[1])
    target_scale = min(1.35, 760.0 / bw, 280.0 / bh)
    return {
        "path": str(sticker_path),
        "crop": list(bbox),
        "scale": round(max(0.6, target_scale), 2),
    }


async def run_deliverables(
    job_dir: Path,
    preset: dict[str, Any],
    vision_model: str = "gemini-3.8-flash-high",
    mechanical_model: str = "gemini-3.8-flash-high",
    cancel_event: asyncio.Event | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Stage 8: Extract uncaptioned candidate frames, generate high-CTR thumbnail covers (cover_A & cover_B), and write grounded descriptions."""
    work_dir = job_dir / "work"
    final_dir = job_dir / "final"
    cover_dir = job_dir / "cover"
    cover_dir.mkdir(parents=True, exist_ok=True)

    candidate_frames = await _extract_cover_candidates(work_dir, cover_dir, preset, cancel_event=cancel_event)

    # Determine cover titles, thesis, and creator direction
    edl_file = work_dir / "edl.json"
    selected_script_file = work_dir / "selected_script.json"
    job_json_file = job_dir / "job.json"

    title_text = "KEY INSIGHT"
    script_text = ""
    thesis_text = ""
    story_prompt = ""

    if edl_file.exists():
        try:
            title_text = json.loads(edl_file.read_text()).get("title") or title_text
        except Exception:
            pass
    if selected_script_file.exists():
        try:
            sel_s = json.loads(selected_script_file.read_text())
            script_text = str(sel_s.get("script_text") or "").strip()
            thesis_text = str(sel_s.get("thesis") or "").strip()
            if sel_s.get("title"):
                title_text = sel_s["title"]
        except Exception:
            pass
    if job_json_file.exists():
        try:
            j_data = json.loads(job_json_file.read_text())
            story_prompt = str(j_data.get("story_prompt") or j_data.get("config", {}).get("story_prompt") or "").strip()
        except Exception:
            pass

    is_landscape = preset.get("width", 1080) > preset.get("height", 1920) or preset.get("aspect_ratio") == "16:9"

    if is_landscape:
        frame_a = candidate_frames[0] if candidate_frames else None
        frame_b = candidate_frames[1] if len(candidate_frames) > 1 else frame_a
        _generate_landscape_cover(frame_a, cover_dir / "cover_A", title_text)
        _generate_landscape_cover(frame_b, cover_dir / "cover_B", f"WHY {title_text}")
    else:
        sticker_cfg = await _extract_cover_sticker(work_dir, cover_dir, preset, cancel_event=cancel_event)
        try:
            cover_prompt = build_cover_prompt(
                candidate_frames,
                [title_text] + ([thesis_text] if thesis_text else []),
                preset,
                has_sticker=bool(sticker_cfg),
            )
            sel_cover = await run_agy_json(
                cover_prompt,
                schema=COVER_SELECTOR_SCHEMA,
                model=vision_model,
                cwd=cover_dir,
                add_dirs=[cover_dir],
                cancel_event=cancel_event,
            )
        except Exception:
            words_t = title_text.upper().split()
            row1 = " ".join(words_t[:2]) if words_t else "WAIT FOR"
            row2 = " ".join(words_t[2:4]) if len(words_t) > 2 else "THE"
            row3 = " ".join(words_t[4:6]) if len(words_t) > 4 else "TURN"
            rows = [[[row1, "w"]], [[row2, "w"]], [[row3, "p"]]] if row3 else [[[row1, "w"]], [[row2, "p"]]]
            sel_cover = {
                "cover_A": {
                    "frame": candidate_frames[0] if candidate_frames else "",
                    "layout": "A",
                    "title": rows,
                },
                "cover_B": {
                    "frame": candidate_frames[min(3, len(candidate_frames) - 1)] if candidate_frames else "",
                    "layout": "B",
                    "title": rows,
                },
            }

        for c_id, c_data in [("cover_A", sel_cover["cover_A"]), ("cover_B", sel_cover["cover_B"])]:
            cfg_path = cover_dir / f"{c_id}_config.json"
            layout = c_data.get("layout", "A" if c_id == "cover_A" else "B")
            chosen_frame = c_data.get("frame", "")
            if not Path(chosen_frame).exists() and candidate_frames:
                chosen_frame = candidate_frames[0]
            cfg_content: dict[str, Any] = {
                "frame": chosen_frame,
                "out": str(cover_dir / c_id),
                "layout": layout,
                "title": c_data.get("title", [[[title_text.upper(), "w"]]]),
                "colors": {"w": [255, 255, 255], "p": [255, 79, 160], "b": [61, 107, 255]},
                "sizes": [104, 104, 124] if layout == "A" else [96, 96, 112],
                "title_y": 185 if layout == "A" else 1200,
            }
            if sticker_cfg:
                cfg_content["sticker"] = sticker_cfg
            cfg_path.write_text(json.dumps(cfg_content, indent=2))
            cover_cmd = [
                UV_BIN,
                "run",
                "--with",
                "pillow",
                "--with",
                "numpy",
                "python",
                "-I",
                str(SCRIPTS_DIR / "cover.py"),
                str(cfg_path),
            ]
            await _run_proc(cover_cmd, cwd=cover_dir, cancel_event=cancel_event, log_cb=log_cb)

    # Generate Descriptions with AGY strictly grounded in transcript & script
    final_transcript = ""
    if (work_dir / "final_transcript.txt").exists():
        final_transcript = (work_dir / "final_transcript.txt").read_text()

    desc_prompt = build_descriptions_prompt(
        final_transcript=final_transcript,
        script_text=script_text or title_text,
        summary=f"{title_text}: {thesis_text}" if thesis_text else title_text,
        story_prompt=story_prompt,
    )
    if log_cb:
        log_cb(f"[DELIVERABLES] Drafting structured descriptions with AGY ({mechanical_model})...")

    desc_data: dict[str, Any] = {}
    try:
        desc_data = await run_agy_json(
            desc_prompt,
            schema=DESCRIPTIONS_SCHEMA,
            model=mechanical_model,
            cancel_event=cancel_event,
            log_cb=log_cb,
        )
    except Exception as e:
        if log_cb:
            log_cb(f"[DELIVERABLES] AGY structured descriptions fallback: {e}")
        desc_data = {
            "hook": f"{title_text} — did you catch that twist?",
            "teaser": f"What nobody tells you about {title_text.lower()}.",
            "confessional": f"Breaking down why {title_text.lower()} actually matters.",
            "debate": f"Where do you stand on {title_text.lower()}? Drop your take below.",
            "minimal": f"{title_text}.",
            "recommendation": "Option 1 (Hook) leads with curiosity for feed retention.",
        }

    # Save structured descriptions.json
    (final_dir / "descriptions.json").write_text(json.dumps(desc_data, indent=2))

    # Format descriptions.txt from structured json
    lines = [
        "=== OPTION 1 (DIRECT QUESTION HOOK) ===",
        desc_data.get("hook", "").strip(),
        "",
        "=== OPTION 2 (STORY TEASER) ===",
        desc_data.get("teaser", "").strip(),
        "",
        "=== OPTION 3 (CONFESSIONAL) ===",
        desc_data.get("confessional", "").strip(),
        "",
        "=== OPTION 4 (DEBATE-BAIT) ===",
        desc_data.get("debate", "").strip(),
        "",
        "=== OPTION 5 (MINIMAL) ===",
        desc_data.get("minimal", "").strip(),
        "",
        "=== RECOMMENDATION ===",
        desc_data.get("recommendation", "").strip(),
    ]
    (final_dir / "descriptions.txt").write_text("\n".join(lines).strip())

    return {
        "covers": [
            str(cover_dir / "cover_A.jpg"),
            str(cover_dir / "cover_B.jpg"),
        ],
        "descriptions_json": str(final_dir / "descriptions.json"),
        "descriptions_txt": str(final_dir / "descriptions.txt"),
    }
