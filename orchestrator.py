"""Job orchestrator, stage state machine, and Studio preview manager.

Coordinates:
- Atomic job state persistence (job.json)
- 8-stage state machine with artifact-check resumability
- Gate A (script approval) and Gate B (audio approval) workflows
- SSE pub/sub log and event streaming
- Studio preview management with hyperframes preview on ports 3002..3030
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from .adapters import (
        recomposite_studio,
        run_agy_copilot_turn,
        run_agy_json,
        run_bgm_and_broll,
        run_build_and_qa,
        run_composite,
        run_deliverables,
        run_ingest_transcribe,
        run_motion_graphics,
    )
    from .agent_prompts import (
        CRITIC_SCHEMA,
        EDL_SCHEMA,
        SCRIPT_OPTIONS_SCHEMA,
        build_edl_prompt,
        build_script_critic_prompt,
        build_script_generation_prompt,
        format_words_for_edl,
        validate_and_fix_edl_overlaps,
    )
    from .presets import (
        AGY_MODELS,
        DEFAULT_MODEL_SETTINGS,
        JOBS_ROOT,
        STAGE_ORDER,
        get_preset,
        validate_model_id,
    )
except ImportError:
    from adapters import (
        recomposite_studio,
        run_agy_copilot_turn,
        run_agy_json,
        run_bgm_and_broll,
        run_build_and_qa,
        run_composite,
        run_deliverables,
        run_ingest_transcribe,
        run_motion_graphics,
    )
    from agent_prompts import (
        CRITIC_SCHEMA,
        EDL_SCHEMA,
        SCRIPT_OPTIONS_SCHEMA,
        build_edl_prompt,
        build_script_critic_prompt,
        build_script_generation_prompt,
        format_words_for_edl,
        validate_and_fix_edl_overlaps,
    )
    from presets import (
        AGY_MODELS,
        DEFAULT_MODEL_SETTINGS,
        JOBS_ROOT,
        STAGE_ORDER,
        get_preset,
        validate_model_id,
    )


HYPERFRAMES_BIN = "/home/sandbox/tools/bin/hyperframes"
GUIDE_MD_PATH = Path("/home/sandbox/ContentCreation/GUIDE.md")


# ============================================================================
# STUDIO MANAGER (HyperFrames Preview Server on Ephemeral Ports 3002..3030)
# ============================================================================

class StudioManager:
    """Manages ephemeral HyperFrames preview servers for interactive composition editing."""

    def __init__(self, port_range: tuple[int, int] = (3002, 3030)):
        self.port_min, self.port_max = port_range
        self._studios: dict[str, dict[str, Any]] = {}

    def _find_free_port(self) -> int:
        for port in range(self.port_min, self.port_max + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError(f"No available ports in range {self.port_min}..{self.port_max}")

    async def open_studio(self, job_id: str, hf_dir: Path | str) -> dict[str, Any]:
        """Launch or return existing preview server for a job's HyperFrames directory."""
        hf_path = Path(hf_dir).resolve()
        if not hf_path.exists():
            raise FileNotFoundError(f"HyperFrames directory not found: {hf_path}")

        # Check existing studio instance
        if job_id in self._studios:
            info = self._studios[job_id]
            proc = info.get("proc")
            if proc and proc.returncode is None:
                return {
                    "job_id": job_id,
                    "port": info["port"],
                    "url": f"http://localhost:{info['port']}",
                    "status": "running",
                }

        port = self._find_free_port()

        # Launch hyperframes preview server with software WebGL
        cmd = [
            HYPERFRAMES_BIN,
            "preview",
            str(hf_path),
            f"--port={port}",
            "--no-open",
            "--no-browser-gpu",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(hf_path.parent),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        info = {
            "job_id": job_id,
            "port": port,
            "url": f"http://localhost:{port}",
            "hf_dir": str(hf_path),
            "proc": proc,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self._studios[job_id] = info

        # Brief delay to allow HTTP server binding
        await asyncio.sleep(0.5)

        # Verify process didn't exit immediately
        if proc.returncode is not None:
            stdout, stderr = await proc.communicate()
            err_msg = stderr.decode(errors="replace") if stderr else stdout.decode(errors="replace")
            self._studios.pop(job_id, None)
            raise RuntimeError(f"HyperFrames preview exited immediately with code {proc.returncode}: {err_msg}")

        return {
            "job_id": job_id,
            "port": port,
            "url": f"http://localhost:{port}",
            "status": "running",
        }

    async def stop_studio(self, job_id: str) -> bool:
        """Stop running preview server for a job."""
        info = self._studios.pop(job_id, None)
        if not info:
            return False

        proc = info.get("proc")
        if proc and proc.returncode is None:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

        # Also trigger CLI stop fallback
        hf_dir = info.get("hf_dir")
        if hf_dir:
            try:
                subprocess.run(
                    [HYPERFRAMES_BIN, "preview", str(hf_dir), "--stop"],
                    capture_output=True,
                    timeout=5.0,
                )
            except Exception:
                pass
        return True

    def get_studio(self, job_id: str) -> dict[str, Any] | None:
        """Retrieve active studio preview server info for a job."""
        info = self._studios.get(job_id)
        if not info:
            return None
        proc = info.get("proc")
        is_alive = proc is not None and proc.returncode is None
        return {
            "job_id": job_id,
            "port": info["port"],
            "url": info["url"],
            "status": "running" if is_alive else "stopped",
        }

    async def stop_all(self):
        """Terminate all registered preview servers."""
        for jid in list(self._studios.keys()):
            await self.stop_studio(jid)


# ============================================================================
# JOB MANAGER & PIPELINE ORCHESTRATOR
# ============================================================================

class JobManager:
    """Manages video production job lifecycles, atomic state persistence, and stage execution."""

    def __init__(self, jobs_root: Path = JOBS_ROOT):
        self.jobs_root = Path(jobs_root).resolve()
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.studio_manager = StudioManager()

        self._active_tasks: dict[str, asyncio.Task[Any]] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
        self._copilot_locks: dict[str, asyncio.Lock] = {}
        self._copilot_busy: dict[str, bool] = {}

    # ------------------------------------------------------------------------
    # PUB / SUB (SSE STREAMING)
    # ------------------------------------------------------------------------

    def subscribe(self, job_id: str) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to live real-time event updates for a job."""
        if job_id not in self._subscribers:
            self._subscribers[job_id] = set()
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers[job_id].add(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue[dict[str, Any]]):
        """Unsubscribe an event listener queue."""
        if job_id in self._subscribers:
            self._subscribers[job_id].discard(q)
            if not self._subscribers[job_id]:
                del self._subscribers[job_id]

    def publish(self, job_id: str, event_type: str, data: dict[str, Any]):
        """Publish an event to all active subscribers of a job."""
        event = {
            "event": event_type,
            "type": event_type,
            "job_id": job_id,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        for q in list(self._subscribers.get(job_id, [])):
            try:
                q.put_nowait(event)
            except Exception:
                pass

    # ------------------------------------------------------------------------
    # JOB PERSISTENCE (ATOMIC)
    # ------------------------------------------------------------------------

    def _save_job(self, job: dict[str, Any]):
        """Atomically persist job state to job.json via temporary file swap."""
        job_dir = self.jobs_root / job["id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        job_file = job_dir / "job.json"
        tmp_file = job_dir / "job.json.tmp"

        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp_file.write_text(json.dumps(job, indent=2))
        os.replace(tmp_file, job_file)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Load job.json for a job if it exists."""
        job_file = self.jobs_root / job_id / "job.json"
        if not job_file.exists():
            return None
        try:
            return json.loads(job_file.read_text())
        except Exception:
            return None

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all known jobs sorted by creation time descending."""
        jobs: list[dict[str, Any]] = []
        if not self.jobs_root.exists():
            return jobs

        for d in self.jobs_root.iterdir():
            if d.is_dir() and (d / "job.json").exists():
                try:
                    data = json.loads((d / "job.json").read_text())
                    jobs.append(data)
                except Exception:
                    continue
        jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
        return jobs

    def create_job(
        self,
        preset_id: str = "story_vertical",
        selected_clips: list[str] | None = None,
        pause_at_script: bool | None = None,
        pause_at_audio: bool | None = None,
        broll_provider: str | None = None,
        config: dict[str, Any] | None = None,
        clip_paths: list[str] | None = None,
        start_immediately: bool = False,
        job_id: str | None = None,
        story_prompt: str = "",
        whisper_model: str = "small.en",
    ) -> dict[str, Any]:
        """Initialize a new pipeline job and persist initial job.json."""
        jid = job_id or f"job_{int(time.time())}_{os.urandom(3).hex()}"
        job_dir = self.jobs_root / jid
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "work").mkdir(exist_ok=True)

        cfg = dict(config or {})
        pid = cfg.get("preset_id") or preset_id or "story_vertical"
        sp = cfg.get("story_prompt") if "story_prompt" in cfg else story_prompt
        # Autonomous by default: pause flags default to False
        pas = cfg.get("gate_a") if "gate_a" in cfg else (pause_at_script if pause_at_script is not None else False)
        paa = cfg.get("gate_b") if "gate_b" in cfg else (pause_at_audio if pause_at_audio is not None else False)
        wm = cfg.get("whisper_model") or whisper_model or "small.en"
        raw_broll = cfg.get("ai_broll") or broll_provider or "off"
        bp = "comfyui" if raw_broll == "comfy" else raw_broll
        clips = clip_paths if clip_paths is not None else (selected_clips or [])

        stages_dict: dict[str, Any] = {}
        for s in STAGE_ORDER:
            stages_dict[s] = {
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "error": None,
                "artifacts": {},
            }

        user_models = cfg.get("models") or {}
        models_dict = {
            slot: validate_model_id(user_models.get(slot, default_val))
            for slot, default_val in DEFAULT_MODEL_SETTINGS.items()
        }
        if "mechanical_model" in user_models and "vision_model" not in user_models:
            models_dict["vision_model"] = models_dict["mechanical_model"]

        job_config = {
            "preset_id": pid,
            "story_prompt": sp,
            "gate_a": bool(pas),
            "gate_b": bool(paa),
            "whisper_model": wm,
            "ai_broll": bp,
            "models": models_dict,
        }

        job = {
            "id": jid,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "config": job_config,
            "preset_id": pid,
            "story_prompt": sp,
            "status": "idle",
            "current_stage": "ingest",
            "stages": stages_dict,
            "pause_at_script": bool(pas),
            "pause_at_audio": bool(paa),
            "whisper_model": wm,
            "selected_clips": clips,
            "broll_provider": bp,
            "error": None,
        }
        self._save_job(job)

        if start_immediately:
            try:
                loop = asyncio.get_running_loop()
                if loop and loop.is_running():
                    self.start_job(jid)
            except RuntimeError:
                pass

        return job

    # ------------------------------------------------------------------------
    # CO-PILOT & MODEL CONFIGURATION
    # ------------------------------------------------------------------------

    def _get_copilot_lock(self, job_id: str) -> asyncio.Lock:
        if job_id not in self._copilot_locks:
            self._copilot_locks[job_id] = asyncio.Lock()
        return self._copilot_locks[job_id]

    def is_copilot_busy(self, job_id: str) -> bool:
        if self._copilot_busy.get(job_id, False):
            return True
        lock = self._copilot_locks.get(job_id)
        return lock.locked() if lock else False

    def update_job_models(self, job_id: str, models_patch: dict[str, Any]) -> dict[str, str]:
        """Update model assignments for a job, persist state, and publish SSE event."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        cfg = job.setdefault("config", {})
        current_models = cfg.get("models") or {}
        updated_models: dict[str, str] = {}
        for slot, default_val in DEFAULT_MODEL_SETTINGS.items():
            if slot in models_patch and models_patch[slot]:
                updated_models[slot] = validate_model_id(models_patch[slot])
            elif slot in current_models and current_models[slot]:
                updated_models[slot] = validate_model_id(current_models[slot])
            else:
                updated_models[slot] = default_val

        # Vision QA shares the Mechanical & QA selector unless explicitly overridden.
        if (
            "mechanical_model" in models_patch
            and models_patch["mechanical_model"]
            and not ("vision_model" in models_patch and models_patch["vision_model"])
        ):
            updated_models["vision_model"] = updated_models["mechanical_model"]

        job["config"]["models"] = updated_models
        self._save_job(job)
        self.publish(job_id, "models_updated", {"models": updated_models})
        return updated_models

    def get_copilot_history(self, job_id: str) -> dict[str, Any]:
        """Retrieve conversation history and active conversation ID for a job's Co-Pilot."""
        job_dir = self.jobs_root / job_id
        history_file = job_dir / "copilot_history.json"
        if history_file.exists():
            try:
                data = json.loads(history_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data.setdefault("conversation_id", None)
                    data.setdefault("messages", [])
                    return data
            except Exception:
                pass
        return {"conversation_id": None, "messages": []}

    def reset_copilot_session(self, job_id: str) -> dict[str, Any]:
        """Clear conversation_id for a job's Co-Pilot session, keeping a clean system marker."""
        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        history = self.get_copilot_history(job_id)
        history["conversation_id"] = None
        marker_msg = {
            "id": f"msg_{uuid.uuid4().hex[:8]}",
            "role": "system",
            "content": "Conversation session reset by user. Next turn will re-seed full job context.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        history["messages"].append(marker_msg)
        history_file = job_dir / "copilot_history.json"
        history_file.write_text(json.dumps(history, indent=2))
        self.publish(job_id, "copilot_reset", {"conversation_id": None})
        return history

    async def send_copilot_message(
        self,
        job_id: str,
        message: str,
        model: str | None = None,
        stage_hint: str | None = None,
    ) -> dict[str, Any]:
        """Execute a single interactive Antigravity Co-Pilot turn against a job with artifact tracking."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        job_dir = self.jobs_root / job_id
        lock = self._get_copilot_lock(job_id)
        if lock.locked():
            raise RuntimeError("Co-Pilot is already busy processing a request for this job.")

        async with lock:
            self._copilot_busy[job_id] = True
            try:
                # Helper to snapshot artifact modification times
                def _snapshot_artifacts(d: Path) -> dict[str, float]:
                    snaps = {}
                    for sub in ["work", "hf", "cover", "final"]:
                        sp = d / sub
                        if sp.exists():
                            for p in sp.rglob("*"):
                                if p.is_file():
                                    try:
                                        rel = str(p.relative_to(d))
                                        snaps[rel] = p.stat().st_mtime
                                    except Exception:
                                        pass
                    return snaps

                mtimes_before = _snapshot_artifacts(job_dir)
                history = self.get_copilot_history(job_id)
                conversation_id = history.get("conversation_id")

                # Resolve model
                cfg_models = job.get("config", {}).get("models", {})
                chosen_model = validate_model_id(model or cfg_models.get("copilot_model"))

                # First message or reset session: inject rich job context
                if not conversation_id:
                    work_dir = job_dir / "work"
                    hf_dir = job_dir / "hf"
                    cover_dir = job_dir / "cover"
                    final_dir = job_dir / "final"

                    avail_files = []
                    for p in [
                        work_dir / "transcript_all.txt",
                        work_dir / "script_options.json",
                        work_dir / "selected_script.json",
                        work_dir / "edl.json",
                        work_dir / "qa_report.json",
                        hf_dir / "index.html",
                        hf_dir / "cues.json",
                        final_dir / "descriptions.json",
                    ]:
                        if p.exists():
                            avail_files.append(str(p.relative_to(job_dir)))
                    if cover_dir.exists():
                        for c in cover_dir.iterdir():
                            if c.is_file():
                                avail_files.append(str(c.relative_to(job_dir)))

                    context_blocks = [
                        f"You are the embedded Antigravity Co-Pilot for video production job '{job_id}'.",
                        f"- Job Directory: {job_dir.resolve()}",
                        f"- Preset: {job.get('preset_id', 'story_vertical')}",
                        f"- Current Stage: {job.get('current_stage', 'ingest')}",
                        f"- Stage Hint: {stage_hint or 'N/A'}",
                        f"- Selected Clips: {json.dumps(job.get('selected_clips', []))}",
                        f"- Available Artifact Files: {json.dumps(avail_files)}",
                        "",
                        "CRITICAL INSTRUCTIONS:",
                        "1. You have full file-edit permissions within the job directory.",
                        "2. Keep edits minimal, precise, and compliant with existing pipeline schemas (e.g. edl.json, script_options.json, hf/index.html, hf/cues.json).",
                        "3. Respond clearly explaining what you inspected, analyzed, or modified.",
                        "",
                        f"User Request:\n{message}",
                    ]
                    full_prompt = "\n".join(context_blocks)
                else:
                    if stage_hint:
                        full_prompt = f"[Context Hint: Current stage is {stage_hint}]\n{message}"
                    else:
                        full_prompt = message

                res = await run_agy_copilot_turn(
                    prompt=full_prompt,
                    job_dir=job_dir,
                    model=chosen_model,
                    conversation_id=conversation_id,
                    timeout_s=600.0,
                    log_cb=lambda txt: self.publish(job_id, "log", {"message": txt}),
                    progress_cb=lambda payload: self.publish(job_id, "copilot_progress", payload),
                )

                new_conv_id = res.get("conversation_id") or conversation_id
                history["conversation_id"] = new_conv_id
                reply_text = res.get("response_text", "")

                # Detect modified files
                mtimes_after = _snapshot_artifacts(job_dir)
                modified_files = []
                for rel, mtime in mtimes_after.items():
                    if rel not in mtimes_before or mtime > mtimes_before[rel] + 0.001:
                        modified_files.append(rel)

                # Infer suggested next action
                suggested_action = None
                mod_set = set(modified_files)
                if any("edl.json" in f for f in mod_set):
                    suggested_action = {
                        "label": "Rebuild Base Cut",
                        "type": "start_stage",
                        "stage": "base_cut",
                    }
                elif any("hf/index.html" in f or "hf/cues.json" in f or "cues.mjs" in f for f in mod_set):
                    suggested_action = {
                        "label": "Re-render Studio & Composite",
                        "type": "recomposite",
                    }
                elif any("script_options.json" in f or "selected_script.json" in f for f in mod_set):
                    suggested_action = {
                        "label": "Refresh Script Options (Gate A)",
                        "type": "refresh_gate",
                        "gate": "A",
                    }

                user_msg = {
                    "id": f"msg_{uuid.uuid4().hex[:8]}",
                    "role": "user",
                    "content": message,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                assistant_msg = {
                    "id": f"msg_{uuid.uuid4().hex[:8]}",
                    "role": "assistant",
                    "content": reply_text,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": chosen_model,
                    "duration_seconds": res.get("duration_seconds", 0.0),
                    "usage": res.get("usage", {}),
                    "modified_files": modified_files,
                    "suggested_action": suggested_action,
                }
                history["messages"].append(user_msg)
                history["messages"].append(assistant_msg)

                (job_dir / "copilot_history.json").write_text(json.dumps(history, indent=2))

                self.publish(job_id, "copilot_reply", {
                    "message": assistant_msg,
                    "modified_files": modified_files,
                    "suggested_action": suggested_action,
                    "conversation_id": new_conv_id,
                })

                refreshed_job = self.get_job(job_id) or job
                return {
                    "message": assistant_msg,
                    "copilot": history,
                    "modified_files": modified_files,
                    "suggested_action": suggested_action,
                    "job": refreshed_job,
                }
            finally:
                self._copilot_busy[job_id] = False

    # ------------------------------------------------------------------------
    # ARTIFACT RESUMABILITY CHECK
    # ------------------------------------------------------------------------

    def _determine_resume_stage(self, job_dir: Path) -> str:
        """Inspect disk artifacts to determine the earliest incomplete stage."""
        work_dir = job_dir / "work"
        final_dir = job_dir / "final"
        cover_dir = job_dir / "cover"
        hf_dir = job_dir / "hf"

        if (final_dir / "final.mp4").exists() and (final_dir / "descriptions.txt").exists():
            return "deliverables"
        if (final_dir / "final.mp4").exists():
            return "deliverables"
        if (work_dir / "overlay.mov").exists() and (work_dir / "base.mp4").exists():
            return "composite"
        if (hf_dir / "index.html").exists() and (work_dir / "base.mp4").exists():
            return "graphics"
        if (work_dir / "assets" / "bgm" / "track.mp3").exists() and (work_dir / "base.mp4").exists():
            return "graphics"
        if (work_dir / "base.mp4").exists():
            return "bgm_broll"
        if (work_dir / "edl.json").exists():
            return "base_cut"
        if (work_dir / "selected_script.json").exists():
            return "edl"
        if (work_dir / "script_options.json").exists():
            return "edl"
        if (work_dir / "words.json").exists() and (work_dir / "transcript_all.txt").exists():
            return "script"
        return "ingest"

    # ------------------------------------------------------------------------
    # EXECUTION CONTROLLER
    # ------------------------------------------------------------------------

    def start_job(
        self,
        job_id: str,
        from_stage: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Launch background asynchronous execution of a job from a given stage."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        # Cancel any active running task for this job
        if job_id in self._active_tasks and not self._active_tasks[job_id].done():
            self._cancel_events[job_id].set()
            self._active_tasks[job_id].cancel()

        cancel_event = asyncio.Event()
        self._cancel_events[job_id] = cancel_event

        task = asyncio.create_task(self._run_job_loop(job_id, from_stage, force, cancel_event))
        self._active_tasks[job_id] = task

        job["status"] = "running"
        job["error"] = None

        reset_from: str | None = None
        if from_stage and from_stage in STAGE_ORDER:
            reset_from = from_stage
        elif force:
            reset_from = STAGE_ORDER[0]

        if reset_from:
            reset_idx = STAGE_ORDER.index(reset_from)
            for s in STAGE_ORDER[reset_idx:]:
                if s in job.get("stages", {}):
                    job["stages"][s] = {
                        "status": "pending",
                        "started_at": None,
                        "completed_at": None,
                        "error": None,
                        "artifacts": {},
                    }

        self._save_job(job)
        self.publish(job_id, "job_started", {"from_stage": from_stage})
        return job

    def cancel_job(self, job_id: str) -> bool:
        """Signal cancellation for an active running job."""
        if job_id in self._cancel_events:
            self._cancel_events[job_id].set()
        task = self._active_tasks.get(job_id)
        if task and not task.done():
            task.cancel()
        job = self.get_job(job_id)
        if job:
            job["status"] = "cancelled"
            self._save_job(job)
            self.publish(job_id, "job_cancelled", {})
            return True
        return False

    @staticmethod
    def _load_clip_summary(work_dir: Path) -> str:
        """Format concise per-clip duration and resolution metadata for prompts."""
        clip_map_path = work_dir / "clip_map.json"
        if not clip_map_path.exists():
            return ""
        try:
            clip_map = json.loads(clip_map_path.read_text())
        except Exception:
            return ""
        if not isinstance(clip_map, dict) or not clip_map:
            return ""
        lines = []
        for cid, meta in clip_map.items():
            if not isinstance(meta, dict):
                continue
            stem = meta.get("stem") or cid
            dur = meta.get("duration_s")
            w, h = meta.get("width"), meta.get("height")
            dur_str = f"{dur}s" if dur is not None else "?"
            res_str = f"{w}x{h}" if w and h else "?"
            lines.append(f"- {cid} ({stem}): {dur_str}, {res_str}")
        return "\n".join(lines)

    async def _run_job_loop(
        self,
        job_id: str,
        from_stage: str | None,
        force: bool,
        cancel_event: asyncio.Event,
    ):
        """Main asynchronous state machine loop executing stages 1 through 8."""
        job_dir = self.jobs_root / job_id
        work_dir = job_dir / "work"

        def _log(msg: str):
            self.publish(job_id, "log", {"message": msg})

        job = self.get_job(job_id)
        if not job:
            return

        preset = get_preset(job.get("preset_id"))

        # Determine starting stage
        start_stage = from_stage
        if not start_stage and not force:
            start_stage = self._determine_resume_stage(job_dir)
        elif not start_stage:
            start_stage = "ingest"

        start_idx = STAGE_ORDER.index(start_stage) if start_stage in STAGE_ORDER else 0

        _log(f"Starting pipeline from stage '{STAGE_ORDER[start_idx]}' ({start_idx + 1}/8)...")

        for stage_idx in range(start_idx, len(STAGE_ORDER)):
            if cancel_event.is_set():
                break

            stage = STAGE_ORDER[stage_idx]
            job = self.get_job(job_id)
            if not job:
                break

            job["current_stage"] = stage
            job["error"] = None
            job["stages"][stage]["error"] = None
            job["stages"][stage]["status"] = "running"
            job["stages"][stage]["started_at"] = datetime.now(timezone.utc).isoformat()
            self._save_job(job)
            self.publish(job_id, "stage_started", {"stage": stage, "stage_index": stage_idx})

            cfg_models = job.get("config", {}).get("models", {})
            creative_model = validate_model_id(cfg_models.get("creative_model"))
            mechanical_model = validate_model_id(cfg_models.get("mechanical_model"))
            vision_model = validate_model_id(cfg_models.get("vision_model") or mechanical_model)

            try:
                # -------------------------------------------------------------
                # STAGE 1: INGEST & TRANSCRIBE
                # -------------------------------------------------------------
                if stage == "ingest":
                    clips = job.get("selected_clips", [])
                    if not clips:
                        # Auto-discover from default raw clip directory if none selected
                        from .presets import DEFAULT_CLIP_ROOTS
                        for root in DEFAULT_CLIP_ROOTS:
                            rpath = Path(root)
                            if rpath.exists():
                                clips = [str(f) for f in rpath.glob("*.mp4")]
                                if clips:
                                    break
                    res = await run_ingest_transcribe(
                        job_dir,
                        selected_clip_paths=clips,
                        whisper_model=job.get("whisper_model", "small.en"),
                        cancel_event=cancel_event,
                        log_cb=_log,
                    )
                    job["stages"][stage]["artifacts"] = res

                # -------------------------------------------------------------
                # STAGE 2: SCRIPT GENERATION & CRITIC PASS
                # -------------------------------------------------------------
                elif stage == "script":
                    transcript_text = (work_dir / "transcript_all.txt").read_text()
                    guide_summary = GUIDE_MD_PATH.read_text() if GUIDE_MD_PATH.exists() else None
                    clip_summary = self._load_clip_summary(work_dir)

                    _log(f"[SCRIPT] Generating 3 script options via AGY ({creative_model})...")
                    gen_prompt = build_script_generation_prompt(
                        transcript_text,
                        guide_summary,
                        preset,
                        story_prompt=job.get("story_prompt", ""),
                        clip_summary=clip_summary,
                    )
                    options_res = await run_agy_json(
                        gen_prompt,
                        schema=SCRIPT_OPTIONS_SCHEMA,
                        model=creative_model,
                        cancel_event=cancel_event,
                        log_cb=_log,
                    )
                    raw_options = options_res.get("options", [])

                    _log(f"[SCRIPT] Conducting adversarial cold-viewer critic pass via AGY ({creative_model})...")
                    critic_prompt = build_script_critic_prompt(raw_options, transcript_text, preset)
                    critic_res = await run_agy_json(
                        critic_prompt,
                        schema=CRITIC_SCHEMA,
                        model=creative_model,
                        cancel_event=cancel_event,
                        log_cb=_log,
                    )

                    final_options = critic_res.get("revised_options", raw_options)
                    (work_dir / "script_options.json").write_text(json.dumps(final_options, indent=2))

                    # Default selection to first option
                    if final_options:
                        (work_dir / "selected_script.json").write_text(json.dumps(final_options[0], indent=2))

                    job["stages"][stage]["artifacts"] = {
                        "script_options_file": str(work_dir / "script_options.json"),
                        "selected_script_file": str(work_dir / "selected_script.json"),
                        "options_count": len(final_options),
                        "critic_passes": critic_res.get("passes", True),
                    }

                    # GATE A: Script Approval
                    if job.get("pause_at_script", True):
                        job["status"] = "awaiting_script_approval"
                        job["stages"][stage]["status"] = "completed"
                        job["stages"][stage]["completed_at"] = datetime.now(timezone.utc).isoformat()
                        self._save_job(job)
                        self.publish(job_id, "awaiting_approval", {"gate": "A", "stage": "script"})
                        _log("Gate A reached: Pausing for script concept selection/approval.")
                        return

                # -------------------------------------------------------------
                # STAGE 3: EDL CREATION & OVERLAP VALIDATION
                # -------------------------------------------------------------
                elif stage == "edl":
                    selected_script_file = work_dir / "selected_script.json"
                    if not selected_script_file.exists():
                        raise FileNotFoundError("selected_script.json missing. Gate A approval required.")

                    selected_script = json.loads(selected_script_file.read_text())
                    words_by_clip = json.loads((work_dir / "words.json").read_text())
                    words_summary = format_words_for_edl(words_by_clip)
                    transcript_text = (work_dir / "transcript_all.txt").read_text()
                    clip_summary = self._load_clip_summary(work_dir)

                    _log(f"[EDL] Generating Edit Decision List (EDL) via AGY ({mechanical_model})...")
                    edl_prompt = build_edl_prompt(
                        selected_script,
                        words_summary,
                        transcript_text,
                        preset,
                        clip_summary=clip_summary,
                    )
                    raw_edl = await run_agy_json(
                        edl_prompt,
                        schema=EDL_SCHEMA,
                        model=mechanical_model,
                        cancel_event=cancel_event,
                        log_cb=_log,
                    )

                    clean_edl = validate_and_fix_edl_overlaps(raw_edl, words_by_clip)
                    (work_dir / "edl.json").write_text(json.dumps(clean_edl, indent=2))

                    job["stages"][stage]["artifacts"] = {
                        "edl_file": str(work_dir / "edl.json"),
                        "parts_count": len(clean_edl.get("parts", [])),
                    }

                # -------------------------------------------------------------
                # STAGE 4: BASE CUT & QA GATE
                # -------------------------------------------------------------
                elif stage == "base_cut":
                    res = await run_build_and_qa(
                        job_dir,
                        preset=preset,
                        cancel_event=cancel_event,
                        log_cb=_log,
                    )
                    job["stages"][stage]["artifacts"] = res

                # -------------------------------------------------------------
                # STAGE 5: BGM & B-ROLL
                # -------------------------------------------------------------
                elif stage == "bgm_broll":
                    res = await run_bgm_and_broll(
                        job_dir,
                        preset=preset,
                        broll_provider=job.get("broll_provider", "local"),
                        cancel_event=cancel_event,
                        log_cb=_log,
                    )
                    job["stages"][stage]["artifacts"] = res

                    # GATE B: Audio / BGM Approval
                    if job.get("pause_at_audio", True):
                        job["status"] = "awaiting_audio_approval"
                        job["stages"][stage]["status"] = "completed"
                        job["stages"][stage]["completed_at"] = datetime.now(timezone.utc).isoformat()
                        self._save_job(job)
                        self.publish(job_id, "awaiting_approval", {"gate": "B", "stage": "bgm_broll"})
                        _log("Gate B reached: Pausing for audio / BGM review and approval.")
                        return

                # -------------------------------------------------------------
                # STAGE 6: HYPERFRAMES MOTION GRAPHICS
                # -------------------------------------------------------------
                elif stage == "graphics":
                    res = await run_motion_graphics(
                        job_dir,
                        preset=preset,
                        creative_model=creative_model,
                        vision_model=vision_model,
                        mechanical_model=mechanical_model,
                        cancel_event=cancel_event,
                        log_cb=_log,
                    )
                    job["stages"][stage]["artifacts"] = res

                # -------------------------------------------------------------
                # STAGE 7: COMPOSITE & MASTERING
                # -------------------------------------------------------------
                elif stage == "composite":
                    res = await run_composite(
                        job_dir,
                        preset=preset,
                        cancel_event=cancel_event,
                        log_cb=_log,
                    )
                    job["stages"][stage]["artifacts"] = res

                # -------------------------------------------------------------
                # STAGE 8: DELIVERABLES (COVERS & DESCRIPTIONS)
                # -------------------------------------------------------------
                elif stage == "deliverables":
                    res = await run_deliverables(
                        job_dir,
                        preset=preset,
                        vision_model=vision_model,
                        mechanical_model=mechanical_model,
                        cancel_event=cancel_event,
                        log_cb=_log,
                    )
                    job["stages"][stage]["artifacts"] = res

                # Stage Succeeded
                job = self.get_job(job_id)
                if not job:
                    break
                job["error"] = None
                job["stages"][stage]["error"] = None
                job["stages"][stage]["status"] = "completed"
                job["stages"][stage]["completed_at"] = datetime.now(timezone.utc).isoformat()
                self._save_job(job)
                self.publish(job_id, "stage_completed", {"stage": stage, "artifacts": job["stages"][stage]["artifacts"]})

            except asyncio.CancelledError:
                job = self.get_job(job_id)
                if job:
                    job["stages"][stage]["status"] = "cancelled"
                    job["status"] = "cancelled"
                    self._save_job(job)
                _log(f"Stage '{stage}' cancelled.")
                return

            except Exception as e:
                job = self.get_job(job_id)
                if job:
                    job["stages"][stage]["status"] = "failed"
                    job["stages"][stage]["error"] = str(e)
                    job["status"] = "failed"
                    job["error"] = str(e)
                    self._save_job(job)
                self.publish(job_id, "stage_failed", {"stage": stage, "error": str(e)})
                _log(f"[ERROR] Stage '{stage}' failed: {e}")
                return

        # All Stages Complete
        job = self.get_job(job_id)
        if job and job.get("status") == "running":
            job["status"] = "completed"
            self._save_job(job)
            self.publish(job_id, "job_completed", {"job_id": job_id})
            _log("All pipeline stages completed successfully!")

    # ------------------------------------------------------------------------
    # GATE APPROVALS
    # ------------------------------------------------------------------------

    def approve_gate_a(
        self,
        job_id: str,
        selected_option_id: str,
        edited_script_text: str | None = None,
    ) -> dict[str, Any]:
        """Approve Gate A: Select script option (A/B/C) with optional text edits, then resume from EDL stage."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        work_dir = self.jobs_root / job_id / "work"
        options_file = work_dir / "script_options.json"
        if not options_file.exists():
            raise FileNotFoundError("script_options.json missing.")

        options = json.loads(options_file.read_text())
        matched = next((opt for opt in options if opt.get("id") == selected_option_id), None)
        if not matched and options:
            matched = options[0]

        if not matched:
            raise ValueError(f"Option ID '{selected_option_id}' not found.")

        chosen = dict(matched)
        if edited_script_text:
            chosen["script_text"] = edited_script_text.strip()

        (work_dir / "selected_script.json").write_text(json.dumps(chosen, indent=2))

        # Clear pause flag and resume from EDL
        job["pause_at_script"] = False
        self._save_job(job)
        return self.start_job(job_id, from_stage="edl")

    def approve_gate_b(
        self,
        job_id: str,
        selected_bgm: str | None = None,
    ) -> dict[str, Any]:
        """Approve Gate B: Select or confirm BGM track, then resume from Graphics stage."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        work_dir = self.jobs_root / job_id / "work"
        if selected_bgm:
            cand = work_dir / "assets" / "bgm" / "candidates" / selected_bgm
            active = work_dir / "assets" / "bgm" / "track.mp3"
            if cand.exists():
                if active.is_symlink() or active.exists():
                    active.unlink()
                active.symlink_to(cand)

        # Clear pause flag and resume from Graphics
        job["pause_at_audio"] = False
        self._save_job(job)
        return self.start_job(job_id, from_stage="graphics")

    async def recomposite_from_studio(self, job_id: str) -> dict[str, Any]:
        """Recomposite output directly from live Studio edits without restarting the full pipeline."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        job_dir = self.jobs_root / job_id

        def _log(msg: str):
            self.publish(job_id, "log", {"message": msg})

        res = await recomposite_studio(job_dir, log_cb=_log)
        self.publish(job_id, "recomposite_completed", res)
        return res

    async def shutdown(self):
        """Cancel all running jobs and stop all active studio preview servers."""
        for jid, event in self._cancel_events.items():
            event.set()
        for jid, task in self._active_tasks.items():
            if not task.done():
                task.cancel()
        await self.studio_manager.stop_all()
