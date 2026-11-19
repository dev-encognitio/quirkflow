# /// script
# requires-python = ">=3.11"
# dependencies = ["fastapi>=0.115", "uvicorn>=0.32", "python-multipart>=0.0.12", "pillow", "numpy"]
# ///

import os
import sys
import json
import uuid
import shutil
import asyncio
import logging
import inspect
import mimetypes
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Query, status
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure both studio-pipeline and its parent workspace are in sys.path
PIPELINE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PIPELINE_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("studio.server")

# -----------------------------------------------------------------------------
# Direct Imports from Clean Pipeline Modules
# -----------------------------------------------------------------------------

try:
    from .presets import (
        AGY_MODELS,
        DEFAULT_CLIP_ROOTS,
        DEFAULT_MODEL_SETTINGS,
        JOBS_ROOT,
        PIPELINE_ROOT,
        PRESETS,
        STAGE_ORDER,
        get_preset,
        validate_model_id,
    )
    from .adapters import (
        check_tool_health,
        get_or_create_thumbnail,
        list_clips,
        probe_media,
    )
    from .orchestrator import JobManager, StudioManager
except ImportError:
    from presets import (
        AGY_MODELS,
        DEFAULT_CLIP_ROOTS,
        DEFAULT_MODEL_SETTINGS,
        JOBS_ROOT,
        PIPELINE_ROOT,
        PRESETS,
        STAGE_ORDER,
        get_preset,
        validate_model_id,
    )
    from adapters import (
        check_tool_health,
        get_or_create_thumbnail,
        list_clips,
        probe_media,
    )
    from orchestrator import JobManager, StudioManager

_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager(jobs_root=JOBS_ROOT)
    return _job_manager


def get_studio_manager() -> StudioManager:
    return get_job_manager().studio_manager


# -----------------------------------------------------------------------------
# Enriched Job Helper
# -----------------------------------------------------------------------------

def enrich_job(job: dict) -> dict:
    if not job:
        return job
    job_id = job.get("id") or job.get("job_id", "")
    if not job_id:
        return job
    job_dir = JOBS_ROOT / job_id

    # 1. Configuration object normalization
    cfg = job.get("config") or {}
    user_models = cfg.get("models") or {}
    cfg_models = {
        slot: validate_model_id(user_models.get(slot, default_val))
        for slot, default_val in DEFAULT_MODEL_SETTINGS.items()
    }
    if not cfg.get("vision_model_explicit"):
        cfg_models["vision_model"] = cfg_models["mechanical_model"]
    def _norm_mode(val: Any) -> str:
        s = str(val or "auto").strip().lower()
        return s if s in ("auto", "on", "off") else "auto"

    job["config"] = {
        "preset_id": cfg.get("preset_id") or job.get("preset_id", "story_vertical"),
        "story_prompt": cfg.get("story_prompt") if "story_prompt" in cfg else job.get("story_prompt", ""),
        "gate_a": cfg.get("gate_a", job.get("pause_at_script", False)),
        "gate_b": cfg.get("gate_b", job.get("pause_at_audio", False)),
        "whisper_model": cfg.get("whisper_model") or job.get("whisper_model", "small.en"),
        "ai_broll": cfg.get("ai_broll") or job.get("broll_provider", "off"),
        "jl_cuts": _norm_mode(cfg.get("jl_cuts")),
        "room_tone": _norm_mode(cfg.get("room_tone")),
        "establishing_shot": _norm_mode(cfg.get("establishing_shot")),
        "models": cfg_models,
        "vision_model_explicit": bool(cfg.get("vision_model_explicit")),
    }

    # 2. Base video URL
    if (job_dir / "work" / "base.mp4").exists():
        job["base_video_url"] = f"/api/jobs/{job_id}/files/work/base.mp4"
    elif (job_dir / "base.mp4").exists():
        job["base_video_url"] = f"/api/jobs/{job_id}/files/base.mp4"
    else:
        job.setdefault("base_video_url", None)

    # 3. Final video URL
    if (job_dir / "final" / "final.mp4").exists():
        job["final_video_url"] = f"/api/jobs/{job_id}/files/final/final.mp4"
    elif (job_dir / "final.mp4").exists():
        job["final_video_url"] = f"/api/jobs/{job_id}/files/final.mp4"
    else:
        job.setdefault("final_video_url", None)

    # 4. Covers
    covers = []
    for folder in [job_dir / "cover", job_dir / "final", job_dir / "work" / "cover"]:
        if folder.exists():
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and "cover" in p.name.lower():
                    rel = p.relative_to(job_dir)
                    label = "Cover A" if "cover_a" in p.name.lower() or "cover-a" in p.name.lower() else (
                        "Cover B" if "cover_b" in p.name.lower() or "cover-b" in p.name.lower() else p.stem.replace("_", " ").title()
                    )
                    covers.append({
                        "id": p.stem,
                        "label": label,
                        "filename": p.name,
                        "url": f"/api/jobs/{job_id}/files/{rel}",
                    })
    job["covers"] = covers

    # 5. Descriptions data
    desc_json_file = job_dir / "final" / "descriptions.json"
    desc_txt_file = job_dir / "final" / "descriptions.txt"
    if desc_json_file.exists():
        try:
            with open(desc_json_file, "r", encoding="utf-8") as f:
                job["descriptions_data"] = json.load(f)
        except Exception:
            pass
    elif desc_txt_file.exists():
        try:
            raw_text = desc_txt_file.read_text(encoding="utf-8")
            job["descriptions_raw"] = raw_text
            options = {}
            for line in raw_text.splitlines():
                if "Hook" in line: options["hook"] = line
                elif "Teaser" in line: options["teaser"] = line
                elif "Confessional" in line: options["confessional"] = line
                elif "Debate" in line: options["debate"] = line
                elif "Minimal" in line: options["minimal"] = line
            job["descriptions_data"] = options if options else {"hook": raw_text}
        except Exception:
            pass
    if "descriptions_data" not in job:
        job["descriptions_data"] = None

    # 6. BGM candidates
    bgm_list = []
    seen_bgm_names: set[str] = set()
    seen_bgm_paths: set[str] = set()
    for folder in [job_dir / "work" / "assets" / "bgm" / "candidates", job_dir / "work" / "assets" / "bgm", job_dir / "assets" / "bgm"]:
        if folder.exists():
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac", ".ogg"}:
                    resolved = str(p.resolve())
                    if p.name in seen_bgm_names or resolved in seen_bgm_paths:
                        continue
                    seen_bgm_names.add(p.name)
                    seen_bgm_paths.add(resolved)
                    rel = p.relative_to(job_dir)
                    bgm_list.append({
                        "id": p.stem,
                        "filename": p.name,
                        "title": p.stem.replace("_", " ").title(),
                        "url": f"/api/jobs/{job_id}/files/{rel}",
                    })
    job["bgm_candidates"] = bgm_list

    # 7. Script options (from Stage 2 Script)
    script_opt_file = job_dir / "work" / "script_options.json"
    if script_opt_file.exists():
        try:
            with open(script_opt_file, "r", encoding="utf-8") as f:
                job["script_options"] = json.load(f)
        except Exception:
            pass

    # 8. Real QA metrics: Read and merge work/qa_report.json and final/qa_report.json cleanly
    qa_merged = dict(job.get("qa") or {})
    for report_file in [
        job_dir / "work" / "qa_report.json",
        job_dir / "final" / "qa_report.json",
        job_dir / "work" / "qa.json",
        job_dir / "final" / "qa.json",
    ]:
        if report_file.exists():
            try:
                data = json.loads(report_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    qa_merged.update(data)
            except Exception:
                pass

    # Normalize real field keys without fabricating fake values
    if "lufs" in qa_merged and "integrated_loudness_lufs" not in qa_merged:
        try:
            qa_merged["integrated_loudness_lufs"] = float(qa_merged["lufs"])
        except (ValueError, TypeError):
            pass
    if "script_match" in qa_merged and "script_match_pct" not in qa_merged:
        try:
            qa_merged["script_match_pct"] = float(qa_merged["script_match"])
        except (ValueError, TypeError):
            pass
    if "cuts_count" in qa_merged and "parts_count" not in qa_merged:
        qa_merged["parts_count"] = qa_merged["cuts_count"]

    job["qa"] = qa_merged

    # 9. Studio status
    sm = get_studio_manager()
    studio_info = sm.get_studio(job_id)
    if studio_info:
        studio_info["open"] = studio_info.get("status") == "running"
        job["studio"] = studio_info
    else:
        job["studio"] = {"open": False}

    # 10. Co-Pilot summary
    jm = get_job_manager()
    copilot_hist = jm.get_copilot_history(job_id)
    job["copilot"] = {
        "conversation_id": copilot_hist.get("conversation_id"),
        "messages": copilot_hist.get("messages", []),
        "busy": jm.is_copilot_busy(job_id),
    }

    return job


# -----------------------------------------------------------------------------
# FastAPI App Lifecycle & Setup
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure upload directory exists (thumbnails live in /tmp/studio-pipeline-thumbs/)
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    (JOBS_ROOT / "_uploads").mkdir(parents=True, exist_ok=True)
    logger.info(f"Studio Server initialized. JOBS_ROOT: {JOBS_ROOT}")
    yield
    # Shutdown
    jm = get_job_manager()
    await jm.shutdown()
    logger.info("Studio Server shut down cleanly.")


app = FastAPI(title="Autonomous Video Studio Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Request Schemas
# -----------------------------------------------------------------------------

class CreateJobRequest(BaseModel):
    config: Dict[str, Any] = Field(default_factory=dict)
    clip_paths: List[str]
    start_immediately: bool = True


class StartJobRequest(BaseModel):
    from_stage: Optional[str] = None
    force: bool = False


class ApproveScriptRequest(BaseModel):
    option_id: str
    script_text: Optional[str] = None


class ApproveAudioRequest(BaseModel):
    bgm_file: Optional[str] = None


class UpdateModelsRequest(BaseModel):
    creative_model: Optional[str] = None
    mechanical_model: Optional[str] = None
    vision_model: Optional[str] = None
    copilot_model: Optional[str] = None


class CopilotMessageRequest(BaseModel):
    message: str
    model: Optional[str] = None
    stage_hint: Optional[str] = None


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    tools = check_tool_health()
    return {
        "ok": True,
        "tools": tools,
        "default_roots": DEFAULT_CLIP_ROOTS,
    }


@app.get("/api/presets")
async def get_presets():
    return {
        "presets": list(PRESETS.values()),
        "stages": STAGE_ORDER,
        "default_roots": DEFAULT_CLIP_ROOTS,
        "agy_models": AGY_MODELS,
        "default_models": DEFAULT_MODEL_SETTINGS,
    }


@app.get("/api/clips")
async def get_clips(folder: str = Query(..., description="Folder path containing video clips")):
    clean_folder = folder.strip()
    if not clean_folder:
        raise HTTPException(status_code=400, detail="Folder path cannot be empty")
    
    res = list_clips(clean_folder)
    if not res.get("exists", False):
        raise HTTPException(status_code=400, detail=f"Folder does not exist or is not a directory: {clean_folder}")
    return res


@app.get("/api/thumb")
async def get_thumbnail(path: str = Query(..., description="Absolute path to video file")):
    if not path or not path.strip():
        raise HTTPException(status_code=400, detail="Path parameter is required")
    v_path = Path(path.strip()).resolve()
    if not v_path.exists() or not v_path.is_file():
        raise HTTPException(status_code=404, detail=f"Video file not found: {path}")
    if v_path.suffix.lower() not in {".mp4", ".mov", ".webm", ".mkv", ".m4v"}:
        raise HTTPException(status_code=400, detail="File must be a supported video format")

    thumb_path = get_or_create_thumbnail(v_path)
    if thumb_path.exists() and thumb_path.stat().st_size > 0:
        return FileResponse(str(thumb_path), media_type="image/jpeg")
    raise HTTPException(status_code=500, detail="Failed to generate thumbnail for video")


@app.get("/api/thumbs/{filename}")
async def get_cached_thumbnail(filename: str):
    thumb_path = (Path("/tmp/studio-pipeline-thumbs") / filename).resolve()
    if thumb_path.exists() and thumb_path.is_file():
        return FileResponse(str(thumb_path), media_type="image/jpeg")
    raise HTTPException(status_code=404, detail=f"Thumbnail not found: {filename}")


@app.post("/api/uploads")
async def upload_clips(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    upload_dir = JOBS_ROOT / "_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for f in files:
        orig_name = f.filename or "upload.mp4"
        clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", Path(orig_name).name) or "upload.mp4"
        u8 = uuid.uuid4().hex[:8]
        saved_name = f"{u8}_{clean_name}"
        saved_path = upload_dir / saved_name

        try:
            with open(saved_path, "wb") as out:
                shutil.copyfileobj(f.file, out)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to write uploaded file {orig_name}: {str(e)}")

        meta = probe_media(saved_path)
        thumb_url = f"/api/thumb?path={saved_path.resolve()}"
        results.append({
            "name": orig_name,
            "path": str(saved_path.resolve()),
            "duration_s": meta.get("duration_s", 0.0),
            "width": meta.get("width", 0),
            "height": meta.get("height", 0),
            "thumb_url": thumb_url,
        })

    return results


@app.get("/api/jobs")
async def list_jobs():
    jm = get_job_manager()
    jobs = jm.list_jobs()
    return [enrich_job(j) for j in jobs]


@app.post("/api/jobs", status_code=status.HTTP_201_CREATED)
async def create_job(req: CreateJobRequest):
    if not req.clip_paths:
        raise HTTPException(status_code=400, detail="clip_paths cannot be empty")
    for cp in req.clip_paths:
        p = Path(cp)
        if not p.exists() or not p.is_file():
            raise HTTPException(status_code=400, detail=f"Clip file does not exist: {cp}")

    jm = get_job_manager()
    job = jm.create_job(
        config=req.config,
        clip_paths=req.clip_paths,
        start_immediately=req.start_immediately,
    )
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=enrich_job(job))


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    jm = get_job_manager()
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return enrich_job(job)


@app.post("/api/jobs/{job_id}/start")
async def start_job(job_id: str, req: Optional[StartJobRequest] = None):
    jm = get_job_manager()
    from_stage = req.from_stage if req else None
    force = req.force if req else False
    res = jm.start_job(job_id, from_stage=from_stage, force=force)
    if inspect.isawaitable(res):
        res = await res
    job = jm.get_job(job_id) or res
    return enrich_job(job)


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    jm = get_job_manager()
    res = jm.cancel_job(job_id)
    if inspect.isawaitable(res):
        res = await res
    job = jm.get_job(job_id) or {"id": job_id, "status": "cancelled"}
    return enrich_job(job)


@app.post("/api/jobs/{job_id}/approve-script")
async def approve_script(job_id: str, req: ApproveScriptRequest):
    jm = get_job_manager()
    res = jm.approve_gate_a(
        job_id=job_id,
        selected_option_id=req.option_id,
        edited_script_text=req.script_text,
    )
    if inspect.isawaitable(res):
        res = await res
    job = jm.get_job(job_id) or res
    return enrich_job(job)


@app.post("/api/jobs/{job_id}/approve-audio")
async def approve_audio(job_id: str, req: ApproveAudioRequest):
    jm = get_job_manager()
    res = jm.approve_gate_b(
        job_id=job_id,
        selected_bgm=req.bgm_file,
    )
    if inspect.isawaitable(res):
        res = await res
    job = jm.get_job(job_id) or res
    return enrich_job(job)


@app.post("/api/jobs/{job_id}/studio/open")
async def open_studio(job_id: str):
    jm = get_job_manager()
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    sm = get_studio_manager()
    hf_dir = JOBS_ROOT / job_id / "hf"
    res = sm.open_studio(job_id=job_id, hf_dir=hf_dir)
    if inspect.isawaitable(res):
        res = await res
    return res


@app.post("/api/jobs/{job_id}/studio/stop")
async def stop_studio(job_id: str):
    sm = get_studio_manager()
    res = sm.stop_studio(job_id=job_id)
    if inspect.isawaitable(res):
        res = await res
    return {"job_id": job_id, "stopped": bool(res)}


@app.post("/api/jobs/{job_id}/recomposite")
async def recomposite(job_id: str):
    jm = get_job_manager()
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    res = jm.recomposite_from_studio(job_id)
    if inspect.isawaitable(res):
        res = await res
    job = jm.get_job(job_id) or job
    return enrich_job(job)


@app.patch("/api/jobs/{job_id}/models")
async def update_job_models(job_id: str, req: UpdateModelsRequest):
    jm = get_job_manager()
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    patch = {k: v for k, v in req.model_dump().items() if v is not None}
    updated_models = jm.update_job_models(job_id, patch)
    job = jm.get_job(job_id)
    return {
        "ok": True,
        "models": updated_models,
        "job": enrich_job(job) if job else None,
    }


@app.get("/api/jobs/{job_id}/copilot")
async def get_job_copilot(job_id: str):
    jm = get_job_manager()
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    history = jm.get_copilot_history(job_id)
    return {
        "conversation_id": history.get("conversation_id"),
        "messages": history.get("messages", []),
        "busy": jm.is_copilot_busy(job_id),
    }


@app.post("/api/jobs/{job_id}/copilot")
async def send_copilot_message(job_id: str, req: CopilotMessageRequest):
    jm = get_job_manager()
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if jm.is_copilot_busy(job_id):
        raise HTTPException(status_code=409, detail="Co-Pilot is currently processing another turn for this job")
    try:
        result = await jm.send_copilot_message(
            job_id,
            message=req.message.strip(),
            model=req.model,
            stage_hint=req.stage_hint,
        )
        if result.get("job"):
            result["job"] = enrich_job(result["job"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Co-Pilot turn error on job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Co-Pilot error: {str(e)}")


@app.post("/api/jobs/{job_id}/copilot/reset")
async def reset_copilot_session(job_id: str):
    jm = get_job_manager()
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    history = jm.reset_copilot_session(job_id)
    return {
        "ok": True,
        "conversation_id": history.get("conversation_id"),
        "messages": history.get("messages", []),
        "busy": jm.is_copilot_busy(job_id),
    }


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    jm = get_job_manager()
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    q = jm.subscribe(job_id)

    async def event_generator():
        try:
            # Send initial state event
            latest = jm.get_job(job_id) or job
            initial_payload = json.dumps({
                "event": "state",
                "type": "state",
                "job": enrich_job(latest),
            })
            yield f"data: {initial_payload}\n\n"

            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15.0)
                    if isinstance(event, dict):
                        ev_type = event.get("type") or event.get("event") or "update"
                        event["type"] = ev_type
                        event["event"] = ev_type

                        # Attach enriched job on state/stage/gate/completion events
                        STATE_EVENT_TYPES = {
                            "state", "job_started", "stage_started", "stage_completed",
                            "awaiting_approval", "stage_failed", "job_completed",
                            "recomposite_completed", "gate_approved", "job_cancelled",
                            "copilot_reply", "models_updated",
                        }
                        if ev_type in STATE_EVENT_TYPES or "job" in event:
                            current = jm.get_job(job_id)
                            if current:
                                event["job"] = enrich_job(current)

                        payload = json.dumps(event)
                    else:
                        payload = str(event)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            jm.unsubscribe(job_id, q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/jobs/{job_id}/files/{rel_path:path}")
async def get_job_file(job_id: str, rel_path: str):
    job_dir = (JOBS_ROOT / job_id).resolve()
    target_path = (job_dir / rel_path).resolve()

    # Path traversal protection
    try:
        if not target_path.is_relative_to(job_dir):
            raise HTTPException(status_code=403, detail="Forbidden: Path traversal detected")
    except AttributeError:
        if not str(target_path).startswith(str(job_dir)):
            raise HTTPException(status_code=403, detail="Forbidden: Path traversal detected")

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {rel_path}")

    mime_type, _ = mimetypes.guess_type(str(target_path))
    return FileResponse(str(target_path), media_type=mime_type or "application/octet-stream")


# -----------------------------------------------------------------------------
# Static Files & Frontend Routing
# -----------------------------------------------------------------------------

web_dir = PIPELINE_DIR / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")


@app.get("/")
async def root():
    index_file = web_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Autonomous Video Studio Pipeline Server Running"}


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("STUDIO_HOST", "127.0.0.1")
    port = int(os.environ.get("STUDIO_PORT", "8400"))
    logger.info(f"Starting server on http://{host}:{port}")
    uvicorn.run("server:app", host=host, port=port, reload=False)
