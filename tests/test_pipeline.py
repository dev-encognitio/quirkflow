"""Comprehensive QA test suite for the autonomous video studio pipeline.

Covers:
1. presets.py configuration & agent_prompts.validate_and_fix_edl_overlaps
2. adapters media analysis (list_clips, probe_media, thumbnail) & unmounted path safety
3. JobManager lifecycle (defaults, config persistence, Gate A / B approvals, cancellation)
4. server.py FastAPI routes via TestClient (health, presets, clips, thumb, jobs, uploads, traversal guard)
5. HyperFrames composition check & StudioManager preview server lifecycle
"""

import os
import sys
import json
import shutil
import asyncio
import tempfile
import subprocess
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

# Ensure pipeline root is in sys.path
PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

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
from agent_prompts import (
    DESCRIPTIONS_SCHEMA,
    SCRIPT_OPTIONS_SCHEMA,
    build_script_generation_prompt,
    validate_and_fix_edl_overlaps,
)
import adapters
from adapters import (
    check_tool_health,
    get_or_create_thumbnail,
    list_clips,
    probe_media,
    _generate_landscape_cover,
    _extract_json_payload,
    run_agy_copilot_turn,
)
from orchestrator import JobManager, StudioManager
from server import app, enrich_job


# ============================================================================
# 1. PRESETS & AGENT PROMPTS / EDL VALIDATOR TESTS
# ============================================================================

class TestPresetsAndEdl:
    def test_stage_order_integrity(self):
        assert len(STAGE_ORDER) == 8
        expected = [
            "ingest",
            "script",
            "edl",
            "base_cut",
            "bgm_broll",
            "graphics",
            "composite",
            "deliverables",
        ]
        assert STAGE_ORDER == expected

    def test_presets_structure_and_dimensions(self):
        assert "story_vertical" in PRESETS
        assert "retention_3step" in PRESETS
        assert "landscape_explainer" in PRESETS

        vert = PRESETS["story_vertical"]
        assert vert["width"] == 1080 and vert["height"] == 1920
        assert vert["aspect_ratio"] == "9:16"

        ret = PRESETS["retention_3step"]
        assert ret["width"] == 1080 and ret["height"] == 1920
        assert ret["aspect_ratio"] == "9:16"

        land = PRESETS["landscape_explainer"]
        assert land["width"] == 1920 and land["height"] == 1080
        assert land["aspect_ratio"] == "16:9"

    def test_get_preset_resolver(self):
        assert get_preset("story_vertical")["id"] == "story_vertical"
        assert get_preset("retention_3step")["id"] == "retention_3step"
        assert get_preset("landscape_explainer")["id"] == "landscape_explainer"
        # Fallback for unknown preset id
        assert get_preset("unknown_custom_id")["id"] == "story_vertical"

    def test_script_generation_prompt_with_story_direction(self):
        prompt = build_script_generation_prompt(
            transcript_text="Test spoken words here",
            preset=get_preset("story_vertical"),
            story_prompt="Focus heavily on the camera anxiety turning point",
        )
        assert "Focus heavily on the camera anxiety turning point" in prompt
        assert "Option A, Option B, Option C" in prompt

    def test_descriptions_schema_keys(self):
        props = DESCRIPTIONS_SCHEMA["properties"]
        req = DESCRIPTIONS_SCHEMA["required"]
        for key in ["hook", "teaser", "confessional", "debate", "minimal", "recommendation"]:
            assert key in props
            assert key in req

    def test_validate_and_fix_edl_overlaps_clean(self):
        clean_edl = {
            "title": "Clean Test",
            "parts": [
                {"id": "p01", "clip": "c01", "range": [1.0, 5.0], "say": "first line", "hold": 0.0},
                {"id": "p02", "clip": "c02", "range": [2.0, 6.0], "say": "second line", "hold": 0.0},
            ],
        }
        res = validate_and_fix_edl_overlaps(clean_edl)
        assert len(res["parts"]) == 2
        assert res["parts"][0]["range"] == [1.0, 5.0]
        assert res["parts"][1]["range"] == [2.0, 6.0]

    def test_validate_and_fix_edl_overlaps_adjacent_collision(self):
        colliding_edl = {
            "title": "Overlap Bug Fix",
            "parts": [
                {"id": "p01", "clip": "c01", "range": [2.0, 7.0], "say": "he said singlet", "hold": 0.1},
                {"id": "p02", "clip": "c01", "range": [6.5, 11.0], "say": "let vest", "hold": 0.0},
            ],
        }
        fixed = validate_and_fix_edl_overlaps(colliding_edl)
        p1 = fixed["parts"][0]
        p2 = fixed["parts"][1]
        assert p1["range"][1] <= p2["range"][0]
        assert p1["hold"] == 0.0

    def test_validate_and_fix_edl_overlaps_with_word_timestamps(self):
        edl = {
            "title": "Word Split Test",
            "parts": [
                {"id": "p01", "clip": "c01", "range": [1.0, 5.0], "say": "word one", "hold": 0.0},
                {"id": "p02", "clip": "c01", "range": [4.0, 8.0], "say": "word two", "hold": 0.0},
            ],
        }
        words_by_clip = {
            "c01": [
                ["word", 1.2, 1.8],
                ["one", 2.0, 3.2],
                ["word", 4.5, 5.1],
                ["two", 5.2, 6.0],
            ]
        }
        fixed = validate_and_fix_edl_overlaps(edl, words_by_clip=words_by_clip)
        p1 = fixed["parts"][0]
        p2 = fixed["parts"][1]
        assert p1["range"][1] <= p2["range"][0]
        # Spoken words must remain enclosed by their repaired ranges (never split mid-word)
        assert p1["range"][0] <= 1.2 and 2.0 <= p1["range"][1]
        assert p2["range"][0] <= 4.5 and 5.2 <= p2["range"][1]

    def test_validate_and_fix_edl_overlaps_non_chronological_jumps(self):
        edl = {
            "title": "Non-chronological Narrative Order",
            "parts": [
                {"id": "p01", "clip": "c01", "range": [31.75, 32.59], "say": "pineapple on pizza", "hold": 0.05},
                {"id": "p02", "clip": "c01", "range": [47.19, 47.87], "say": "unmask yourself", "hold": 0.05},
                {"id": "p03", "clip": "c01", "range": [28.59, 31.25], "say": "something people debate", "hold": 0.05},
                {"id": "p04", "clip": "c01", "range": [131.0, 139.38], "say": "probably cannot change opinion", "hold": 0.0},
                {"id": "p05", "clip": "c01", "range": [67.26, 70.62], "say": "express our personalities", "hold": 0.1},
            ],
        }
        fixed = validate_and_fix_edl_overlaps(edl)
        expected = [
            ([31.75, 32.59], 0.05),
            ([47.19, 47.87], 0.05),
            ([28.59, 31.25], 0.05),
            ([131.0, 139.38], 0.0),
            ([67.26, 70.62], 0.1),
        ]
        for part, (exp_rng, exp_hold) in zip(fixed["parts"], expected):
            assert part["range"] == exp_rng
            assert part["hold"] == exp_hold

    def test_validate_and_fix_edl_overlaps_repairs_crushed_or_shifted_word_ranges(self):
        words_by_clip = {
            "c01": [
                ["something", 28.59, 29.01],
                ["where", 29.13, 29.51],
                ["people", 29.57, 29.83],
                ["unmask", 45.23, 45.9],
                ["yourself", 45.9, 46.97],
                ["unmask", 47.19, 47.59],
                ["yourself", 47.59, 47.87],
                ["allow", 67.26, 67.8],
                ["ourselves", 68.56, 69.2],
                ["personalities", 70.2, 70.62],
            ]
        }
        edl = {
            "title": "Repair Crushed Ranges",
            "parts": [
                # Nearest occurrence (47.19..47.87) should win over earlier (45.23..46.97)
                {"id": "p01", "clip": "c01", "range": [47.19, 47.29], "say": "unmask yourself", "hold": 0.0},
                # Crushed to trailing 0.10s at end
                {"id": "p02", "clip": "c01", "range": [31.15, 31.25], "say": "something where people", "hold": 0.05},
                # Slightly shifted past first word start
                {"id": "p03", "clip": "c01", "range": [67.35, 70.62], "say": "allow ourselves personalities", "hold": 0.0},
            ],
        }
        fixed = validate_and_fix_edl_overlaps(edl, words_by_clip=words_by_clip)
        assert fixed["parts"][0]["range"] == [47.19, 47.87]
        assert fixed["parts"][1]["range"] == [28.59, 29.83]
        assert fixed["parts"][2]["range"] == [67.26, 70.62]

    def test_validate_and_fix_edl_overlaps_hold_nested_and_reverse_partial(self):
        edl = {
            "title": "Edge Overlaps",
            "parts": [
                # Hold-only overlap: hold pushes 5.0 + 0.5 > 5.2, speech ranges disjoint
                {"id": "p01", "clip": "c01", "range": [1.0, 5.0], "say": "", "hold": 0.5},
                {"id": "p02", "clip": "c01", "range": [5.2, 8.0], "say": "", "hold": 0.0},
                # Nested overlap on c02
                {"id": "p03", "clip": "c02", "range": [2.0, 12.0], "say": "", "hold": 0.0},
                {"id": "p04", "clip": "c02", "range": [4.0, 6.0], "say": "", "hold": 0.0},
                # Reverse partial overlap on c03 ([10, 15] then [8, 11])
                {"id": "p05", "clip": "c03", "range": [10.0, 15.0], "say": "", "hold": 0.1},
                {"id": "p06", "clip": "c03", "range": [8.0, 11.0], "say": "", "hold": 0.0},
            ],
        }
        fixed = validate_and_fix_edl_overlaps(edl)
        p1, p2, p3, p4, p5, p6 = fixed["parts"]

        assert p1["range"] == [1.0, 5.0] and p1["hold"] == 0.0
        assert p2["range"] == [5.2, 8.0]

        assert p3["range"][0] < p3["range"][1] <= p4["range"][0] < p4["range"][1]
        assert p6["range"][0] < p6["range"][1] <= p5["range"][0] < p5["range"][1]
        assert p5["hold"] == 0.0

    def test_format_words_for_edl_preserves_all_clips_past_6000_chars(self):
        from agent_prompts import format_words_for_edl

        words_by_clip = {
            "c01": [[f"alpha{i}", round(i * 0.5, 2), round(i * 0.5 + 0.3, 2)] for i in range(450)],
            "c02": [["beta", 0.25, 0.75], ["second", 0.8, 1.2]],
            "c03": [["gamma", 1.0, 1.5], ["third", 1.6, 2.0]],
        }
        full_formatted = format_words_for_edl(words_by_clip)
        assert len(full_formatted) > 6000
        parsed = json.loads(full_formatted)
        assert set(parsed.keys()) == {"c01", "c02", "c03"}
        assert len(parsed["c01"]) == 450
        assert parsed["c03"][-1][0] == "third"

        # When forced under a tight cap, output must remain valid JSON with all clips present
        capped = format_words_for_edl(words_by_clip, max_chars=400)
        parsed_capped = json.loads(capped)
        assert set(parsed_capped.keys()) == {"c01", "c02", "c03"}


# ============================================================================
# 2. ADAPTERS MEDIA PROBING & THUMBNAILS TESTS
# ============================================================================

class TestAdaptersMediaAndTools:
    def test_tool_health_check(self):
        tools = check_tool_health()
        assert isinstance(tools, dict)
        assert "claude" not in tools
        for key in ["agy", "hyperframes", "ffmpeg", "ffprobe", "uv", "comfy"]:
            assert key in tools
            assert "available" in tools[key]
            assert "path" in tools[key]

    def test_raw_clips_listing(self):
        raw_clips_dir = "/home/sandbox/ContentCreation/raw_clips"
        res = list_clips(raw_clips_dir)
        assert res["exists"] is True
        assert res["path"] == str(Path(raw_clips_dir).resolve())
        clip_names = [c["name"] for c in res["clips"]]
        assert "PXL_20261110_074640398.mp4" in clip_names

        # Find target clip
        clip = next(c for c in res["clips"] if c["name"] == "PXL_20261110_074640398.mp4")
        assert clip["duration_s"] > 0
        assert clip["width"] > 0
        assert clip["height"] > 0
        assert "/api/thumb?path=" in clip["thumb_url"]

    def test_probe_media_on_real_clip(self):
        clip_path = "/home/sandbox/ContentCreation/raw_clips/PXL_20261110_074640398.mp4"
        meta = probe_media(clip_path)
        assert meta["duration_s"] > 5.0
        assert meta["width"] == 1080 or meta["width"] == 1920
        assert meta["height"] == 1920 or meta["height"] == 1080
        assert meta["has_audio"] is True
        assert meta["fps"] > 0

    def test_get_or_create_thumbnail_generation(self):
        clip_path = "/home/sandbox/ContentCreation/raw_clips/PXL_20261110_074640398.mp4"
        thumb_path = get_or_create_thumbnail(clip_path)
        assert thumb_path.exists()
        assert thumb_path.stat().st_size > 500
        assert str(thumb_path).startswith("/tmp/studio-pipeline-thumbs")

    def test_unmounted_or_nonexistent_path_handling(self):
        unmounted = "/mnt/99/never_mounted_drive/videos"
        res = list_clips(unmounted)
        assert res["exists"] is False
        assert res["clips"] == []

        nonexistent_file = "/tmp/does_not_exist_987654.mp4"
        meta = probe_media(nonexistent_file)
        assert meta["duration_s"] == 0.0
        assert meta["width"] == 0
        assert meta["height"] == 0

    def test_landscape_cover_generation_pillow(self):
        with tempfile.TemporaryDirectory() as td:
            out_base = Path(td) / "cover_landscape_test"
            _generate_landscape_cover(
                frame_path=None,  # Tests synthetic background path
                out_base=out_base,
                title_text="Case Study: Algorithmic Retention Dynamics",
            )
            jpg_file = Path(f"{out_base}.jpg")
            png_file = Path(f"{out_base}.png")
            assert jpg_file.exists() and jpg_file.stat().st_size > 1000
            assert png_file.exists() and png_file.stat().st_size > 1000

            # Verify dimensions via Pillow
            from PIL import Image
            with Image.open(jpg_file) as img:
                assert img.size == (1920, 1080)


# ============================================================================
# 3. JOBMANAGER LIFECYCLE & CONFIG PERSISTENCE TESTS
# ============================================================================

class TestJobManagerAndLifecycle:
    def test_job_defaults_autonomous(self):
        with tempfile.TemporaryDirectory() as td:
            jm = JobManager(jobs_root=Path(td))
            job = jm.create_job()
            assert job["status"] == "idle"
            assert job["preset_id"] == "story_vertical"
            # Autonomous by default!
            assert job["pause_at_script"] is False
            assert job["pause_at_audio"] is False
            assert job["whisper_model"] == "small.en"
            assert job["broll_provider"] == "off"

            # Check job.json persistence
            loaded = jm.get_job(job["id"])
            assert loaded is not None
            assert loaded["config"]["gate_a"] is False
            assert loaded["config"]["gate_b"] is False
            assert loaded["config"]["story_prompt"] == ""
            assert loaded["config"]["ai_broll"] == "off"

    def test_config_persistence_and_normalization(self):
        with tempfile.TemporaryDirectory() as td:
            jm = JobManager(jobs_root=Path(td))
            custom_cfg = {
                "preset_id": "retention_3step",
                "story_prompt": "Deconstruct viral retention hooks",
                "gate_a": True,
                "gate_b": True,
                "whisper_model": "medium.en",
                "ai_broll": "comfy",  # must normalize to comfyui
            }
            job = jm.create_job(config=custom_cfg, clip_paths=["/tmp/clip1.mp4"])
            assert job["config"]["preset_id"] == "retention_3step"
            assert job["config"]["story_prompt"] == "Deconstruct viral retention hooks"
            assert job["config"]["gate_a"] is True
            assert job["config"]["gate_b"] is True
            assert job["config"]["whisper_model"] == "medium.en"
            assert job["config"]["ai_broll"] == "comfyui"  # normalized!
            assert job["pause_at_script"] is True
            assert job["pause_at_audio"] is True
            assert job["selected_clips"] == ["/tmp/clip1.mp4"]

            # Verify reloaded
            loaded = jm.get_job(job["id"])
            assert loaded["config"]["ai_broll"] == "comfyui"

    def test_gate_a_script_approval(self):
        with tempfile.TemporaryDirectory() as td:
            jm = JobManager(jobs_root=Path(td))
            job = jm.create_job(config={"gate_a": True})
            jid = job["id"]
            job_dir = Path(td) / jid
            work_dir = job_dir / "work"
            work_dir.mkdir(parents=True, exist_ok=True)

            # Mock script options
            script_options = [
                {"id": "A", "thesis": "Thesis A", "script_text": "Original text A"},
                {"id": "B", "thesis": "Thesis B", "script_text": "Original text B"},
            ]
            (work_dir / "script_options.json").write_text(json.dumps(script_options))

            # Simulate pause at Gate A
            job["status"] = "awaiting_script_approval"
            jm._save_job(job)

            # Mock start_job to prevent actual stage loop execution during unit test
            with mock.patch.object(jm, "start_job", return_value=job) as mock_start:
                res = jm.approve_gate_a(jid, selected_option_id="A", edited_script_text="Polished text A")
                assert mock_start.called
                assert mock_start.call_args[1].get("from_stage") == "edl"

            # Check selected_script.json was saved with edited text
            sel_file = work_dir / "selected_script.json"
            assert sel_file.exists()
            sel = json.loads(sel_file.read_text())
            assert sel["id"] == "A"
            assert sel["script_text"] == "Polished text A"

            # Check pause flag cleared
            reloaded = jm.get_job(jid)
            assert reloaded["pause_at_script"] is False

    def test_gate_b_audio_approval(self):
        with tempfile.TemporaryDirectory() as td:
            jm = JobManager(jobs_root=Path(td))
            job = jm.create_job(config={"gate_b": True})
            jid = job["id"]
            job_dir = Path(td) / jid
            work_dir = job_dir / "work"
            bgm_cand_dir = work_dir / "assets" / "bgm" / "candidates"
            bgm_cand_dir.mkdir(parents=True, exist_ok=True)
            (bgm_cand_dir / "track_pulse.mp3").write_text("fake_mp3_data")

            # Simulate pause at Gate B
            job["status"] = "awaiting_audio_approval"
            jm._save_job(job)

            with mock.patch.object(jm, "start_job", return_value=job) as mock_start:
                jm.approve_gate_b(jid, selected_bgm="track_pulse.mp3")
                assert mock_start.called
                assert mock_start.call_args[1].get("from_stage") == "graphics"

            # Check symlink or active track
            active_track = work_dir / "assets" / "bgm" / "track.mp3"
            assert active_track.exists()

            reloaded = jm.get_job(jid)
            assert reloaded["pause_at_audio"] is False

    def test_cancel_job(self):
        with tempfile.TemporaryDirectory() as td:
            jm = JobManager(jobs_root=Path(td))
            job = jm.create_job()
            cancelled = jm.cancel_job(job["id"])
            assert cancelled is True
            loaded = jm.get_job(job["id"])
            assert loaded["status"] == "cancelled"


# ============================================================================
# 4. FASTAPI SERVER TESTCLIENT ENDPOINTS & TRAVERSAL TESTS
# ============================================================================

class TestServerRoutes:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_endpoint(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["ok"] is True
        assert "claude" not in data["tools"]
        assert "agy" in data["tools"]
        assert "hyperframes" in data["tools"]
        assert "ffmpeg" in data["tools"]
        assert "/home/sandbox/ContentCreation/raw_clips" in data["default_roots"]

    def test_presets_endpoint(self, client):
        res = client.get("/api/presets")
        assert res.status_code == 200
        data = res.json()
        preset_ids = [p["id"] for p in data["presets"]]
        assert "story_vertical" in preset_ids
        assert "retention_3step" in preset_ids
        assert "landscape_explainer" in preset_ids
        assert len(data["stages"]) == 8

    def test_clips_endpoint_valid_and_empty(self, client):
        # Valid raw_clips directory
        res = client.get("/api/clips?folder=/home/sandbox/ContentCreation/raw_clips")
        assert res.status_code == 200
        data = res.json()
        assert data["exists"] is True
        assert len(data["clips"]) > 0

        # Empty folder query parameter
        res_empty = client.get("/api/clips?folder=")
        assert res_empty.status_code == 400

    def test_thumb_endpoint(self, client):
        clip_path = "/home/sandbox/ContentCreation/raw_clips/PXL_20261110_074640398.mp4"
        res = client.get(f"/api/thumb?path={clip_path}")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/jpeg"

    def test_job_creation_and_enrichment(self, client):
        clip_path = "/home/sandbox/ContentCreation/raw_clips/PXL_20261110_074640398.mp4"
        req_payload = {
            "config": {
                "preset_id": "retention_3step",
                "story_prompt": "Automated test story direction",
                "gate_a": False,
                "gate_b": False,
                "whisper_model": "small.en",
                "ai_broll": "off",
            },
            "clip_paths": [clip_path],
            "start_immediately": False,
        }
        res = client.post("/api/jobs", json=req_payload)
        assert res.status_code == 201
        job = res.json()
        jid = job["id"]
        assert job["config"]["preset_id"] == "retention_3step"
        assert job["config"]["gate_a"] is False
        assert job["config"]["story_prompt"] == "Automated test story direction"
        assert "qa" in job
        assert isinstance(job["qa"], dict)

        # GET /api/jobs/{job_id}
        get_res = client.get(f"/api/jobs/{jid}")
        assert get_res.status_code == 200
        assert get_res.json()["id"] == jid

        # Clean up created test job directory
        shutil.rmtree(JOBS_ROOT / jid, ignore_errors=True)

    def test_uploads_endpoint(self, client):
        fake_video_bytes = b"fake MP4 video payload for upload test"
        files = [("files", ("test_upload.mp4", fake_video_bytes, "video/mp4"))]
        res = client.post("/api/uploads", files=files)
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 1
        uploaded_path = Path(data[0]["path"])
        assert uploaded_path.exists()
        # Clean up uploaded test file
        uploaded_path.unlink(missing_ok=True)

    def test_path_traversal_protection(self, client):
        # Attempt to access /etc/passwd via job files endpoint
        res = client.get("/api/jobs/job_test_security/files/%2E%2E%2F%2E%2E%2Fetc%2Fpasswd")
        assert res.status_code == 403
        assert "Path traversal" in res.json().get("detail", "")


# ============================================================================
# 5. HYPERFRAMES COMPOSITION CHECK & STUDIO PREVIEW MANAGER TESTS
# ============================================================================

class TestHyperFramesAndStudio:
    def test_hyperframes_check_minimal_valid_composition(self):
        with tempfile.TemporaryDirectory() as td:
            hf_dir = Path(td) / "hf"
            hf_dir.mkdir()
            (hf_dir / "hyperframes.json").write_text(
                json.dumps(
                    {
                        "$schema": "https://hyperframes.heygen.com/schema/hyperframes.json",
                        "width": 1080,
                        "height": 1920,
                        "fps": 30,
                        "duration": 5.0,
                    }
                )
            )
            (hf_dir / "index.html").write_text(
                """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=1080, height=1920">
</head>
<body>
  <div id="root" data-composition-id="main" data-no-timeline data-start="0" data-duration="5.0" data-width="1080" data-height="1920" data-fps="30">
    <div id="stage"></div>
  </div>
</body>
</html>"""
            )
            res = subprocess.run(
                ["/home/sandbox/tools/bin/hyperframes", "check", str(hf_dir), "--json"],
                capture_output=True,
                text=True,
            )
            assert res.returncode == 0
            check_data = json.loads(res.stdout)
            assert check_data.get("ok") is True
            assert check_data.get("lint", {}).get("errorCount", 0) == 0

    @pytest.mark.asyncio
    async def test_studio_manager_open_stop_lifecycle(self):
        with tempfile.TemporaryDirectory() as td:
            hf_dir = Path(td) / "hf"
            hf_dir.mkdir()
            (hf_dir / "hyperframes.json").write_text(
                json.dumps(
                    {
                        "$schema": "https://hyperframes.heygen.com/schema/hyperframes.json",
                        "width": 1080,
                        "height": 1920,
                        "fps": 30,
                        "duration": 5.0,
                    }
                )
            )
            (hf_dir / "index.html").write_text(
                """<!doctype html>
<html lang="en"><head><meta charset="utf-8"></head>
<body><div id="root" data-composition-id="main" data-no-timeline data-start="0" data-duration="5.0" data-width="1080" data-height="1920" data-fps="30"></div></body></html>"""
            )

            # Test on ephemeral ports 3040..3050
            sm = StudioManager(port_range=(3040, 3050))
            job_id = "test_studio_preview_job"
            info = await sm.open_studio(job_id, hf_dir)

            assert info["job_id"] == job_id
            assert info["status"] == "running"
            assert 3040 <= info["port"] <= 3050
            assert f"http://localhost:{info['port']}" == info["url"]

            # Query studio status
            status_info = sm.get_studio(job_id)
            assert status_info is not None
            assert status_info["status"] == "running"

            # Stop studio
            stopped = await sm.stop_studio(job_id)
            assert stopped is True
            assert sm.get_studio(job_id) is None


# ============================================================================
# 6. AGY MODELS, JSON ENVELOPE & CO-PILOT BACKEND TESTS
# ============================================================================

class TestAgyModelsAndCopilot:
    def test_validate_model_id_and_presets(self):
        # 1. Default settings validation
        assert DEFAULT_MODEL_SETTINGS == {
            "creative_model": "gemini-3.8-flash-high",
            "mechanical_model": "gemini-3.8-flash-high",
            "vision_model": "gemini-3.8-flash-high",
            "copilot_model": "gemini-3.8-flash-high",
        }

        # 2. Known models resolution
        assert validate_model_id("gemini-3.8-flash-high") == "gemini-3.8-flash-high"
        assert validate_model_id("gemini-4-argon-high") == "gemini-4-argon-high"
        assert validate_model_id("claude-sonnet-5-5-high") == "claude-sonnet-5-5-high"

        # 3. Unknown or empty models fallback to gemini-3.8-flash-high
        assert validate_model_id("unknown-model-xyz") == "gemini-3.8-flash-high"
        assert validate_model_id("") == "gemini-3.8-flash-high"
        assert validate_model_id(None) == "gemini-3.8-flash-high"

        # 4. Tiers and quota warnings
        tiers = {m["tier"] for m in AGY_MODELS}
        assert tiers == {"fast", "balanced", "creative_boost"}

        claude_entries = [m for m in AGY_MODELS if "claude" in m["id"]]
        assert len(claude_entries) >= 6
        for m in claude_entries:
            assert "Quota-scarce" in m["note"]

    def test_extract_json_payload_stale_envelope_quirk(self):
        # Turn 1: with --json-schema, structured_output is fresh
        envelope_turn1 = json.dumps({
            "conversation_id": "conv-123",
            "status": "SUCCESS",
            "structured_output": {"greeting": "hello"},
            "response": '{"greeting": "hello"}',
        })
        res1 = _extract_json_payload(envelope_turn1, prefer_structured=True)
        assert res1 == {"greeting": "hello"}

        # Turn 2: resumed conversation without schema.
        # Empirical agy v1.5.9 quirk: structured_output holds stale turn 1 object, response holds fresh turn 2
        envelope_turn2 = json.dumps({
            "conversation_id": "conv-123",
            "status": "SUCCESS",
            "structured_output": {"greeting": "hello"},  # STALE!
            "response": '{"count": 42}',                 # FRESH!
        })
        res2_copilot = _extract_json_payload(envelope_turn2, prefer_structured=False)
        assert res2_copilot == {"count": 42}
        assert res2_copilot != {"greeting": "hello"}

        # Interrupted / ERROR envelope must not be returned as a valid schema payload
        envelope_error = json.dumps({
            "conversation_id": "conv-err",
            "status": "ERROR",
            "response": "",
            "error": "interrupted",
        })
        assert _extract_json_payload(envelope_error, prefer_structured=True) is None

    def test_job_manager_models_update_and_enrich(self):
        with tempfile.TemporaryDirectory() as td:
            jm = JobManager(jobs_root=Path(td))
            job = jm.create_job(preset_id="story_vertical", clip_paths=[])
            job_id = job["id"]

            # Defaults populated
            assert job["config"]["models"] == DEFAULT_MODEL_SETTINGS

            # Update creative model only
            updated = jm.update_job_models(job_id, {"creative_model": "gemini-4-hydrogen-high"})
            assert updated["creative_model"] == "gemini-4-hydrogen-high"
            assert updated["mechanical_model"] == "gemini-3.8-flash-high"
            assert updated["vision_model"] == "gemini-3.8-flash-high"

            # Updating mechanical_model without explicit vision_model keeps Vision & Covers synced
            updated_mech = jm.update_job_models(job_id, {"mechanical_model": "gemini-4-hydrogen-medium"})
            assert updated_mech["mechanical_model"] == "gemini-4-hydrogen-medium"
            assert updated_mech["vision_model"] == "gemini-4-hydrogen-medium"

            # Explicit vision_model override is still honored
            updated_vis = jm.update_job_models(job_id, {"vision_model": "gemini-4-hydrogen-high"})
            assert updated_vis["mechanical_model"] == "gemini-4-hydrogen-medium"
            assert updated_vis["vision_model"] == "gemini-4-hydrogen-high"

            # Creating a job with mechanical_model syncs vision_model unless explicitly set
            job_custom = jm.create_job(
                preset_id="story_vertical",
                clip_paths=[],
                config={
                    "models": {
                        "creative_model": "gemini-4-hydrogen-xhigh",
                        "mechanical_model": "gemini-4-hydrogen-medium",
                    }
                },
            )
            assert job_custom["config"]["models"]["creative_model"] == "gemini-4-hydrogen-xhigh"
            assert job_custom["config"]["models"]["mechanical_model"] == "gemini-4-hydrogen-medium"
            assert job_custom["config"]["models"]["vision_model"] == "gemini-4-hydrogen-medium"

            # Enriched job retains models
            refreshed = jm.get_job(job_id)
            enriched = enrich_job(refreshed)
            assert enriched["config"]["models"]["creative_model"] == "gemini-4-hydrogen-high"
            assert "copilot" in enriched
            assert enriched["copilot"]["busy"] is False

    @pytest.mark.asyncio
    async def test_copilot_session_and_actions(self):
        with tempfile.TemporaryDirectory() as td:
            jm = JobManager(jobs_root=Path(td))
            job = jm.create_job(preset_id="story_vertical", clip_paths=[])
            job_id = job["id"]
            job_dir = Path(td) / job_id

            # Initial history
            hist = jm.get_copilot_history(job_id)
            assert hist["conversation_id"] is None
            assert hist["messages"] == []

            # Mock run_agy_copilot_turn to simulate turn and file creation
            async def mock_turn(prompt, job_dir, model, conversation_id=None, timeout_s=600.0, log_cb=None, progress_cb=None):
                work_dir = job_dir / "work"
                work_dir.mkdir(parents=True, exist_ok=True)
                # Simulate AI editing work/edl.json
                (work_dir / "edl.json").write_text(json.dumps({"parts": [{"clip": "c1"}]}))
                return {
                    "response_text": "I have created the EDL for you with 1 part.",
                    "conversation_id": "mock-conv-456",
                    "model": model,
                    "duration_seconds": 1.5,
                    "usage": {"total_tokens": 100},
                }

            with mock.patch("orchestrator.run_agy_copilot_turn", side_effect=mock_turn):
                res = await jm.send_copilot_message(job_id, "build edl for this job")
                assert res["copilot"]["conversation_id"] == "mock-conv-456"
                assert len(res["copilot"]["messages"]) == 2
                assert "work/edl.json" in res["modified_files"]
                assert res["suggested_action"] == {
                    "label": "Rebuild Base Cut",
                    "type": "start_stage",
                    "stage": "base_cut",
                }

            # Reset session
            reset_hist = jm.reset_copilot_session(job_id)
            assert reset_hist["conversation_id"] is None
            assert len(reset_hist["messages"]) == 3  # user, assistant, system marker

    def test_server_api_models_and_copilot(self):
        client = TestClient(app)
        # 1. Presets endpoint exposes agy_models and default_models
        presets_res = client.get("/api/presets")
        assert presets_res.status_code == 200
        pdata = presets_res.json()
        assert "agy_models" in pdata
        assert "default_models" in pdata
        assert pdata["default_models"]["creative_model"] == "gemini-3.8-flash-high"

        # 2. Create job and check models patch endpoint
        clip_path = "/home/sandbox/ContentCreation/raw_clips/PXL_20261110_074640398.mp4"
        c_res = client.post("/api/jobs", json={"clip_paths": [clip_path], "start_immediately": False})
        assert c_res.status_code == 201
        job_id = c_res.json()["id"]

        try:
            patch_res = client.patch(f"/api/jobs/{job_id}/models", json={"creative_model": "claude-sonnet-5-5-high"})
            assert patch_res.status_code == 200
            assert patch_res.json()["models"]["creative_model"] == "claude-sonnet-5-5-high"

            # 3. Copilot GET endpoint
            cop_res = client.get(f"/api/jobs/{job_id}/copilot")
            assert cop_res.status_code == 200
            assert cop_res.json()["busy"] is False

            # 4. Copilot POST turn
            async def mock_api_turn(prompt, job_dir, model, conversation_id=None, timeout_s=600.0, log_cb=None, progress_cb=None):
                return {
                    "response_text": "Hello! I am ready to help.",
                    "conversation_id": "api-conv-789",
                    "model": model,
                    "duration_seconds": 1.0,
                    "usage": {"total_tokens": 50},
                }

            with mock.patch("orchestrator.run_agy_copilot_turn", side_effect=mock_api_turn):
                send_res = client.post(f"/api/jobs/{job_id}/copilot", json={"message": "hello"})
                assert send_res.status_code == 200
                sdata = send_res.json()
                assert sdata["message"]["content"] == "Hello! I am ready to help."
                assert sdata["copilot"]["conversation_id"] == "api-conv-789"

            # 5. Copilot reset endpoint
            reset_res = client.post(f"/api/jobs/{job_id}/copilot/reset")
            assert reset_res.status_code == 200
            assert reset_res.json()["conversation_id"] is None
        finally:
            shutil.rmtree(JOBS_ROOT / job_id, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_run_proc_no_false_timeout_on_early_pipe_eof(self):
        # Subprocess closes stdout/stderr pipes early, sleeps briefly, and exits 0.
        script = "import os,time; os.close(1); os.close(2); time.sleep(0.15)"
        code, out, err = await adapters._run_proc([sys.executable, "-c", script], timeout_s=None)
        assert code == 0, f"Unexpected non-zero code on early pipe EOF: {code}, err={err}"
        assert "timed out" not in err.lower()

        # Genuine short timeout still kills and returns -1 cleanly
        t_code, _, t_err = await adapters._run_proc(
            [sys.executable, "-c", "import time; time.sleep(2.0)"],
            timeout_s=0.1,
        )
        assert t_code == -1
        assert "Process timed out after 0.1s" in t_err

    @pytest.mark.asyncio
    async def test_copilot_stream_json_progress_callbacks(self):
        ndjson_lines = [
            json.dumps({"event": "init", "conversation_id": "conv-stream-1", "init": {"model": "gemini-3.8-flash-high"}}),
            json.dumps({
                "event": "step_update",
                "step_update": {
                    "step_index": 2,
                    "state": "ACTIVE",
                    "step_type": "tool",
                    "tool_name": "view_file",
                    "tool_info": {"name": "view_file", "parameters": {"AbsolutePath": "/tmp/job/work/edl.json"}},
                },
            }),
            json.dumps({
                "event": "step_update",
                "step_update": {"step_index": 3, "state": "ACTIVE", "step_type": "agent_response", "text_delta": "Checked EDL."},
            }),
            json.dumps({
                "event": "result",
                "result": {
                    "conversation_id": "conv-stream-1",
                    "status": "SUCCESS",
                    "response": "Checked EDL.",
                    "duration_seconds": 2.4,
                    "usage": {"total_tokens": 321},
                },
            }),
        ]

        async def fake_run_proc(cmd, cwd=None, timeout_s=None, cancel_event=None, log_cb=None, env=None):
            assert "--output-format" in cmd and cmd[cmd.index("--output-format") + 1] == "stream-json"
            assert "--dangerously-skip-permissions" in cmd
            for line in ndjson_lines:
                if log_cb:
                    log_cb(f"[OUT] {line}")
            return (0, "\n".join(ndjson_lines) + "\n", "")

        seen_progress = []
        with tempfile.TemporaryDirectory() as td, mock.patch("adapters._run_proc", side_effect=fake_run_proc):
            res = await adapters.run_agy_copilot_turn(
                prompt="check edl",
                job_dir=Path(td),
                model="gemini-3.8-flash-high",
                progress_cb=seen_progress.append,
            )

        assert res["conversation_id"] == "conv-stream-1"
        assert res["response_text"] == "Checked EDL."
        assert res["duration_seconds"] == pytest.approx(2.4)
        assert any(p.get("kind") == "tool" and "edl.json" in p.get("text", "") for p in seen_progress)
        assert any(p.get("kind") == "delta" and "Checked EDL." in p.get("preview", "") for p in seen_progress)


