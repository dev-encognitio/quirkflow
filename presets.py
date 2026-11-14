"""Preset configurations and directory roots for the autonomous video production pipeline.

Defines vertical, landscape, and retention presets, safe layout bands,
caption formatting rules, and pipeline stage order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Pipeline & Jobs Directory Roots
PIPELINE_ROOT = Path("/home/sandbox/ContentCreation/studio-pipeline")
JOBS_ROOT = Path("/home/sandbox/ContentCreation/jobs")
DEFAULT_CLIP_ROOTS = [
    "/home/sandbox/ContentCreation/raw_clips",
    "/mnt/01/pixel9a/Camera",
]

# Strict 8-Stage Pipeline Order
STAGE_ORDER = [
    "ingest",
    "script",
    "edl",
    "base_cut",
    "bgm_broll",
    "graphics",
    "composite",
    "deliverables",
]

# Preset Definitions
PRESETS: dict[str, dict[str, Any]] = {
    "story_vertical": {
        "id": "story_vertical",
        "name": "Story Vertical (TikTok / Reels / Shorts)",
        "description": "Fast-paced raw monologue style. 1080x1920 vertical with safe graphics and caption bands.",
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "aspect_ratio": "9:16",
        "zoom": 1.08,
        "gamma": 1.08,
        "triple_hook": True,
        "climax_silence_dip_s": 0.0,
        "optional_broll": False,
        "cover_layout": "A",
        "graphics_band": {
            "top": 1140,
            "bottom": 1500,
            "left": 0,
            "right": 1080,
            "clip_path": "inset(1140px 0 420px 0)",
        },
        "caption_band": {
            "top": 1500,
            "bottom": 1640,
            "left": 70,
            "right": 1010,
        },
        "ass_style": {
            "fontname": "Inter Black",
            "fontsize": 76,
            "primary_colour": "&HFFFFFF&",
            "secondary_colour": "&HFFFFFF&",
            "outline_colour": "&H000000&",
            "back_colour": "&H80000000&",
            "outline": 6,
            "shadow": 2,
            "alignment": 2,
            "margin_l": 70,
            "margin_r": 70,
            "margin_v": 330,
            "play_res_x": 1080,
            "play_res_y": 1920,
        },
        "highlight_colors": {
            "pink": "&HA04FFF&",
            "blue": "&HFF6B3D&",
        },
    },
    "retention_3step": {
        "id": "retention_3step",
        "name": "Retention 3-Step Framework",
        "description": "High-retention 1080x1920 vertical with mandatory Triple Hook (0-3s), 0.25s climax silence dip, and optional B-roll cuts.",
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "aspect_ratio": "9:16",
        "zoom": 1.08,
        "gamma": 1.08,
        "triple_hook": True,
        "climax_silence_dip_s": 0.25,
        "optional_broll": True,
        "cover_layout": "A",
        "graphics_band": {
            "top": 1140,
            "bottom": 1500,
            "left": 0,
            "right": 1080,
            "clip_path": "inset(1140px 0 420px 0)",
        },
        "caption_band": {
            "top": 1500,
            "bottom": 1640,
            "left": 70,
            "right": 1010,
        },
        "ass_style": {
            "fontname": "Inter Black",
            "fontsize": 76,
            "primary_colour": "&HFFFFFF&",
            "secondary_colour": "&HFFFFFF&",
            "outline_colour": "&H000000&",
            "back_colour": "&H80000000&",
            "outline": 6,
            "shadow": 2,
            "alignment": 2,
            "margin_l": 70,
            "margin_r": 70,
            "margin_v": 330,
            "play_res_x": 1080,
            "play_res_y": 1920,
        },
        "highlight_colors": {
            "pink": "&HA04FFF&",
            "blue": "&HFF6B3D&",
        },
    },
    "landscape_explainer": {
        "id": "landscape_explainer",
        "name": "Landscape Explainer (16:9)",
        "description": "1920x1080 horizontal format with broad canvas diagrams and centered lower-third captions.",
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "aspect_ratio": "16:9",
        "zoom": 1.05,
        "gamma": 1.06,
        "triple_hook": True,
        "climax_silence_dip_s": 0.15,
        "optional_broll": True,
        "cover_layout": "landscape",
        "graphics_band": {
            "top": 80,
            "bottom": 860,
            "left": 100,
            "right": 1820,
            "clip_path": "inset(80px 100px 220px 100px)",
        },
        "caption_band": {
            "top": 880,
            "bottom": 1020,
            "left": 120,
            "right": 1800,
        },
        "ass_style": {
            "fontname": "Inter Black",
            "fontsize": 68,
            "primary_colour": "&HFFFFFF&",
            "secondary_colour": "&HFFFFFF&",
            "outline_colour": "&H000000&",
            "back_colour": "&H80000000&",
            "outline": 5,
            "shadow": 2,
            "alignment": 2,
            "margin_l": 120,
            "margin_r": 120,
            "margin_v": 80,
            "play_res_x": 1920,
            "play_res_y": 1080,
        },
        "highlight_colors": {
            "pink": "&HA04FFF&",
            "blue": "&HFF6B3D&",
        },
    },
}


def get_preset(preset_id: str | None) -> dict[str, Any]:
    """Retrieve a preset dictionary by ID with fallback to 'story_vertical'."""
    if not preset_id or preset_id not in PRESETS:
        return dict(PRESETS["story_vertical"])
    return dict(PRESETS[preset_id])


# ============================================================================
# AGY MODELS & CONFIGURATION
# ============================================================================

AGY_MODELS: list[dict[str, str]] = [
    {
        "id": "gemini-3.8-flash-high",
        "label": "Gemini 3.8 Flash (High)",
        "tier": "fast",
        "note": "Default fast workhorse model with high reasoning effort. Recommended for all slots.",
    },
    {
        "id": "gemini-3.8-flash-medium",
        "label": "Gemini 3.8 Flash (Medium)",
        "tier": "fast",
        "note": "Fast workhorse model with medium reasoning effort.",
    },
    {
        "id": "gemini-3.8-flash-low",
        "label": "Gemini 3.8 Flash (Low)",
        "tier": "fast",
        "note": "Ultra-fast low-latency model for rapid processing.",
    },
    {
        "id": "gemini-4-argon-high",
        "label": "Gemini 4 Argon (High)",
        "tier": "balanced",
        "note": "Balanced general-purpose model with high reasoning effort.",
    },
    {
        "id": "gemini-4-argon-medium",
        "label": "Gemini 4 Argon (Medium)",
        "tier": "balanced",
        "note": "Balanced general-purpose model with medium reasoning.",
    },
    {
        "id": "gemini-4-argon-low",
        "label": "Gemini 4 Argon (Low)",
        "tier": "fast",
        "note": "Lightweight Argon model for fast tasks.",
    },
    {
        "id": "gemini-4-hydrogen-high",
        "label": "Gemini 4 Hydrogen (High)",
        "tier": "balanced",
        "note": "Advanced deep reasoning model with high effort.",
    },
    {
        "id": "gemini-4-hydrogen-xhigh",
        "label": "Gemini 4 Hydrogen (XHigh)",
        "tier": "creative_boost",
        "note": "Deepest reasoning depth Hydrogen model.",
    },
    {
        "id": "gemini-4-hydrogen-medium",
        "label": "Gemini 4 Hydrogen (Medium)",
        "tier": "balanced",
        "note": "Balanced Hydrogen model for analytical tasks.",
    },
    {
        "id": "gemini-4-hydrogen-low",
        "label": "Gemini 4 Hydrogen (Low)",
        "tier": "fast",
        "note": "Fast Hydrogen model.",
    },
    {
        "id": "claude-sonnet-5-5-high",
        "label": "Claude Sonnet 5.5 (High)",
        "tier": "creative_boost",
        "note": "Quota-scarce: High-tier creative reasoning. Use selectively.",
    },
    {
        "id": "claude-sonnet-5-5-medium",
        "label": "Claude Sonnet 5.5 (Medium)",
        "tier": "creative_boost",
        "note": "Quota-scarce: Medium-tier creative reasoning. Use selectively.",
    },
    {
        "id": "claude-sonnet-5-5-low",
        "label": "Claude Sonnet 5.5 (Low)",
        "tier": "creative_boost",
        "note": "Quota-scarce: Low-tier creative reasoning. Use selectively.",
    },
    {
        "id": "claude-opus-5-5-high",
        "label": "Claude Opus 5.5 (High)",
        "tier": "creative_boost",
        "note": "Quota-scarce: Deepest reasoning & creative power. High quota consumption.",
    },
    {
        "id": "claude-opus-5-5-medium",
        "label": "Claude Opus 5.5 (Medium)",
        "tier": "creative_boost",
        "note": "Quota-scarce: Deep reasoning model. High quota consumption.",
    },
    {
        "id": "claude-opus-5-5-low",
        "label": "Claude Opus 5.5 (Low)",
        "tier": "creative_boost",
        "note": "Quota-scarce: Opus reasoning model. High quota consumption.",
    },
    {
        "id": "gpt-oss-120b-medium",
        "label": "GPT-OSS 120B (Medium)",
        "tier": "balanced",
        "note": "Open weights 120B parameter model.",
    },
]

DEFAULT_MODEL_SETTINGS: dict[str, str] = {
    "creative_model": "gemini-3.8-flash-high",
    "mechanical_model": "gemini-3.8-flash-high",
    "vision_model": "gemini-3.8-flash-high",
    "copilot_model": "gemini-3.8-flash-high",
}

VALID_MODEL_IDS: set[str] = {m["id"] for m in AGY_MODELS}


def validate_model_id(model_id: str | None) -> str:
    """Validate that a model ID exists in AGY_MODELS, falling back to gemini-3.8-flash-high."""
    if model_id and model_id in VALID_MODEL_IDS:
        return model_id
    return "gemini-3.8-flash-high"

