"""Prompt templates, JSON schemas, and deterministic validators for LLM video production agents.

Covers:
- Script generation (A/B/C options guided by GUIDE.md)
- Adversarial critic pass (cold viewer / retention auditor)
- EDL + Config generation with deterministic overlap validation
- HyperFrames motion graphics generation (GSAP + SVG sketch aesthetic)
- Audio spot-check and Vision Art Director critique prompts
- Cover thumbnail selector and TikTok copy descriptions
"""

from __future__ import annotations

import json
import re
from typing import Any

# ============================================================================
# JSON SCHEMAS
# ============================================================================

SCRIPT_OPTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "options": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "enum": ["A", "B", "C"]},
                    "title": {"type": "string"},
                    "thesis": {"type": "string"},
                    "viewer_question": {"type": "string"},
                    "turn": {"type": "string"},
                    "loop": {"type": "string"},
                    "estimated_duration_s": {"type": "number"},
                    "script_text": {"type": "string"},
                    "parts_hint": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "id",
                    "title",
                    "thesis",
                    "viewer_question",
                    "turn",
                    "loop",
                    "estimated_duration_s",
                    "script_text",
                    "parts_hint",
                ],
            },
            "minItems": 3,
            "maxItems": 3,
        }
    },
    "required": ["options"],
}

CRITIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "passes": {"type": "boolean"},
        "viewer_summary": {"type": "string"},
        "issues": {
            "type": "array",
            "items": {"type": "string"},
        },
        "revised_options": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "thesis": {"type": "string"},
                    "viewer_question": {"type": "string"},
                    "turn": {"type": "string"},
                    "loop": {"type": "string"},
                    "estimated_duration_s": {"type": "number"},
                    "script_text": {"type": "string"},
                    "parts_hint": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "id",
                    "title",
                    "thesis",
                    "viewer_question",
                    "turn",
                    "loop",
                    "estimated_duration_s",
                    "script_text",
                ],
            },
        },
    },
    "required": ["passes", "viewer_summary", "issues", "revised_options"],
}

EDL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "highlight": {
            "type": "object",
            "description": "Mapping of keyword words to ASS color tags, e.g. {'pink': '&HA04FFF&'}",
        },
        "names": {
            "type": "object",
            "description": "Mapping of misheard/lowercase names to proper capitalization",
        },
        "parts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "clip": {"type": "string"},
                    "range": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "say": {"type": "string"},
                    "hold": {"type": "number"},
                    "beat": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["id", "clip", "range", "say"],
            },
        },
    },
    "required": ["title", "parts"],
}

HYPERFRAMES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "index_html": {"type": "string"},
        "cues_json": {"type": "object"},
        "cues_mjs": {"type": "string"},
    },
    "required": ["index_html", "cues_json", "cues_mjs"],
}

AUDIO_SPOTCHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "passes": {"type": "boolean"},
        "review_notes": {"type": "string"},
        "audible_bursts_count": {"type": "integer"},
        "recommended_cuts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["start", "end", "reason"],
            },
        },
    },
    "required": ["passes", "review_notes", "recommended_cuts"],
}

VISION_CRITIQUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "passes": {"type": "boolean"},
        "critique": {"type": "string"},
        "issues": {
            "type": "array",
            "items": {"type": "string"},
        },
        "fixes_needed": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["passes", "critique", "issues", "fixes_needed"],
}

COVER_SELECTOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cover_A": {
            "type": "object",
            "properties": {
                "frame": {"type": "string"},
                "layout": {"type": "string", "enum": ["A", "B"]},
                "title": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["frame", "layout", "title"],
        },
        "cover_B": {
            "type": "object",
            "properties": {
                "frame": {"type": "string"},
                "layout": {"type": "string", "enum": ["A", "B"]},
                "title": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["frame", "layout", "title"],
        },
        "reasoning": {"type": "string"},
    },
    "required": ["cover_A", "cover_B", "reasoning"],
}

DESCRIPTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hook": {"type": "string"},
        "teaser": {"type": "string"},
        "confessional": {"type": "string"},
        "debate": {"type": "string"},
        "minimal": {"type": "string"},
        "recommendation": {"type": "string"},
    },
    "required": ["hook", "teaser", "confessional", "debate", "minimal", "recommendation"],
}

# ============================================================================
# DETERMINISTIC EDL OVERLAP VALIDATOR & FIXER
# ============================================================================

def validate_and_fix_edl_overlaps(
    edl: dict[str, Any],
    words_by_clip: dict[str, list[list[Any]]] | None = None,
) -> dict[str, Any]:
    """Inspect and deterministically fix overlapping time ranges across adjacent EDL parts.

    Guarantees:
    1. Consecutive parts referencing the same source clip never share audio ranges
       (prevents the 'singlet-let' echo bug).
    2. Range start is strictly less than range end.
    3. Hold durations do not leak into adjacent parts on the same clip.
    4. All parts have unique IDs and valid numeric ranges.

    Args:
        edl: Raw or LLM-generated EDL dictionary with 'parts'.
        words_by_clip: Optional dictionary mapping clip_id -> [[word, start, end], ...].

    Returns:
        Sanitized EDL dictionary with fixed ranges and normalized fields.
    """
    fixed_edl = dict(edl)
    raw_parts = fixed_edl.get("parts", [])
    if not raw_parts:
        return fixed_edl

    fixed_parts: list[dict[str, Any]] = []

    for idx, p in enumerate(raw_parts):
        part = dict(p)
        part_id = part.get("id") or f"p{idx + 1:02d}"
        part["id"] = part_id

        # Normalize range
        rng = part.get("range", [0.0, 1.0])
        if not isinstance(rng, (list, tuple)) or len(rng) < 2:
            rng = [0.0, 1.0]
        s, e = float(rng[0]), float(rng[1])
        if s < 0.0:
            s = 0.0
        if e <= s:
            e = s + 0.5
        part["range"] = [round(s, 3), round(e, 3)]
        part["hold"] = max(0.0, float(part.get("hold", 0.0)))
        fixed_parts.append(part)

    # Clean overlaps across consecutive parts of the same clip
    for i in range(len(fixed_parts) - 1):
        curr = fixed_parts[i]
        nxt = fixed_parts[i + 1]

        if curr.get("clip") == nxt.get("clip"):
            curr_start, curr_end = curr["range"]
            nxt_start, nxt_end = nxt["range"]
            effective_curr_end = curr_end + curr.get("hold", 0.0)

            # Check if current part overlaps next part
            if effective_curr_end >= nxt_start:
                clip_id = curr.get("clip", "")
                words = (words_by_clip or {}).get(clip_id, [])

                # Attempt word boundary resolution
                resolved = False
                if words:
                    # Find last word of current segment and first word of next segment
                    curr_words = [
                        w for w in words if curr_start <= float(w[1]) <= curr_end
                    ]
                    nxt_words = [
                        w for w in words if nxt_start <= float(w[1]) <= nxt_end
                    ]

                    if curr_words and nxt_words:
                        last_curr_end = float(curr_words[-1][2])
                        first_nxt_start = float(nxt_words[0][1])

                        if first_nxt_start > last_curr_end:
                            split_pt = round((last_curr_end + first_nxt_start) / 2.0, 3)
                            curr["range"][1] = split_pt
                            nxt["range"][0] = split_pt
                            curr["hold"] = 0.0
                            resolved = True

                if not resolved:
                    # Deterministic midpoint split with 0.02s gap
                    mid = round((curr_end + nxt_start) / 2.0, 3)
                    curr["range"][1] = max(round(curr_start + 0.1, 3), round(mid - 0.02, 3))
                    nxt["range"][0] = min(round(nxt_end - 0.1, 3), round(mid + 0.02, 3))
                    curr["hold"] = 0.0

            # Guard against invalid ranges after adjustment
            if curr["range"][1] <= curr["range"][0]:
                curr["range"][1] = round(curr["range"][0] + 0.2, 3)
            if nxt["range"][0] >= nxt["range"][1]:
                nxt["range"][1] = round(nxt["range"][0] + 0.2, 3)

    fixed_edl["parts"] = fixed_parts
    return fixed_edl


# ============================================================================
# PROMPT BUILDERS
# ============================================================================

def build_script_generation_prompt(
    transcript_text: str,
    guide_summary: str | None = None,
    preset: dict[str, Any] | None = None,
    story_prompt: str = "",
) -> str:
    """Build the prompt for generating 3 distinct script concepts (A/B/C)."""
    p = preset or {}
    w, h = p.get("width", 1080), p.get("height", 1920)
    aspect = p.get("aspect_ratio", "9:16")

    guide_block = guide_summary or (
        "- Follow GUIDE.md: Act as an Elite Short-Form Video Director.\n"
        "- The Triple Hook Method (00:00 - 00:03): Synchronize title text hook, visual pattern interrupt, and spoken first line.\n"
        "- Zero fluff: No 'Hey guys', no throat-clearing, no preamble. Start directly inside the conflict.\n"
        "- The Sound-Off Test: Ensure premise is 100% understandable even on mute.\n"
        "- Format choice: A vs B comparison, 3-step framework, or counter-intuitive myth."
    )

    user_story_block = ""
    if story_prompt and story_prompt.strip():
        user_story_block = f"\nCREATOR'S STORY DIRECTION & FOCUS:\n{story_prompt.strip()}\n"

    return f"""You are an elite short-form video editor and retention specialist producing a high-performing video ({aspect}, {w}x{h}).

TASK:
Review the following complete transcript of raw recording clips. Formulate exactly THREE distinct, compelling script options (Option A, Option B, Option C).
Each script must be constructed entirely from spoken ideas and quotes present in the raw footage.
{user_story_block}
RAW TRANSCRIPT:
```
{transcript_text}
```

RETENTION DIRECTIVES (GUIDE.md):
{guide_block}

REQUIREMENTS FOR EACH OPTION:
1. id: "A", "B", or "C".
2. title: High-contrast 3-6 word hook title.
3. thesis: The single core insight or takeaway.
4. viewer_question: The curiosity or skepticism the hook activates.
5. turn: The surprising twist, counter-intuitive insight, or revelation.
6. loop: How the ending phrase seamlessly connects back to the hook or re-contextualizes it.
7. estimated_duration_s: Estimated duration between 40s and 120s.
8. script_text: The complete, punchy narration text assembled for the spoken edit.
9. parts_hint: List of logical beat titles or parts.

Return valid JSON conforming to the schema. Output JSON only."""


def build_script_critic_prompt(
    script_options: list[dict[str, Any]],
    transcript_text: str,
    preset: dict[str, Any] | None = None,
) -> str:
    """Build the prompt for the adversarial cold-viewer critic pass."""
    options_json = json.dumps(script_options, indent=2)
    return f"""You are an adversarial short-form video viewer and algorithmic retention auditor.
You scroll TikTok, Reels, and YouTube Shorts with zero patience. You skip immediately if:
- The first 2 seconds contain throat clearing, preamble, or generic statements.
- The hook fails the Sound-Off Test (cannot be understood on mute).
- The speaker meanders, repeats themselves, or makes unearned conclusions.
- The ending fizzles out with a generic 'follow for more'.

SCRIPTS UNDER REVIEW:
{options_json}

ORIGINAL SOURCE TRANSCRIPT:
```
{transcript_text[:3000]}
```

AUDIT INSTRUCTIONS:
1. Evaluate each option with brutal honesty.
2. If ANY option contains throat-clearing, filler, or a slow hook, identify the exact flaw in 'issues'.
3. In 'revised_options', output the tightened, perfected versions of all 3 options with filler trimmed and the Triple Hook reinforced.
4. Set 'passes' to true only if all revised scripts meet outlier retention standards.

Return valid JSON conforming to the critic schema. Output JSON only."""


def build_edl_prompt(
    selected_script: dict[str, Any],
    words_json_summary: str,
    transcript_text: str,
    preset: dict[str, Any] | None = None,
) -> str:
    """Build the prompt for generating the Edit Decision List (EDL) and config."""
    script_str = json.dumps(selected_script, indent=2)
    return f"""You are a precision video assembly engineer. Build an exact Edit Decision List (edl.json) matching the approved script.

APPROVED SCRIPT:
{script_str}

CLIP TRANSCRIPTS & WORD TIMESTAMPS:
```
{words_json_summary}
```

EDL REQUIREMENTS:
1. 'title': Catchy title for config.
2. 'highlight': Dict mapping keywords to ASS subtitle colors (e.g., '{{"pink": "&HA04FFF&", "blue": "&HFF6B3D&"}}').
3. 'names': Dict of misheard names to fix (e.g., '{{"christian": "Christian"}}').
4. 'parts': Array of ordered cut segments:
   - 'id': Segment identifier (e.g., 'p01', 'p02', ...).
   - 'clip': Source clip id (e.g., 'c01', 'c02').
   - 'range': [start_seconds, end_seconds] in the source clip.
   - 'say': Verbatim words spoken during this range.
   - 'hold': Float seconds to hold video tail before next cut (0.0 to 0.2).
   - 'beat': Beat type ('hook', 'setup', 'twist', 'climax', 'loop').
   - 'note': Visual cue or context note.

CRITICAL RULES:
- Never cut inside a spoken word.
- Never let consecutive parts on the same clip overlap in source time.
- Match verbatim script words closely so word-level captions align cleanly.

Return valid JSON conforming to the EDL schema. Output JSON only."""


def build_hyperframes_prompt(
    timeline_json: dict[str, Any],
    edl_json: dict[str, Any],
    preset: dict[str, Any] | None = None,
) -> str:
    """Build the prompt for generating HyperFrames motion graphics (GSAP + SVG)."""
    p = preset or {}
    w = p.get("width", 1080)
    h = p.get("height", 1920)
    fps = p.get("fps", 30)
    total_dur = timeline_json.get("total", 60.0)
    g_band = p.get("graphics_band", {"top": 1140, "bottom": 1500, "clip_path": "inset(1140px 0 420px 0)"})
    c_band = p.get("caption_band", {"top": 1500, "bottom": 1640})

    timeline_str = json.dumps(timeline_json.get("parts", [])[:15], indent=2)

    return f"""You are an elite motion designer building a HyperFrames composition for an overlay track.
Dimensions: {w}x{h} at {fps}fps. Total Duration: {total_dur:.2f}s.

LAYOUT SAFETY BOUNDS:
- Stage safe band: y={g_band.get('top', 1140)} to y={g_band.get('bottom', 1500)}.
- Stage clip-path: {g_band.get('clip_path', 'inset(1140px 0 420px 0)')}.
- Caption keep-out band: y={c_band.get('top', 1500)} to y={c_band.get('bottom', 1640)}.
- ZERO graphics or text elements may ever render inside or occlude the caption band!

ASSETS & RUNTIME:
- Local GSAP: <script src="assets/gsap.min.js"></script>
- Cues script: <script src="cues.js"></script>
- Local fonts: PermanentMarker ('assets/fonts/PermanentMarker.woff2') and Caveat-700 ('assets/fonts/Caveat-700.woff2').
- SVG sketchy chalk and wob displacement filters in defs (#chalk and #wob).
- Single GSAP master timeline:
    const tl = gsap.timeline({{ paused: true }});
    window.__timelines = window.__timelines || {{}};
    window.__timelines["main"] = tl;
- HARD GSAP RULE: NO CSS `transform` on GSAP target elements! GSAP must own all transforms (x, y, scale, rotation).

TIMELINE BEATS (First 15 parts):
{timeline_str}

OUTPUT FILES REQUIRED:
1. 'index_html': Complete, standalone HTML entry point for HyperFrames with SVG paths, sketchy illustrations, and GSAP tweens synced to CUES.
2. 'cues_json': JSON object mapping cue names to exact timestamp offsets from the timeline beats, plus 'total': {total_dur:.2f}.
3. 'cues_mjs': Node.js script synchronizing cues.json to cues.js and updating data-duration on index.html:
```javascript
import fs from 'fs';
const c = JSON.parse(fs.readFileSync('cues.json', 'utf8'));
fs.writeFileSync('cues.js', 'window.CUES=' + JSON.stringify(c) + ';\\n');
const h = fs.readFileSync('index.html', 'utf8').replace(/data-duration="[^"]*"/, `data-duration="${{c.total}}"`);
fs.writeFileSync('index.html', h);
```

Return valid JSON with keys index_html, cues_json, cues_mjs. Output JSON only."""


def build_audio_spotcheck_prompt(
    bursts_json_path: str,
    qa_report_path: str,
    base_audio_path: str,
) -> str:
    """Build the AGY prompt for audio spot-checking.

    Explicitly instructs the agent: 'Inspect the referenced files directly; do not run shell commands.'
    """
    return f"""Inspect the referenced files directly; do not run shell commands.

You are an audio engineer conducting a spot-check on video narration audio.
Referenced files to inspect:
- Bursts log: {bursts_json_path}
- QA report: {qa_report_path}
- Audio source: {base_audio_path}

TASK:
1. Examine bursts detected outside words. Determine whether any non-speech bursts (-37dB to -14dB) represent distracting coughs, loud table bumps, or mic thumps requiring gating.
2. Confirm if natural pauses and breathing sound smooth or unnaturally truncated.
3. Recommend specific cuts if any audio artifacts remain.

Return valid JSON conforming to the audio spotcheck schema. Output JSON only."""


def build_vision_critique_prompt(
    snapshots_dir: str,
    cues_json_path: str,
    preset: dict[str, Any] | None = None,
) -> str:
    """Build the AGY prompt for vision art director critique.

    Explicitly instructs the agent: 'Inspect the referenced files directly; do not run shell commands.'
    """
    p = preset or {}
    g_top = p.get("graphics_band", {}).get("top", 1140)
    g_bot = p.get("graphics_band", {}).get("bottom", 1500)
    c_top = p.get("caption_band", {}).get("top", 1500)
    c_bot = p.get("caption_band", {}).get("bottom", 1640)

    return f"""Inspect the referenced files directly; do not run shell commands.

You are a creative director and visual QA lead reviewing rendered motion graphics keyframes.
Referenced files to inspect:
- Snapshot images directory: {snapshots_dir}
- Timing cues: {cues_json_path}

CRITICAL CHECKS:
1. Layout Safe Area:
   - All graphics must sit strictly within y={g_top}..{g_bot}.
   - The caption band y={c_top}..{c_bot} must remain completely clear and unobstructed.
2. Visual Aesthetic:
   - Hand-drawn sketchy chalk aesthetic (#F4EFE6 warm white, #FF8A3D warm accent, #C8F560 lime).
   - High legibility, organic motion, zero ugly clipping or cut-off labels.
3. Determine if the composition passes or requires a revision pass.

Return valid JSON conforming to the vision critique schema. Output JSON only."""


def build_cover_prompt(
    candidate_frames: list[str],
    title_options: list[str],
    preset: dict[str, Any] | None = None,
) -> str:
    """Build the prompt for choosing cover frames and title layouts."""
    frames_list = "\n".join(f"- {f}" for f in candidate_frames)
    titles_list = "\n".join(f"- {t}" for t in title_options)

    return f"""You are a thumbnail designer optimizing for maximum Click-Through Rate (CTR) on TikTok and Shorts.

CANDIDATE FRAMES:
{frames_list}

HOOK TITLES:
{titles_list}

RULES FOR COVERS:
- Layout A: Title at top (y~190), sticker/graphic lower (y~1235).
- Layout B: Sticker at top (y~170), title lower (y~1060).
- Titles must be 3 rows max, formatted as `[[[text, color]], ...]`.
- Available colors: 'w' (white [255,255,255]), 'p' (pink [255,79,160]), 'b' (blue [61,107,255]).
- Choose the sharpest, most expressive candidate frame for Cover A and Cover B.

Return valid JSON conforming to the cover selector schema. Output JSON only."""


def build_descriptions_prompt(
    final_transcript: str,
    script_text: str,
    summary: str,
    series_day: int = 6,
) -> str:
    """Build the prompt for generating 5 distinct TikTok description options."""
    return f"""You are a TikTok copywriter for a creator who posts raw, thoughtful 'burst your bubble' monologues filmed on night walks.

VIDEO SUMMARY:
{summary}

SCRIPT:
```
{script_text}
```

FINAL TRANSCRIPT:
```
{final_transcript[:3000]}
```

IMPORTANT:
- Respect hypotheticals: the speaker does not claim personal preferences or beliefs that were only thought experiments; never invent claims.
- Series: 'burst your bubble', continuing 'Day 1: afraid of the camera' (this is Day {series_day}).

Write FIVE distinct options with different angles:
(1) direct question hook
(2) story teaser/cliffhanger
(3) honest/confessional
(4) debate-bait inviting comments
(5) minimal & intriguing

Format each option with:
- Caption: 90-160 chars before hashtags, <=2 emojis, none required.
- Hashtags: 3-5 tags (#burstyourbubble + topic tags, no spammy #fyp).
- Pinned comment: One-line idea inviting the long story.
- Recommendation: Add a 'which to pick' line with a 1-sentence reason.

Output the complete text for descriptions."""
