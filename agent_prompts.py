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
import math
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

HYPERFRAMES_SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cues_json": {"type": "object"},
        "scenes_js": {"type": "string"},
    },
    "required": ["cues_json", "scenes_js"],
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

        # Snap/repair misaligned ranges against word timestamps so build.py match() succeeds.
        clip_id = str(part.get("clip", ""))
        words = (words_by_clip or {}).get(clip_id, [])
        say = str(part.get("say", "")).strip()
        if words and say:
            in_range = _find_word_match_span(words, say, target_range=part["range"], restrict_to_range=True)
            if in_range is None:
                best_span = _find_word_match_span(words, say, target_range=part["range"], restrict_to_range=False)
                if best_span is not None:
                    cand_s, cand_e = best_span
                    # Floor start / ceil end to 3 decimals so build.py's rng[0] <= word_start <= rng[1] always holds.
                    s_aligned = math.floor(max(0.0, cand_s) * 1000.0) / 1000.0
                    e_aligned = math.ceil(cand_e * 1000.0) / 1000.0
                    if e_aligned <= s_aligned:
                        e_aligned = round(s_aligned + 0.2, 3)
                    part["range"] = [s_aligned, e_aligned]

        fixed_parts.append(part)

    # Clean true chronological overlaps across consecutive parts of the same clip.
    # Disjoint reorders (forward: eff_curr_end <= nxt_start, or backward: nxt_end <= curr_start)
    # must remain untouched.
    for i in range(len(fixed_parts) - 1):
        curr = fixed_parts[i]
        nxt = fixed_parts[i + 1]
        if curr.get("clip") != nxt.get("clip"):
            continue
        _resolve_same_clip_overlap(curr, nxt, (words_by_clip or {}).get(str(curr.get("clip", "")), []))

    fixed_edl["parts"] = fixed_parts
    return fixed_edl


def _norm_token(text: str) -> str:
    """Normalize a word token using the exact regex expected by build.py."""
    return re.sub(r"[^a-z0-9']", "", str(text).lower())


def _find_word_match_span(
    words: list[list[Any]],
    say: str,
    target_range: list[float] | None = None,
    restrict_to_range: bool = True,
) -> tuple[float, float] | None:
    """Locate the [start, end] word span matching `say` with >= 0.80 token overlap."""
    toks = [t for t in (_norm_token(x) for x in say.split()) if t]
    if not toks or len(words) < len(toks):
        return None

    lo = float(target_range[0]) if (restrict_to_range and target_range) else None
    hi = float(target_range[1]) if (restrict_to_range and target_range) else None
    ws = [
        (_norm_token(str(w[0])), float(w[1]), float(w[2]))
        for w in words
        if lo is None or (lo <= float(w[1]) <= hi)
    ]
    if len(ws) < len(toks):
        return None

    ref_s = float(target_range[0]) if target_range else 0.0
    ref_e = float(target_range[1]) if target_range else 0.0
    best_in_range: tuple[float, float, float] | None = None
    candidates: list[tuple[int, float, float, float, float]] = []

    for i in range(len(ws) - len(toks) + 1):
        score = sum(a == b for a, b in zip(toks, (w[0] for w in ws[i : i + len(toks)]))) / len(toks)
        if score < 0.8:
            continue
        cand_s = ws[i][1]
        cand_e = ws[i + len(toks) - 1][2]
        if restrict_to_range:
            if best_in_range is None or score > best_in_range[0]:
                best_in_range = (score, cand_s, cand_e)
        else:
            dist = abs(cand_s - ref_s) + abs(cand_e - ref_e)
            is_far = 0 if dist <= 1.0 else 1
            candidates.append((is_far, -round(score, 3), dist, cand_s, cand_e))

    if restrict_to_range:
        return (best_in_range[1], best_in_range[2]) if best_in_range else None
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _, _, _, best_s, best_e = candidates[0]
    return (best_s, best_e)


def _resolve_same_clip_overlap(
    curr: dict[str, Any],
    nxt: dict[str, Any],
    words: list[list[Any]],
) -> None:
    """Resolve true chronological overlap between adjacent EDL parts on the same clip."""
    curr_start, curr_end = curr["range"]
    nxt_start, nxt_end = nxt["range"]
    effective_curr_end = curr_end + curr.get("hold", 0.0)

    # Disjoint forward OR disjoint non-chronological backward jump -> never modify either range.
    if effective_curr_end <= nxt_start or nxt_end <= curr_start:
        return

    # Only hold exceeded nxt_start; drop hold without altering speech ranges.
    if curr_end <= nxt_start:
        curr["hold"] = 0.0
        return

    earlier, later = (curr, nxt) if curr_start <= nxt_start else (nxt, curr)
    e_start, e_end = earlier["range"]
    l_start, l_end = later["range"]

    resolved = False
    if words:
        e_say = str(earlier.get("say", "")).strip()
        l_say = str(later.get("say", "")).strip()
        e_span = _find_word_match_span(words, e_say, earlier["range"], True) if e_say else None
        l_span = _find_word_match_span(words, l_say, later["range"], True) if l_say else None
        boundary: tuple[float, float] | None = None
        if e_span and l_span:
            boundary = (e_span[1], l_span[0])
        else:
            e_words = [w for w in words if e_start <= float(w[1]) <= e_end]
            l_words = [w for w in words if l_start <= float(w[1]) <= l_end]
            if e_words and l_words:
                boundary = (float(e_words[-1][2]), float(l_words[0][1]))

        if boundary and boundary[1] > boundary[0]:
            split_pt = round((boundary[0] + boundary[1]) / 2.0, 3)
            if e_start < split_pt < l_end:
                earlier["range"][1] = split_pt
                later["range"][0] = split_pt
                curr["hold"] = 0.0
                resolved = True

    if not resolved:
        overlap_s = l_start
        overlap_e = min(e_end, l_end)
        mid = round((overlap_s + overlap_e) / 2.0, 3)
        earlier["range"][1] = max(round(e_start + 0.1, 3), round(mid - 0.02, 3))
        later["range"][0] = min(round(l_end - 0.1, 3), round(mid + 0.02, 3))
        curr["hold"] = 0.0

    if earlier["range"][1] <= earlier["range"][0]:
        earlier["range"][1] = round(earlier["range"][0] + 0.2, 3)
    if later["range"][0] >= later["range"][1]:
        later["range"][1] = round(later["range"][0] + 0.2, 3)


def format_words_for_edl(words_by_clip: dict[str, list[list[Any]]], max_chars: int = 45000) -> str:
    """Serialize word timestamps for the EDL prompt without truncating clips mid-JSON."""
    compact = json.dumps(words_by_clip, separators=(",", ":"))
    if len(compact) <= max_chars:
        return compact

    dense_dict = {
        str(cid): " ".join(f"{w[0]}@{round(float(w[1]), 2)}" for w in ws if len(w) >= 2)
        for cid, ws in words_by_clip.items()
    }
    dense = json.dumps(dense_dict, separators=(",", ":"))
    if len(dense) <= max_chars:
        return dense

    total_len = sum(len(v) for v in dense_dict.values()) or 1
    usable = max(200, max_chars - 64 - 16 * len(dense_dict))
    bounded: dict[str, str] = {}
    for cid, text in dense_dict.items():
        share = max(80, int(usable * len(text) / total_len))
        if len(text) <= share:
            bounded[cid] = text
        else:
            cut = text[: max(0, share - 4)]
            if " " in cut:
                cut = cut.rsplit(" ", 1)[0]
            bounded[cid] = cut + " ..."
    return json.dumps(bounded, separators=(",", ":"))


# ============================================================================
# PROMPT BUILDERS
# ============================================================================

def build_script_generation_prompt(
    transcript_text: str,
    guide_summary: str | None = None,
    preset: dict[str, Any] | None = None,
    story_prompt: str = "",
    clip_summary: str = "",
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

    clip_block = f"\nSOURCE CLIPS METADATA:\n{clip_summary}\n" if clip_summary else ""

    return f"""Do NOT read external files or run shell commands. Work strictly from the provided text and output valid JSON only.

You are an elite short-form video editor and retention specialist producing a high-performing video ({aspect}, {w}x{h}).

TASK:
Review the complete transcript of raw recording clips below. Formulate THREE distinct, compelling script options (Option A, Option B, Option C).
{user_story_block}{clip_block}
RAW TRANSCRIPT:
```
{transcript_text[:36000]}
```

COHERENCE & GROUNDING RULES (NON-NEGOTIABLE):
1. Construct EVERY line from actual spoken clauses and sentences present in the raw transcript — never invent transitions or words the speaker never said!
2. Prefer contiguous multi-sentence takes and complete clauses with natural grammatical lead-ins. NEVER plan rapid 1-second Frankencut stitches of unrelated noun phrases or dangling fragments.
3. Read each assembled `script_text` aloud mentally: every transition between sentences must sound like continuous, grammatical speech from one human thought to the next.
4. Respect hypotheticals: if the speaker describes a hypothetical example, keep that hypothetical framing intact — never present a thought experiment as a factual personal claim.
5. Exclude filler rambling, coughs, throat-clears, profanity, or meta comments ("sorry it's noisy").
6. Avoid mixing visually mismatched clips mid-thought when a single cohesive take covers the idea.

RETENTION DIRECTIVES (GUIDE.md):
{guide_block[:4000]}

REQUIREMENTS FOR EACH OPTION:
1. id: "A", "B", or "C".
2. title: High-contrast 3-6 word hook title.
3. thesis: The single core insight or takeaway.
4. viewer_question: The curiosity or skepticism the hook activates.
5. turn: The surprising twist, counter-intuitive insight, or revelation.
6. loop: How the ending phrase seamlessly connects back to the hook or re-contextualizes it.
7. estimated_duration_s: Realistic spoken duration (typically 40s-120s matching selected clauses).
8. script_text: Complete spoken edit built cleanly from grounded clauses in the transcript.
9. parts_hint: List of logical beat titles or parts.

Return valid JSON conforming to the schema. Output JSON only."""


def build_script_critic_prompt(
    script_options: list[dict[str, Any]],
    transcript_text: str,
    preset: dict[str, Any] | None = None,
) -> str:
    """Build the prompt for the adversarial cold-viewer critic pass."""
    options_json = json.dumps(script_options, indent=2)
    return f"""Do NOT read external files or run shell commands. Work strictly from the provided text and output valid JSON only.

You are an adversarial short-form video viewer and algorithmic retention auditor.
You scroll TikTok, Reels, and YouTube Shorts with zero patience. You skip immediately if:
- The first 2 seconds contain throat clearing, preamble, or generic statements.
- Lines read like choppy stitched fragments, broken grammar, or missing clause lead-ins.
- Words in `script_text` were invented and do NOT appear verbatim inside ORIGINAL SOURCE TRANSCRIPT.
- The ending fizzles out instead of delivering a clear turn and loop.

SCRIPTS UNDER REVIEW:
{options_json}

ORIGINAL SOURCE TRANSCRIPT:
```
{transcript_text[:36000]}
```

AUDIT INSTRUCTIONS:
1. Check every option for spoken coherence (read `script_text` aloud sentence-by-sentence) AND verify phrases exist in ORIGINAL SOURCE TRANSCRIPT.
2. Identify weak hooks, broken transitions, dangling pronouns, or ungrounded wording in 'issues'.
3. In 'revised_options', output perfected, tightened, 100% transcript-grounded versions of all 3 options with clean clause transitions and strong Triple Hooks.
4. Set 'passes' to true only if all revised scripts read like fluent, natural speech with high retention hooks.

Return valid JSON conforming to the critic schema. Output JSON only."""


def build_edl_prompt(
    selected_script: dict[str, Any],
    words_json_summary: str,
    transcript_text: str,
    preset: dict[str, Any] | None = None,
    clip_summary: str = "",
) -> str:
    """Build the prompt for generating the Edit Decision List (EDL) and config."""
    script_str = json.dumps(selected_script, indent=2)
    clip_block = f"\nSOURCE CLIP INFO:\n{clip_summary}\n" if clip_summary else ""
    return f"""Do NOT read external files or run shell commands. Work strictly from the provided inputs and output valid JSON only.

You are a precision video assembly engineer. Build an exact Edit Decision List (edl.json) matching the approved script.

APPROVED SCRIPT:
{script_str}
{clip_block}
CLIP TRANSCRIPTS & WORD TIMESTAMPS:
```
{words_json_summary}
```

EDL REQUIREMENTS:
1. 'title': Catchy 3-6 word hook title.
2. 'highlight': Dict mapping 4-6 meaningful lowercase single keywords from 'say' to ASS BGR hex colors (NOTE ASS uses &HBBGGRR& order! E.g. coral/orange='&H3D8AFF&', lime='&H60F5C8&', cyan='&HFFD400&', pink='&HFFA04F&', gold='&H4FD5FF&'). Keys MUST be lowercase stripped words matching captions!
3. 'names': Dict of lowercase/misheard words or proper nouns to fix in captions (e.g., '{{"christian": "Christian"}}').
4. 'parts': Ordered cut segments:
   - 'id': 'p01', 'p02', ...
   - 'clip': Source clip id ('c01', 'c02', ...).
   - 'range': [start_seconds, end_seconds] enclosing every spoken word in 'say' from CLIP TRANSCRIPTS.
   - 'say': EXACT verbatim words spoken in that source range.
   - 'hold': Float tail hold (0.0 to 0.15).
   - 'beat': 'hook' | 'setup' | 'twist' | 'climax' | 'loop'.
   - 'note': Specific visual motion-graphic cue idea for this beat (what hand-drawn SVG diagram/object/text should animate in the safe band!).

CRITICAL ASSEMBLY & COHERENCE RULES:
- Prefer longer natural continuous takes (at least 2.5s-15s per part when available; typically 5-12 cohesive parts for ~45-60s) instead of tiny 1s fragments.
- Every cut transition `p[i].say` -> `p[i+1].say` MUST form fluent, grammatical spoken English when read straight through! Always include clause lead-ins ("because...", "and when...", "so...") inside the part so joins never sound like chopped word salad.
- Avoid gratuitous clip ping-ponging (e.g., switching c03 <-> c01 every 2 seconds) or jumping to takes with different framing/glasses mid-thought unless starting a deliberate distinct beat.
- Never let consecutive chronological parts on the same clip overlap in source time.
- Ensure 'say' matches verbatim words inside 'range' so word-level matching scores >= 0.90.

Return valid JSON conforming to the EDL schema. Output JSON only."""


def _extract_hyperframes_beats(
    timeline_json: dict[str, Any],
    edl_json: dict[str, Any],
) -> list[dict[str, Any]]:
    """Merge timeline parts with EDL beat notes and visual cues."""
    edl_by_id = {str(p.get("id")): p for p in (edl_json or {}).get("parts", []) if p.get("id")}
    edl_parts = (edl_json or {}).get("parts", [])
    merged: list[dict[str, Any]] = []
    for idx, tp in enumerate(timeline_json.get("parts", [])):
        pid = str(tp.get("id") or f"p{idx + 1:02d}")
        ep = edl_by_id.get(pid) or (edl_parts[idx] if idx < len(edl_parts) else {})
        merged.append({
            "id": pid,
            "beat": tp.get("beat") or ep.get("beat") or "setup",
            "start": round(float(tp.get("start", 0.0)), 2),
            "end": round(float(tp.get("end", 0.0)), 2),
            "say": str(ep.get("say") or tp.get("say") or "").strip(),
            "note": str(ep.get("note") or "").strip(),
        })
    return merged


def render_hyperframes_index_html(
    preset: dict[str, Any] | None,
    scenes_js: str,
    total_dur: float,
) -> str:
    """Wrap LLM-generated or fallback GSAP/SVG `scenes_js` in the battle-tested HyperFrames chalk scaffold."""
    p = preset or {}
    w = int(p.get("width", 1080))
    h = int(p.get("height", 1920))
    fps = int(p.get("fps", 30))
    g_band = p.get("graphics_band", {})
    g_top = int(g_band.get("top", 1140 if h >= w else 80))
    g_bot = int(g_band.get("bottom", 1500 if h >= w else 860))
    g_left = int(g_band.get("left", 0 if h >= w else 100))
    g_right = int(g_band.get("right", w if h >= w else w - 100))
    clip_path = g_band.get("clip_path") or f"inset({g_top}px {max(0, w - g_right)}px {max(0, h - g_bot)}px {g_left}px)"

    title_size = 56 if h >= w else 54
    cv_size = 50 if h >= w else 46
    clean_scenes = (scenes_js or "").strip()

    return f"""<!doctype html><html lang="en"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width={w}, height={h}"/>
<script src="assets/gsap.min.js"></script><script src="cues.js"></script>
<style>
@font-face{{font-family:'Marker';src:url('assets/fonts/PermanentMarker.woff2') format('woff2')}}
@font-face{{font-family:'Caveat';font-weight:700;src:url('assets/fonts/Caveat-700.woff2') format('woff2')}}
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{margin:0;width:{w}px;height:{h}px;overflow:hidden;background:transparent}}
#root{{position:absolute;inset:0}}
#stage{{position:absolute;inset:0;clip-path:{clip_path}}}
.g{{position:absolute;left:0;top:0;width:{w}px;height:{h}px;opacity:0;filter:drop-shadow(0 3px 0 rgba(0,0,0,.5)) drop-shadow(0 6px 14px rgba(0,0,0,.6))}}
.g svg{{overflow:visible;display:block}}
.ln{{fill:none;stroke-linecap:round;stroke-linejoin:round}}
.hw{{position:absolute;opacity:0;white-space:nowrap;color:#F4EFE6;font:400 {title_size}px/1.1 'Marker',cursive;text-shadow:0 3px 0 rgba(0,0,0,.5),0 6px 16px rgba(0,0,0,.65)}}
.cv{{font:700 {cv_size}px/1.1 'Caveat',cursive}}
</style></head><body>
<div id="root" data-composition-id="main" data-start="0" data-duration="{total_dur:.2f}" data-width="{w}" data-height="{h}" data-fps="{fps}">
<svg width="0" height="0" style="position:absolute">
 <defs>
  <filter id="chalk" x="-20%" y="-20%" width="140%" height="140%"><feTurbulence type="fractalNoise" baseFrequency="0.045" numOctaves="2" seed="3"/><feDisplacementMap in="SourceGraphic" scale="4"/></filter>
  <filter id="wob" x="-20%" y="-20%" width="140%" height="140%"><feTurbulence type="fractalNoise" baseFrequency="0.022" numOctaves="2" seed="11"/><feDisplacementMap in="SourceGraphic" scale="9"/></filter>
 </defs>
</svg>
<div id="stage"></div>
</div>
<script>
const C=window.CUES||{{}};
const W="#F4EFE6",L="#C8F560",T="#FF8A3D";
const stage=document.getElementById("stage");
const add=h=>stage.insertAdjacentHTML("beforeend",h);
let MK=0;
const pl=(d,w,c)=>`<path class="ln" d="${{d}}" stroke="${{c||W}}" stroke-width="${{w}}" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1"/>`+
 `<path class="ln" d="${{d}}" stroke="${{c||W}}" stroke-width="${{(w*.45).toFixed(1)}}" opacity=".55" transform="translate(1.6,-1.4)" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1"/>`;
const dl=(d,w,c)=>{{const id="mk"+(MK++);return `<mask id="${{id}}" maskUnits="userSpaceOnUse" x="-3000" y="-3000" width="7000" height="7000"><path class="mk" d="${{d}}" stroke="#fff" stroke-width="40" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1"/></mask>`+
 `<path class="ln" d="${{d}}" stroke="${{c||L}}" stroke-width="${{w}}" stroke-dasharray="0.1 15" mask="url(#${{id}})"/>`;}};
function draw(id,inner,filt){{add(`<div id="${{id}}" class="g"><svg width="{w}" height="{h}"><g filter="url(#${{filt||"chalk"}})">${{inner}}</g></svg></div>`);}}
function label(id,html,x,y,o){{o=o||{{}};add(`<div id="${{id}}" class="hw${{o.cv?" cv":""}}" data-layout-allow-safe-zone style="left:${{x}}px;top:${{y}}px;${{o.w?`width:${{o.w}}px;text-align:center;white-space:normal;`:""}}font-size:${{o.size||(o.cv?{cv_size}:{title_size})}}px;${{o.col?`color:${{o.col}};`:""}}">${{html}}</div>`);}}
const G=(x,y,s,inner)=>`<g transform="translate(${{x}},${{y}}) scale(${{s}})">${{inner}}</g>`;
const tl=gsap.timeline({{paused:true}});
const X=end=>Math.max(0,end-.28);
function out(sel,t,dy){{tl.to(sel,{{opacity:0,y:"+="+(dy||12),duration:.26,ease:"power2.in"}},t);}}
function lin(id,t,dur){{tl.set("#"+id,{{opacity:1}},t);tl.fromTo("#"+id,{{clipPath:"inset(-30% 100% -30% -4%)"}},{{clipPath:"inset(-30% -4% -30% -4%)",duration:dur||.4,ease:"power1.inOut"}},t);}}
function strokes(id,t,dur,st,ease){{const el=document.getElementById(id);tl.set(el,{{opacity:1}},t);const lns=el?el.querySelectorAll(".ln[pathLength]:not(.mk)"):[];if(lns.length)tl.to(lns,{{strokeDashoffset:0,duration:dur||.45,ease:ease||"power2.inOut",stagger:st||0}},t);}}
function dots(id,t,dur,st){{const el=document.getElementById(id);tl.set(el,{{opacity:1}},t);const mks=el?el.querySelectorAll(".mk"):[];if(mks.length)tl.to(mks,{{strokeDashoffset:0,duration:dur||.6,ease:"power2.inOut",stagger:st||0}},t);}}
function pop(id,t,o){{const el=document.getElementById(id);tl.set(el,{{opacity:1}},t);const lns=el?el.querySelectorAll(".ln[pathLength]:not(.mk)"):[];if(lns.length)tl.set(lns,{{strokeDashoffset:0}},t);tl.fromTo(el,{{scale:1.55,opacity:0,transformOrigin:o||"50% 50%"}},{{scale:1,opacity:1,duration:.3,ease:"power4.in"}},t);}}
function pulse(sel,o,t,s){{tl.to(sel,{{scale:s||1.08,transformOrigin:o||"50% 50%",duration:.14,ease:"power2.out"}},t);tl.to(sel,{{scale:1,duration:.35,ease:"back.out(3)"}},t+.14);}}
function shake(sel,t,a,n){{a=a||5;n=n||6;for(let i=0;i<n;i++)tl.to(sel,{{x:"+="+(i%2?-2*a:2*a),duration:.06,ease:"none"}},t+.06*i);tl.to(sel,{{x:"-="+(n%2?2*a:0),duration:.06}},t+.06*n);}}
function swap(a,b,t){{tl.to("#"+a,{{opacity:0,duration:.15}},t);lin(b,t+.1,.4);}}

{clean_scenes}

window.__timelines=window.__timelines||{{}};
window.__timelines["main"]=tl;
</script></body></html>"""


def build_fallback_scenes_js(
    timeline_json: dict[str, Any],
    edl_json: dict[str, Any],
    preset: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Build deterministic animated cue-synced chalk graphics fallback so #stage is never blank."""
    p = preset or {}
    w = int(p.get("width", 1080))
    h = int(p.get("height", 1920))
    g_band = p.get("graphics_band", {})
    g_top = int(g_band.get("top", 1140 if h >= w else 80))
    g_bot = int(g_band.get("bottom", 1500 if h >= w else 860))
    total = float(timeline_json.get("total", 45.0) or 45.0)
    title = str((edl_json or {}).get("title") or "KEY TAKEAWAY").strip()
    beats = _extract_hyperframes_beats(timeline_json, edl_json)

    safe_right = 850 if h >= w else w - 90
    label_w = max(320, safe_right - 90)
    cx = (90 + safe_right) // 2
    y_title = g_top + 48
    y_sub = min(g_bot - 90, y_title + 92)
    bx0, bx1 = max(80, cx - 340), min(safe_right, cx + 340)
    by0, by1 = g_top + 28, min(g_bot - 28, y_title + 170)

    cues_json: dict[str, Any] = {
        "total": round(total, 2),
        "title": title,
        "beats": [],
    }
    js_lines = [
        f"const TITLE = {json.dumps(title)};",
        f'draw("fbFrame", pl("M{bx0} {by0 + 18} L{bx0} {by0} L{bx0 + 36} {by0} M{bx1 - 36} {by0} L{bx1} {by0} L{bx1} {by0 + 18} M{bx0} {by1 - 18} L{bx0} {by1} L{bx0 + 36} {by1} M{bx1 - 36} {by1} L{bx1} {by1} L{bx1} {by1 - 18}", 7, T));',
        f'label("fbTitle", TITLE, 90, {y_title}, {{w: {label_w}, size: 52, col: W}});',
        'tl.set("#fbFrame,#fbTitle", {opacity: 1}, 0);',
        'tl.set("#fbFrame .ln[pathLength]", {strokeDashoffset: 0}, 0);',
    ]

    # Group beats chronologically cleanly (non-overlapping card swaps across the safe band)
    last_end = min(3.2, total)
    if last_end > 2.2:
        js_lines.append(f'pulse("#fbTitle", "{cx}px {y_title + 28}px", {min(1.5, last_end - 0.8):.2f}, 1.05);')
    js_lines.append(f'out("#fbFrame,#fbTitle", {max(0.4, last_end - 0.3):.2f});')
    for idx, b in enumerate(beats[:10]):
        s = max(last_end if idx > 0 else 0.0, float(b["start"]))
        e = min(total, float(b["end"]))
        if idx == 0:
            s = max(0.0, float(b["start"]))
            if e <= 3.2:
                continue
            s = 3.25
        if e - s < 1.2:
            continue
        cue_text = b["note"] or b["say"][:48]
        # Clean prefix like "Setup: "
        if ":" in cue_text:
            cue_text = cue_text.split(":", 1)[-1].strip()
        cue_text = cue_text[:44].strip()
        if not cue_text:
            continue
        bid = f"beat{idx}"
        lid = f"lbl{idx}"
        col_var = "T" if idx % 2 == 0 else "L"
        cues_json["beats"].append({"id": bid, "start": round(s, 2), "end": round(e, 2), "text": cue_text})
        js_lines.extend([
            f'draw("{bid}", pl("M{cx - 240} {y_sub + 58} Q{cx} {y_sub + 44} {cx + 240} {y_sub + 58}", 7, {col_var}));',
            f'label("{lid}", {json.dumps(cue_text)}, 90, {y_title + 26}, {{cv: 1, w: {label_w}, size: 52, col: {col_var}}});',
            f'strokes("{bid}", {s:.2f}, 0.45);',
            f'lin("{lid}", {s + 0.08:.2f}, 0.38);',
        ])
        if e - s > 2.2:
            mid_t = round((s + e) / 2.0, 2)
            js_lines.append(f'pulse("#{lid}", "{cx}px {y_title + 52}px", {mid_t:.2f}, 1.06);')
        js_lines.append(f'out("#{bid},#{lid}", {max(s + 0.5, e - 0.28):.2f});')
        last_end = e

    return cues_json, "\n".join(js_lines)


def build_hyperframes_prompt(
    timeline_json: dict[str, Any],
    edl_json: dict[str, Any],
    preset: dict[str, Any] | None = None,
    word_cues_text: str = "",
) -> str:
    """Build a concise, focused prompt returning cues_json + scenes_js for our HyperFrames chalk scaffold."""
    p = preset or {}
    w = int(p.get("width", 1080))
    h = int(p.get("height", 1920))
    fps = int(p.get("fps", 30))
    total_dur = float(timeline_json.get("total", 60.0) or 60.0)
    g_band = p.get("graphics_band", {"top": 1140, "bottom": 1500})
    g_top = int(g_band.get("top", 1140 if h >= w else 80))
    g_bot = int(g_band.get("bottom", 1500 if h >= w else 860))
    safe_right = 980 if h >= w else w - 80
    safe_y_bot = min(g_bot - 40, 1440 if h >= w else g_bot)
    c_band = p.get("caption_band", {"top": 1500, "bottom": 1640})

    beats = _extract_hyperframes_beats(timeline_json, edl_json)
    beats_str = json.dumps(beats, indent=2)
    title = str((edl_json or {}).get("title") or "Motion Graphics").strip()
    word_cues_block = f"\nSPOKEN WORD TIMESTAMPS (timeline seconds):\n```\n{word_cues_text[:5000]}\n```\n" if word_cues_text else ""

    return f"""Do NOT read external workspace files, do NOT read skills, and do NOT run shell commands. All required inputs and helper APIs are provided below. Output JSON only.

You are designing hand-drawn sketchy chalkboard motion graphics for a {w}x{h} @ {fps}fps video ({total_dur:.2f}s total).
Video Title: {json.dumps(title)}

STAGE BOUNDS & PALETTE:
- Graphics safe band: x=80..{safe_right}, y={g_top}..{safe_y_bot}. Keep EVERY SVG stroke and HTML label strictly inside this box!
- Caption keep-out zone: y={c_band.get('top', 1500)}..{c_band.get('bottom', 1640)} (NOTHING may extend into captions).
- Already-defined JS palette constants: W="#F4EFE6" (warm chalk white), L="#C8F560" (dotted/highlight lime), T="#FF8A3D" (accent orange).
- Do NOT put colored rectangles/boxes, opaque cards, or drop-shadow backgrounds behind illustrations; use clean hand-drawn SVG outlines, arrows, figures, props, and handwritten labels (`Marker` / `Caveat`).

ALREADY-SCAFFOLDED JS HELPERS YOU MUST CALL IN `scenes_js`:
- `#stage`, `tl` (paused GSAP timeline), `C` (`window.CUES`), `W`, `L`, `T` are already initialized. DO NOT redeclare `tl`, `C`, `W`, `L`, `T`, `pl`, `dl`, `draw`, `label`, `G`, or HTML `<head>` tags!
- `pl(pathD, strokeWidth, color)` -> sketchy double-pass chalk SVG stroke (`pathLength="1"`).
- `dl(pathD, strokeWidth, color)` -> masked dotted lime/chalk SVG line (`.mk` inside `<mask>`).
- `draw(id, svgMarkup, filterName)` -> mounts `<div id="{{id}}" class="g">` containing an SVG (`filterName`: `"chalk"` default or `"wob"`).
- `label(id, htmlText, x, y, opts)` -> adds `.hw` text at `(x, y)` inside `x=80..{safe_right}, y={g_top}..{safe_y_bot}` (`opts`: `{{cv:1, size:52, w:400, col:T}}`). Note: Do NOT pass `rot` CSS transform on targets animated by GSAP! Keep all transforms inside GSAP (`rotation`, `x`, `y`, `scale`).
- `G(x, y, scale, innerSvg)` -> groups `<g transform="translate(x,y) scale(scale)">`.
- Timeline helpers:
  - `strokes(id, t, dur, stagger)` -> animates `#id` solid `pl()` strokes drawing in at `t`s.
  - `dots(id, t, dur, stagger)` -> animates `#id` dotted `dl()` lines drawing in at `t`s (wait: keep `dots` start after/separate cleanly, zero conflicting `.mk` tweens).
  - `lin(id, t, dur)` -> wipes text label left-to-right at `t`s.
  - `pop(id, t, transformOrigin)` -> crisp impact stamp pop-in at `t`s (works on labels or SVGs).
  - `pulse(sel, transformOrigin, t, scale)` / `shake(sel, t, amp, n)` / `swap(a, b, t)`.
  - `out(selector, t, dy)` -> fades & drops elements cleanly out BEFORE next beat starts!

CHOREOGRAPHY REQUIREMENTS:
1. Frame 0 visibility: Ensure the opening hook title/icon is ALREADY visibly drawn at `t=0` (`tl.set("#hookId", {{opacity:1}}, 0); tl.set("#hookId .ln[pathLength]", {{strokeDashoffset:0}}, 0);`) so frame 0 is never blank (`first_frame_empty`).
2. Bespoke visual metaphors per beat: For EACH narrative beat (`hook`, `setup`, `twist`, `climax`, `loop`), draw recognizable simple props matching that beat's `say` & `note` (e.g., pizza slice, masks, scales, arrows, figures, checkmarks, comparison diagrams — never blank screens or single static text!).
3. Continuous motion (`dead_stretch` guard): Never leave >2.5s without visual change! For any beat >= 2.6s long, schedule a mid-beat `pulse(...)`, `dots(...)`, or secondary `lin(...)` halfway through before `out(...)`.
4. Strict non-overlap cleanup: Call `out("#prevIds...", endTime - 0.28)` before the next beat arrives so elements never stack on top of old elements.
5. Keep `scenes_js` clean, bug-free ES6 JS (around 45-130 lines total, zero syntax errors).

MERGED TIMELINE BEATS & VISUAL NOTES:
{beats_str}
{word_cues_block}
OUTPUT JSON SCHEMA:
- `cues_json`: object containing `"total": {total_dur:.2f}` plus named timestamp objects for each beat.
- `scenes_js`: JavaScript body calling `draw(...)`, `label(...)`, `strokes(...)`, `dots(...)`, `lin(...)`, `pop(...)`, `out(...)` referenced cleanly against `C.*` cues.

Output valid JSON with keys `cues_json` and `scenes_js`. Output JSON only."""


def build_hyperframes_revision_prompt(
    issues: list[str],
    fixes_needed: list[str],
    cues_json: dict[str, Any],
    scenes_js: str,
    timeline_json: dict[str, Any],
    edl_json: dict[str, Any],
    preset: dict[str, Any] | None = None,
) -> str:
    """Build a focused revision prompt after hyperframes check or Vision Art Director flags issues."""
    p = preset or {}
    w = int(p.get("width", 1080))
    h = int(p.get("height", 1920))
    g_band = p.get("graphics_band", {"top": 1140 if h >= w else 80, "bottom": 1500 if h >= w else 860})
    safe_right = 980 if h >= w else w - 80
    safe_y_bot = min(int(g_band.get("bottom", 1500)), 1440 if h >= w else int(g_band.get("bottom", 860)))
    beats = _extract_hyperframes_beats(timeline_json, edl_json)
    return f"""Do NOT read external workspace files or run shell commands. Output JSON only.

The Vision Art Director / HyperFrames DOM audit REJECTED the current motion graphics for failing QA checks.
Fix ALL reported issues and return updated `cues_json` + `scenes_js`.

REPORTED ISSUES:
{json.dumps(issues, indent=2)}

REQUIRED FIXES:
{json.dumps(fixes_needed, indent=2)}

LAYOUT & CONSTRAINTS:
- Safe graphics band: x=80..{safe_right}, y={g_band.get('top', 1140)}..{safe_y_bot}.
- Key hook element must already be visible at t=0 (`tl.set(..., {{opacity:1}}, 0)` + `strokeDashoffset:0`).
- No static stretch >2.5s: add a mid-beat `pulse(...)`/`dots(...)` inside long beats.
- Every beat must animate visibly inside `#stage` using `draw`, `label`, `pl`, `dl`, `strokes`, `lin`, `pop`, `out` (`W`, `L`, `T`, `tl`, `C` already defined).

TIMELINE BEATS:
{json.dumps(beats, indent=2)}

CURRENT cues_json:
{json.dumps(cues_json, indent=2)}

CURRENT scenes_js:
```javascript
{scenes_js}
```

Output valid JSON with updated `cues_json` and `scenes_js`. Output JSON only."""


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
    snapshot_image_path: str,
    cues_json_path: str = "",
    preset: dict[str, Any] | None = None,
    telemetry_text: str = "",
    beats_summary: str = "",
) -> str:
    """Build a fast single-image AGY Vision Art Director critique prompt."""
    p = preset or {}
    g_top = p.get("graphics_band", {}).get("top", 1140)
    g_bot = p.get("graphics_band", {}).get("bottom", 1500)
    c_top = p.get("caption_band", {}).get("top", 1500)
    c_bot = p.get("caption_band", {}).get("bottom", 1640)

    tel_block = f"\nPRE-COMPUTED KEYFRAME ALPHA/BBOX TELEMETRY:\n{telemetry_text}\n" if telemetry_text else ""
    beats_block = f"\nSCHEDULED BEATS:\n{beats_summary}\n" if beats_summary else ""
    cues_note = f"- Cue timing reference file (optional read only if needed): {cues_json_path}\n" if cues_json_path else ""

    return f"""Inspect ONLY the single contact-sheet image below. Do NOT run shell commands, do NOT list directories, and do NOT open HTML/JS source files. Output valid JSON immediately.

You are a creative director and visual QA lead reviewing rendered motion graphics keyframes.
Contact-sheet image containing all sampled timestamps:
- {snapshot_image_path}
{cues_note}{tel_block}{beats_block}
CRITICAL CHECKS:
1. Non-Blank Animated Presence:
   - Keyframes MUST contain crisp, visible hand-drawn chalk graphics/labels (a completely blank frame-00 or multiple empty frames = `passes: false`). Note: slightly lower opacity right near a beat fade-out boundary is normal and should NOT fail QA.
2. Layout Safe Area:
   - Centered graphics cleanly within x=40..1040, y={g_top}..{g_bot} are valid.
   - Caption keep-out band y={c_top}..{c_bot} must remain clear (do NOT fail minor bbox touches at y={g_bot}, only true spills below y={c_top + 10}).
3. Visual Aesthetic:
   - Hand-drawn sketchy chalk aesthetic (#F4EFE6 warm white, #FF8A3D warm accent, #C8F560 lime).
   - High legibility, zero colliding/stacked text from previous beats, zero clipped/cut-off labels.
4. Set `passes: true` if illustrations are legible, on-style chalk sketches cleanly framing beats inside `#stage`; set `passes: false` ONLY for blank renders, overlapping stale beat elements, unreadable text, or true caption-band spills, listing concrete fixes in `fixes_needed`.

Return valid JSON conforming to the vision critique schema. Output JSON only."""


def build_cover_prompt(
    candidate_frames: list[str],
    title_options: list[str],
    preset: dict[str, Any] | None = None,
    has_sticker: bool = False,
    contact_sheet_path: str | None = None,
) -> str:
    """Build the prompt for choosing cover frames and title layouts."""
    frames_list = "\n".join(f"- #{i + 1}: {f}" for i, f in enumerate(candidate_frames))
    titles_list = "\n".join(f"- {t}" for t in title_options)
    sticker_note = (
        "- A transparent graphic sticker extracted from the motion overlay will be placed at y~1235 (Layout A) or top y~170 (Layout B)."
        if has_sticker
        else "- No graphics sticker is overlaid on covers; use Layout A (title top y~190) or Layout B (title lower chest y~1200) cleanly framing the speaker's face!"
    )
    inspect_block = (
        f"Inspect ONLY this single labeled contact-sheet grid image (tiles #1..#{len(candidate_frames)}) instead of opening every PNG:\n- {contact_sheet_path}\n\nExact frame path mapping for your JSON output `frame` field:\n{frames_list}"
        if contact_sheet_path
        else f"CANDIDATE FRAMES (uncaptioned, 1080x1920):\n{frames_list}"
    )

    return f"""Inspect ONLY the image(s) listed below; do NOT run shell commands or list directories. Output valid JSON immediately.

You are a thumbnail designer optimizing for maximum Click-Through Rate (CTR) on TikTok, Reels, and Shorts.

{inspect_block}

HOOK TITLES / CONTEXT:
{titles_list}

RULES FOR COVERS:
- Layout A: Title top (y~190). Best when speaker's eyes/face sit lower half or clear of top band y=180..520.
- Layout B: Lower headline (y~1200, below chin/mouth). Pick ONLY frames where the speaker's face is high enough that y>=1200 sits cleanly on chest/torso and NEVER covers eyes, nose, or mouth! If unsure, use Layout A and pick a crisp eyes-open frame where forehead/top margin has space.
{sticker_note}
- Titles must be <=3 short lines, max 2-4 words per line, formatted as `[[["WORD", "w"], [" KEYWORD", "p"]], ...]` so `cover.py` wraps cleanly inside width 940px.
- Available colors: 'w' (white [255,255,255]), 'p' (pink [255,79,160]), 'b' (blue [61,107,255]).
- Pick two DIFFERENT expressions/frames for Cover A and Cover B (A = punch angle, B = curiosity angle).
- Set `frame` in your JSON to the exact path string (or filename `cand_XX.png`) from the mapping above.

Return valid JSON conforming to the cover selector schema. Output JSON only."""


def build_descriptions_prompt(
    final_transcript: str,
    script_text: str,
    summary: str,
    story_prompt: str = "",
) -> str:
    """Build the prompt for generating 5 distinct TikTok/Reels description options grounded strictly in the actual video."""
    angle_block = f"\nCREATOR DIRECTION / SERIES CONTEXT:\n{story_prompt.strip()}\n" if story_prompt and story_prompt.strip() else ""

    return f"""Do NOT inspect files or run shell commands. Work strictly from the provided video transcript and script below. Output valid JSON only.

You are a social video copywriter crafting 5 angle-distinct post descriptions (captions + hashtags + pinned comments) strictly grounded in what the speaker actually filmed and said.

VIDEO TITLE / SUMMARY:
{summary}
{angle_block}
APPROVED SCRIPT TEXT:
```
{script_text[:6000]}
```

FINAL TRANSCRIPT (WHAT IS ACTUALLY HEARD):
```
{final_transcript[:4000]}
```

GROUNDING RULES:
- Never invent filming locations, weather/time-of-day ("night walk"), episode numbers, or creator backstory unless explicitly stated in the transcript or CREATOR DIRECTION above!
- Respect hypotheticals: if the speaker used a thought experiment, keep it framed as an example/idea rather than a false personal confession.
- Sound authentic, conversational, and concise (zero spammy hashtags like #fyp).

Write FIVE distinct options in the JSON fields:
- `hook`: Direct question hook (90-160 chars + 3-5 topical hashtags + 1-line pinned comment)
- `teaser`: Story teaser / curiosity gap
- `confessional`: Honest personal reflection grounded in the actual speech
- `debate`: Balanced debate-bait inviting real viewer stories in comments
- `minimal`: Clean 1-2 sentence minimal caption
- `recommendation`: Which option (1-5) to pick and a 1-sentence reason why.

Return valid JSON conforming to DESCRIPTIONS_SCHEMA. Output JSON only."""
