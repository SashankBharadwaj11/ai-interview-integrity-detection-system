# src/analysis/scoring.py

from typing import Dict, List, Optional, Union


def compute_video_score(summary: Dict, alerts: List[Dict]) -> float:
    weights = {
        "FACE_MISSING": 1.0,
        "GAZE_AWAY": 1.0,
        "MOUTH_MOVEMENT": 0.5,
        "MULTI_FACE": 3.0,
        "OBJECT_DETECTED": 4.0,
    }

    total_penalty = 0.0
    for a in alerts:
        t = a.get("type", "")
        total_penalty += weights.get(t, 1.0)

    duration_sec = summary.get("duration_seconds", 0.0)
    duration_min = max(duration_sec / 60.0, 1e-6)
    penalty_rate = total_penalty / duration_min

    raw = max(0.0, 100.0 - penalty_rate * 2.0)
    return min(100.0, raw)


def compute_audio_score(
    audio_summary: Optional[Union[Dict, float, int]]
) -> float:
    """
    Turn speaker consistency metrics into a 0–100 audio integrity score.

    Accepts:
      - dict from SpeakerConsistencyAnalyzer.analyze(...)
      - or a float/int if a score was precomputed already.
    """
    if audio_summary is None:
        return 100.0

    # If it's already a numeric score, just clamp and return
    if isinstance(audio_summary, (int, float)):
        return max(0.0, min(100.0, float(audio_summary)))

    # If it's not a dict at this point, bail out safely
    if not isinstance(audio_summary, dict):
        return 100.0

    score_01 = audio_summary.get("speaker_consistency_score", None)
    if score_01 is None:
        return 100.0

    # map 0–1 → 0–100
    base = float(score_01) * 100.0

    # if a potential speaker change was detected, reduce score
    if audio_summary.get("speaker_change_flag"):
        base -= 20.0  # tune as needed

    return max(0.0, min(100.0, base))


def compute_overall_score(video_score: float, audio_score: float) -> float:
    return 0.7 * video_score + 0.3 * audio_score
