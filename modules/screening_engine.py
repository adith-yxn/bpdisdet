"""
screening_engine.py
────────────────────
Combines facial-affect metrics and linguistic analysis into a unified
bipolar-spectrum screening score.  Follows an evidence-informed
multi-modal fusion approach.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from modules.facial_analysis import FacialSession
from modules.text_analysis import TextAnalysisResult


# ── Weights for multimodal fusion ─────────────────────────────────────────────
# Empirically inspired by multi-modal emotion research; not clinically validated.
MODAL_WEIGHTS = {
    "facial": 0.40,
    "text":   0.60,          # text is slightly more diagnostic
}

RISK_THRESHOLDS = {
    "high":     70,
    "moderate": 45,
    "low":      20,
}

RISK_COLORS = {
    "high":     "#e05252",
    "moderate": "#e09a2f",
    "low":      "#4caf7d",
    "minimal":  "#5b9bd5",
}


@dataclass
class ScreeningResult:
    # -- Composite scores ------------------------------------------------
    composite_mania_score:       float = 0.0
    composite_depression_score:  float = 0.0
    composite_mixed_score:       float = 0.0
    affective_instability_index: float = 0.0

    # -- Classification -------------------------------------------------
    overall_risk:        str = "minimal"     # minimal / low / moderate / high
    dominant_state:      str = "euthymic"
    confidence_pct:      float = 0.0

    # -- Source data ----------------------------------------------------
    facial_session:   Optional[FacialSession]       = None
    text_results:     list[TextAnalysisResult]       = field(default_factory=list)

    # -- Narrative & guidance -------------------------------------------
    clinical_summary:    str  = ""
    recommendations:     list = field(default_factory=list)
    red_flags:           list = field(default_factory=list)

    # -- Metadata -------------------------------------------------------
    timestamp:       float = field(default_factory=time.time)
    session_id:      str   = ""
    has_facial_data: bool  = False
    has_text_data:   bool  = False


def compute_screening_result(
    facial_session: Optional[FacialSession] = None,
    text_results:   Optional[list[TextAnalysisResult]] = None,
    patient_info:   Optional[dict] = None,
) -> ScreeningResult:
    """
    Fuse facial and text modalities into a unified screening result.
    Either modality can be absent; weights are re-normalised accordingly.
    """
    import uuid
    result = ScreeningResult(
        session_id=str(uuid.uuid4())[:8].upper(),
        facial_session=facial_session,
        text_results=text_results or [],
    )

    # ── 1. Extract per-modality scores ────────────────────────────────
    f_mania = f_depr = f_mixed = f_instab = 0.0
    t_mania = t_depr = t_mixed = 0.0
    t_confidence = 50.0

    w_f = MODAL_WEIGHTS["facial"]
    w_t = MODAL_WEIGHTS["text"]

    if facial_session and len(facial_session.frames) >= 3:
        result.has_facial_data = True
        f_mania  = facial_session.mania_score
        f_depr   = facial_session.depression_score
        f_mixed  = facial_session.mixed_state_score
        f_instab = min(100, facial_session.affective_instability * 20)
    else:
        w_f = 0.0
        w_t = 1.0

    if text_results:
        result.has_text_data = True
        t_mania  = sum(r.mania_score    for r in text_results) / len(text_results)
        t_depr   = sum(r.depression_score for r in text_results) / len(text_results)
        t_mixed  = sum(r.mixed_score    for r in text_results) / len(text_results)
        t_confidence = sum(r.confidence for r in text_results) / len(text_results)
    else:
        w_t = 0.0
        w_f = 1.0

    if w_f + w_t == 0:
        return result  # nothing to compute

    norm = w_f + w_t

    # ── 2. Composite scores ───────────────────────────────────────────
    result.composite_mania_score      = (w_f * f_mania  + w_t * t_mania)  / norm
    result.composite_depression_score = (w_f * f_depr   + w_t * t_depr)   / norm
    result.composite_mixed_score      = (w_f * f_mixed  + w_t * t_mixed)  / norm
    result.affective_instability_index = min(100,
        (f_instab * w_f + result.composite_mixed_score * w_t * 0.5) / norm)

    # ── 3. Dominant state ─────────────────────────────────────────────
    max_score = max(result.composite_mania_score,
                    result.composite_depression_score,
                    result.composite_mixed_score)

    if result.composite_mixed_score == max_score and max_score > 30:
        result.dominant_state = "mixed"
    elif result.composite_mania_score == max_score and max_score > 25:
        result.dominant_state = "manic"
    elif result.composite_depression_score == max_score and max_score > 25:
        result.dominant_state = "depressive"
    else:
        result.dominant_state = "euthymic"

    # ── 4. Overall risk ───────────────────────────────────────────────
    peak = max(result.composite_mania_score,
               result.composite_depression_score,
               result.composite_mixed_score,
               result.affective_instability_index)

    if peak >= RISK_THRESHOLDS["high"]:
        result.overall_risk = "high"
    elif peak >= RISK_THRESHOLDS["moderate"]:
        result.overall_risk = "moderate"
    elif peak >= RISK_THRESHOLDS["low"]:
        result.overall_risk = "low"
    else:
        result.overall_risk = "minimal"

    # Override with text risk if higher
    for tr in text_results:
        if tr.risk_level == "high":
            result.overall_risk = "high"

    # ── 5. Confidence ─────────────────────────────────────────────────
    face_conf = 60.0 if result.has_facial_data else 0.0
    text_conf = t_confidence if result.has_text_data else 0.0
    inputs    = (1 if result.has_facial_data else 0) + (1 if result.has_text_data else 0)
    result.confidence_pct = (face_conf + text_conf) / max(inputs * 100, 1) * 100 if inputs else 0.0
    result.confidence_pct = min(85.0, result.confidence_pct)  # cap at 85% — screening only

    # ── 6. Red flags ──────────────────────────────────────────────────
    for tr in text_results:
        for phrase in tr.key_phrases:
            if phrase.startswith("⚠️"):
                result.red_flags.append(phrase)
        for rec in tr.recommendations:
            if "URGENT" in rec:
                result.red_flags.append(rec)

    if result.affective_instability_index > 60:
        result.red_flags.append("High affective instability detected in facial expression.")

    # ── 7. Clinical summary ───────────────────────────────────────────
    modalities = []
    if result.has_facial_data:
        modalities.append("facial affect analysis")
    if result.has_text_data:
        modalities.append("linguistic analysis")

    result.clinical_summary = (
        f"Multimodal screening using {' and '.join(modalities)} "
        f"suggests a **{result.dominant_state.upper()}** affective pattern "
        f"with a **{result.overall_risk.upper()} risk** signal. "
        f"Mania indicators: {result.composite_mania_score:.0f}/100. "
        f"Depression indicators: {result.composite_depression_score:.0f}/100. "
        f"Affective instability index: {result.affective_instability_index:.0f}/100. "
        f"Confidence: {result.confidence_pct:.0f}% (screening only — "
        f"not a clinical diagnosis)."
    )

    # ── 8. Recommendations ────────────────────────────────────────────
    result.recommendations = _build_recommendations(result, patient_info)

    return result


def _build_recommendations(result: ScreeningResult, patient_info: Optional[dict]) -> list:
    recs = []

    # Crisis first
    if result.red_flags:
        recs.append("🔴 IMMEDIATE: Potential crisis indicators found. Contact a mental health "
                    "professional or a crisis helpline without delay.")

    # Risk-level guidance
    if result.overall_risk == "high":
        recs += [
            "Schedule an urgent psychiatric evaluation within 24–48 hours.",
            "Share these screening results with your mental health provider.",
            "Avoid making major decisions until evaluated by a clinician.",
        ]
    elif result.overall_risk == "moderate":
        recs += [
            "Book an appointment with a psychiatrist or psychologist within the next week.",
            "Track mood, sleep, and energy levels daily using a mood diary.",
        ]
    elif result.overall_risk == "low":
        recs += [
            "Consult your general practitioner about your mood patterns.",
            "Consider a structured mood monitoring app (e.g. eMoods).",
        ]
    else:
        recs.append("Continue monitoring mood wellbeing; no urgent action indicated.")

    # State-specific
    if result.dominant_state == "manic":
        recs += [
            "Prioritise sleep regularity — disrupted sleep can trigger/worsen mania.",
            "Limit stimulants (caffeine, alcohol) and avoid high-stimulation environments.",
        ]
    elif result.dominant_state == "depressive":
        recs += [
            "Maintain a regular daily routine including light physical activity.",
            "Reach out to a trusted person in your support network.",
        ]
    elif result.dominant_state == "mixed":
        recs += [
            "Mixed states carry elevated risk — clinical evaluation is strongly recommended.",
            "Avoid abrupt changes in sleep schedule.",
        ]

    # SDG-3 resource note
    recs.append(
        "💡 Free/low-cost resources: iCall (India): 9152987821 | Vandrevala Foundation: 1860-2662-345 "
        "| NIMHANS Helpline: 080-46110007 | WHO mhGAP resources: who.int/mhgap"
    )

    return recs