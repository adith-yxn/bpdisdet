"""
screening_engine.py  ·  bpdisdet v2
═════════════════════════════════════
Three-modality fusion engine:
  Facial Affect (25%) + Linguistic Analysis (40%) + Questionnaire (35%)
Weights re-normalise when modalities are missing.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from modules.facial_analysis  import FacialSession
from modules.text_analysis    import TextAnalysisResult
from modules.questionnaire    import QuestionnaireResult


MODAL_WEIGHTS = {"facial": 0.25, "text": 0.40, "questionnaire": 0.35}

RISK_THRESHOLDS = {"high": 65, "moderate": 40, "low": 20}


@dataclass
class ScreeningResult:
    # Composite scores
    composite_mania:       float = 0.0
    composite_depression:  float = 0.0
    composite_mixed:       float = 0.0
    affective_instability: float = 0.0

    # Classification
    overall_risk:     str   = "minimal"
    dominant_state:   str   = "euthymic"
    confidence_pct:   float = 0.0

    # Source data
    facial_session:       Optional[FacialSession]       = None
    text_results:         list[TextAnalysisResult]       = field(default_factory=list)
    questionnaire_result: Optional[QuestionnaireResult]  = None

    # Flags
    has_facial:       bool = False
    has_text:         bool = False
    has_questionnaire:bool = False

    # Narrative
    clinical_summary:  str  = ""
    recommendations:   list = field(default_factory=list)
    red_flags:         list = field(default_factory=list)

    # Metadata
    timestamp:   float = field(default_factory=time.time)
    session_id:  str   = field(default_factory=lambda: str(uuid.uuid4())[:8].upper())

    # Per-modality breakdowns (for radar chart)
    modality_scores: dict = field(default_factory=dict)


def compute_screening_result(
    facial_session:       Optional[FacialSession]      = None,
    text_results:         Optional[list[TextAnalysisResult]] = None,
    questionnaire_result: Optional[QuestionnaireResult] = None,
    patient_info:         Optional[dict]               = None,
) -> ScreeningResult:

    result = ScreeningResult(
        facial_session=facial_session,
        text_results=text_results or [],
        questionnaire_result=questionnaire_result,
    )

    # ── Per-modality scores ───────────────────────────────────────────
    wf = wt = wq = 0.0
    f_mania = f_depr = f_mixed = f_inst = 0.0
    t_mania = t_depr = t_mixed = t_conf = 0.0
    q_mania = q_depr = q_mixed = 0.0

    if facial_session and len(facial_session.frames) >= 2:
        result.has_facial = True
        wf      = MODAL_WEIGHTS["facial"]
        f_mania = facial_session.mania_score
        f_depr  = facial_session.depression_score
        f_mixed = facial_session.mixed_state_score
        f_inst  = min(100, facial_session.affective_instability * 25)

    if text_results:
        result.has_text = True
        wt      = MODAL_WEIGHTS["text"]
        t_mania = sum(r.mania_score    for r in text_results) / len(text_results)
        t_depr  = sum(r.depression_score for r in text_results) / len(text_results)
        t_mixed = sum(r.mixed_score    for r in text_results) / len(text_results)
        t_conf  = sum(r.confidence     for r in text_results) / len(text_results)

    if questionnaire_result:
        result.has_questionnaire = True
        wq      = MODAL_WEIGHTS["questionnaire"]
        q_mania = questionnaire_result.mdq_scaled
        q_depr  = questionnaire_result.phq9_scaled
        q_mixed = questionnaire_result.als_scaled

    total_w = wf + wt + wq
    if total_w == 0:
        return result

    # ── Composite scores ──────────────────────────────────────────────
    result.composite_mania      = (wf*f_mania + wt*t_mania + wq*q_mania)      / total_w
    result.composite_depression = (wf*f_depr  + wt*t_depr  + wq*q_depr)       / total_w
    result.composite_mixed      = (wf*f_mixed + wt*t_mixed + wq*q_mixed*0.7)   / total_w
    result.affective_instability = (
        f_inst * wf + result.composite_mixed * wt * 0.4 +
        (questionnaire_result.als_scaled if questionnaire_result else 0) * wq
    ) / total_w

    # ── Dominant state ────────────────────────────────────────────────
    peak_m = result.composite_mania
    peak_d = result.composite_depression
    peak_x = result.composite_mixed

    if peak_x == max(peak_m, peak_d, peak_x) and peak_x > 28:
        result.dominant_state = "mixed"
    elif peak_m == max(peak_m, peak_d, peak_x) and peak_m > 22:
        result.dominant_state = "manic"
    elif peak_d == max(peak_m, peak_d, peak_x) and peak_d > 22:
        result.dominant_state = "depressive"
    else:
        result.dominant_state = "euthymic"

    # ── Overall risk ──────────────────────────────────────────────────
    peak = max(result.composite_mania, result.composite_depression,
               result.composite_mixed, result.affective_instability)

    if   peak >= RISK_THRESHOLDS["high"]:     result.overall_risk = "high"
    elif peak >= RISK_THRESHOLDS["moderate"]: result.overall_risk = "moderate"
    elif peak >= RISK_THRESHOLDS["low"]:      result.overall_risk = "low"
    else:                                      result.overall_risk = "minimal"

    # Override from text or questionnaire high-risk flags
    for tr in result.text_results:
        if tr.risk_level == "high" or tr.suicidal_flag:
            result.overall_risk = "high"
    if questionnaire_result and questionnaire_result.phq9_safety_flag:
        result.overall_risk = "high"

    # ── Confidence ────────────────────────────────────────────────────
    n = sum([result.has_facial, result.has_text, result.has_questionnaire])
    base = ((60 if result.has_facial else 0) +
            (t_conf if result.has_text else 0) +
            (72 if result.has_questionnaire else 0))
    result.confidence_pct = min(88.0, base / max(n * 100, 1) * 100)

    # ── Red flags ─────────────────────────────────────────────────────
    for tr in result.text_results:
        for kp in tr.key_phrases:
            if kp.startswith("⚠"):
                result.red_flags.append(f"[Text] {kp}")
        if tr.suicidal_flag:
            result.red_flags.append("[Text] Suicidal ideation markers detected in written text.")

    if questionnaire_result and questionnaire_result.phq9_safety_flag:
        result.red_flags.append("[Questionnaire] PHQ-9 item 9 (suicidal ideation) endorsed.")
    if result.affective_instability > 60:
        result.red_flags.append("[Facial] High affective instability index detected.")

    # ── Modality scores for radar ─────────────────────────────────────
    result.modality_scores = {
        "Facial Mania":    round(f_mania, 1),
        "Facial Depr":     round(f_depr,  1),
        "Facial Mixed":    round(f_mixed, 1),
        "Text Mania":      round(t_mania, 1),
        "Text Depr":       round(t_depr,  1),
        "Text Mixed":      round(t_mixed, 1),
        "MDQ (Mania)":     round(q_mania, 1),
        "PHQ-9 (Depr)":    round(q_depr,  1),
        "ALS (Lability)":  round(q_mixed, 1),
    }

    # ── Clinical summary ──────────────────────────────────────────────
    modalities_used = []
    if result.has_facial:        modalities_used.append("facial affect analysis")
    if result.has_text:          modalities_used.append("linguistic analysis")
    if result.has_questionnaire: modalities_used.append("validated questionnaires (MDQ-7, PHQ-9, ALS-SF)")

    result.clinical_summary = (
        f"Three-modality screening using {', '.join(modalities_used)} suggests a "
        f"**{result.dominant_state.upper()}** affective pattern with **{result.overall_risk.upper()} RISK** signal. "
        f"Composite mania index: {result.composite_mania:.0f}/100. "
        f"Composite depression index: {result.composite_depression:.0f}/100. "
        f"Affective instability index: {result.affective_instability:.0f}/100. "
        f"Screening confidence: {result.confidence_pct:.0f}% "
        f"(maximum 88% — this is a screening tool, not a clinical diagnosis)."
    )

    # ── Recommendations ────────────────────────────────────────────────
    result.recommendations = _build_recommendations(result, patient_info)
    return result


def _build_recommendations(result: ScreeningResult, patient_info: Optional[dict]) -> list:
    recs = []

    if result.red_flags:
        recs.append("🔴 IMMEDIATE ACTION: Crisis indicators detected — contact mental health services now.")

    if result.overall_risk == "high":
        recs += [
            "Urgent psychiatric evaluation within 24–48 hours.",
            "Do not make major financial, relationship, or work decisions until evaluated.",
            "Inform a trusted person about your current state.",
        ]
    elif result.overall_risk == "moderate":
        recs += [
            "Schedule a psychiatrist or psychologist appointment within 1 week.",
            "Begin structured mood tracking (sleep, energy, mood, 0–10 daily).",
        ]
    elif result.overall_risk == "low":
        recs += [
            "Discuss these results with your general practitioner.",
            "Consider psychoeducation about mood disorders.",
        ]
    else:
        recs.append("No urgent concern. Maintain regular sleep, exercise, and social connection.")

    if result.dominant_state == "manic":
        recs += [
            "Stabilise sleep schedule — aim for 7–9 hours at consistent times.",
            "Reduce stimulants, high-stimulation environments, and major decisions.",
            "Mood stabiliser evaluation may be warranted.",
        ]
    elif result.dominant_state == "depressive":
        recs += [
            "Maintain daily structure and light physical activity.",
            "Avoid alcohol and substance use.",
            "If antidepressants are considered, ensure bipolar disorder has been ruled out first.",
        ]
    elif result.dominant_state == "mixed":
        recs += [
            "Mixed states carry elevated risk — clinical evaluation is strongly recommended.",
            "DBT-based emotion regulation and distress tolerance skills may be beneficial.",
        ]

    recs.append(
        "💡 Crisis lines: iCall 9152987821 · Vandrevala 1860-2662-345 · "
        "NIMHANS 080-46110007 · International: findahelpline.com"
    )
    return recs