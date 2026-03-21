"""
screening_engine.py · bpdisdet v6
════════════════════════════════════
Three-modality fusion engine — bulletproof, zero-crash.
Weights: Facial 25% + Text 40% + Questionnaire 35%
Auto-renormalises when any modality is missing.
All scores clamped. All NaN/Inf handled.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, List

from modules.facial_analysis import FacialSession
from modules.text_analysis   import TextAnalysisResult

# Import QuestionnaireResult safely
try:
    from modules.questionnaire import QuestionnaireResult
except ImportError:
    QuestionnaireResult = None  # type: ignore


# Fusion weights (normalised when modalities missing)
_W_FACIAL = 0.25
_W_TEXT   = 0.40
_W_QUEST  = 0.35

# Risk thresholds
_HIGH_THRESH = 62
_MOD_THRESH  = 38
_LOW_THRESH  = 18


def _sc(val, lo=0.0, hi=100.0) -> float:
    """Safe clamp — handles None, NaN, Inf."""
    try:
        v = float(val)
        if v != v or abs(v) == float("inf"):
            return 0.0
        return max(lo, min(hi, v))
    except Exception:
        return 0.0


@dataclass
class ScreeningResult:
    # ── Composite scores ────────────────────────────────────────────────
    composite_mania:       float = 0.0
    composite_depression:  float = 0.0
    composite_mixed:       float = 0.0
    affective_instability: float = 0.0

    # ── Classification ──────────────────────────────────────────────────
    overall_risk:    str   = "minimal"   # minimal / low / moderate / high
    dominant_state:  str   = "euthymic"  # euthymic / manic / depressive / mixed
    confidence_pct:  float = 0.0

    # ── Source data ─────────────────────────────────────────────────────
    facial_session:       Optional[FacialSession]         = None
    text_results:         List[TextAnalysisResult]         = field(default_factory=list)
    questionnaire_result: object                           = None   # QuestionnaireResult

    # ── Flags ───────────────────────────────────────────────────────────
    has_facial:        bool = False
    has_text:          bool = False
    has_questionnaire: bool = False

    # ── Narrative ───────────────────────────────────────────────────────
    clinical_summary:  str        = ""
    recommendations:   List[str]  = field(default_factory=list)
    red_flags:         List[str]  = field(default_factory=list)

    # ── Per-modality breakdown (for radar chart) ─────────────────────────
    modality_scores:   dict       = field(default_factory=dict)

    # ── Metadata ────────────────────────────────────────────────────────
    session_id:  str   = field(default_factory=lambda: str(uuid.uuid4())[:8].upper())
    timestamp:   float = field(default_factory=time.time)
    user_role:   str   = "patient"


def compute_screening_result(
    facial_session:       Optional[FacialSession]       = None,
    text_results:         Optional[List[TextAnalysisResult]] = None,
    questionnaire_result: object                         = None,
    patient_info:         Optional[dict]                = None,
) -> ScreeningResult:
    """
    Fuse all available modalities into a composite screening result.
    Safe — never raises an exception.
    """
    result = ScreeningResult(
        facial_session=facial_session,
        text_results=text_results or [],
        questionnaire_result=questionnaire_result,
    )

    try:
        # ── Per-modality raw scores ────────────────────────────────────
        wf = wt = wq = 0.0
        f_mania = f_depr = f_mixed = f_inst = 0.0
        t_mania = t_depr = t_mixed = t_conf = 0.0
        q_mania = q_depr = q_mixed = 0.0

        # Facial
        if facial_session is not None and len(facial_session.frames) >= 1:
            result.has_facial = True
            wf      = _W_FACIAL
            f_mania = _sc(facial_session.mania_score)
            f_depr  = _sc(facial_session.depression_score)
            f_mixed = _sc(facial_session.mixed_state_score)
            f_inst  = _sc(facial_session.affective_instability * 28.0, 0, 100)

        # Text
        tr_list = text_results or []
        if tr_list:
            result.has_text = True
            wt      = _W_TEXT
            t_mania = _sc(sum(r.mania_score      for r in tr_list) / len(tr_list))
            t_depr  = _sc(sum(r.depression_score for r in tr_list) / len(tr_list))
            t_mixed = _sc(sum(r.mixed_score      for r in tr_list) / len(tr_list))
            t_conf  = _sc(sum(r.confidence       for r in tr_list) / len(tr_list))

        # Questionnaire
        if questionnaire_result is not None:
            result.has_questionnaire = True
            wq      = _W_QUEST
            q_mania = _sc(getattr(questionnaire_result, "mdq_scaled",  0))
            q_depr  = _sc(getattr(questionnaire_result, "phq9_scaled", 0))
            q_mixed = _sc(getattr(questionnaire_result, "als_scaled",  0))

        total_w = wf + wt + wq
        if total_w < 1e-9:
            return result   # No data — return empty result

        # ── Composite scores ───────────────────────────────────────────
        result.composite_mania = _sc(
            (wf * f_mania + wt * t_mania + wq * q_mania) / total_w)
        result.composite_depression = _sc(
            (wf * f_depr  + wt * t_depr  + wq * q_depr)  / total_w)
        result.composite_mixed = _sc(
            (wf * f_mixed + wt * t_mixed + wq * q_mixed * 0.75) / total_w)

        # Affective instability: facial MSSD + text mixed + questionnaire lability
        result.affective_instability = _sc(
            (f_inst * wf + result.composite_mixed * wt * 0.45 +
             q_mixed * wq) / total_w)

        # ── Dominant state ─────────────────────────────────────────────
        cm = result.composite_mania
        cd = result.composite_depression
        cx = result.composite_mixed

        if   cx > 40 and cx >= cm and cx >= cd:  result.dominant_state = "mixed"
        elif cm > cd and cm > 24:                result.dominant_state = "manic"
        elif cd > cm and cd > 24:                result.dominant_state = "depressive"
        else:                                     result.dominant_state = "euthymic"

        # ── Overall risk ───────────────────────────────────────────────
        peak = max(cm, cd, cx, result.affective_instability)

        if   peak >= _HIGH_THRESH: result.overall_risk = "high"
        elif peak >= _MOD_THRESH:  result.overall_risk = "moderate"
        elif peak >= _LOW_THRESH:  result.overall_risk = "low"
        else:                       result.overall_risk = "minimal"

        # Override from critical flags
        for tr in tr_list:
            if tr.suicidal_flag or tr.risk_level == "high":
                result.overall_risk = "high"
        if questionnaire_result is not None:
            if getattr(questionnaire_result, "phq9_safety_flag", False):
                result.overall_risk = "high"
        if result.affective_instability > 75:
            if result.overall_risk == "minimal":
                result.overall_risk = "low"

        # ── Confidence ────────────────────────────────────────────────
        n_mods = sum([result.has_facial, result.has_text, result.has_questionnaire])
        base   = (
            (62.0 if result.has_facial else 0.0) +
            (t_conf if result.has_text else 0.0) +
            (74.0 if result.has_questionnaire else 0.0)
        )
        result.confidence_pct = _sc(
            min(88.0, base / max(n_mods * 100.0, 1.0) * 100.0))

        # ── Red flags ──────────────────────────────────────────────────
        for tr in tr_list:
            if tr.suicidal_flag:
                result.red_flags.append(
                    "[Text] Suicidal ideation markers detected in written text.")
            for kp in tr.key_phrases:
                if "CRISIS" in kp.upper() or "URGENT" in kp.upper():
                    result.red_flags.append(f"[Text] {kp}")

        if questionnaire_result is not None:
            if getattr(questionnaire_result, "phq9_safety_flag", False):
                result.red_flags.append(
                    "[Questionnaire] PHQ-9 item 9 (suicidal ideation) endorsed.")

        if result.affective_instability > 60:
            result.red_flags.append(
                "[Facial] High affective instability index — rapid emotional fluctuations detected.")

        # Deduplicate red flags
        result.red_flags = list(dict.fromkeys(result.red_flags))

        # ── Modality scores for charts ─────────────────────────────────
        result.modality_scores = {
            "Facial Mania":   round(f_mania, 1),
            "Facial Depr":    round(f_depr,  1),
            "Facial Mixed":   round(f_mixed, 1),
            "Text Mania":     round(t_mania, 1),
            "Text Depr":      round(t_depr,  1),
            "Text Mixed":     round(t_mixed, 1),
            "MDQ (Mania)":    round(q_mania, 1),
            "PHQ-9 (Depr)":   round(q_depr,  1),
            "ALS (Lability)": round(q_mixed, 1),
        }

        # ── Clinical summary ───────────────────────────────────────────
        mods_used = []
        if result.has_facial:        mods_used.append("facial affect analysis")
        if result.has_text:          mods_used.append("linguistic screening")
        if result.has_questionnaire: mods_used.append("MDQ-7, PHQ-9, and ALS-SF questionnaires")

        result.clinical_summary = (
            f"Three-modality screening using {', '.join(mods_used)} indicates a "
            f"**{result.dominant_state.upper()}** affective pattern with "
            f"**{result.overall_risk.upper()} RISK** signal. "
            f"Composite mania index: {result.composite_mania:.0f}/100. "
            f"Composite depression index: {result.composite_depression:.0f}/100. "
            f"Mixed/lability index: {result.composite_mixed:.0f}/100. "
            f"Affective instability: {result.affective_instability:.0f}/100. "
            f"Confidence: {result.confidence_pct:.0f}% "
            f"(maximum 88% — screening tool, not a clinical diagnosis)."
        )

        # ── Recommendations ────────────────────────────────────────────
        result.recommendations = _build_recommendations(result)

    except Exception as e:
        result.clinical_summary = (
            f"Screening computation encountered an error: {str(e)[:100]}. "
            f"Please retry or consult a mental health professional directly."
        )
        result.recommendations = [
            "Please consult a licensed mental health professional.",
            "Crisis: iCall 9152987821 | Vandrevala 1860-2662-345 | Emergency 112",
        ]

    return result


def _build_recommendations(result: ScreeningResult) -> List[str]:
    recs = []

    # ── Crisis first ───────────────────────────────────────────────────
    if result.red_flags:
        recs.append(
            "🔴 IMMEDIATE: Crisis indicators detected. Contact a mental health "
            "professional or crisis helpline without delay.")

    # ── Risk-level guidance ────────────────────────────────────────────
    if result.overall_risk == "high":
        recs += [
            "Urgent psychiatric evaluation within 24-48 hours.",
            "Share these screening results with your mental health provider.",
            "Avoid major financial, legal, or relationship decisions until formally evaluated.",
            "Inform a trusted person about your current state.",
        ]
    elif result.overall_risk == "moderate":
        recs += [
            "Schedule a psychiatric or psychological evaluation within 1 week.",
            "Begin structured daily mood tracking (sleep, energy, mood on 0-10 scale).",
        ]
    elif result.overall_risk == "low":
        recs += [
            "Discuss these results with your GP or a mental health professional.",
            "Consider psychoeducation about mood disorders.",
        ]
    else:
        recs.append(
            "No urgent concern detected. Continue monitoring mood, sleep, and energy.")

    # ── State-specific ─────────────────────────────────────────────────
    if result.dominant_state == "manic":
        recs += [
            "Prioritise sleep regularity (7-9 hours at consistent times).",
            "Limit caffeine, alcohol, and high-stimulation environments.",
            "Mood stabiliser evaluation may be clinically warranted.",
        ]
    elif result.dominant_state == "depressive":
        recs += [
            "Maintain a regular daily routine with light physical activity.",
            "Avoid alcohol and substance use.",
            "If antidepressants are considered, bipolar disorder must be ruled out first "
            "(antidepressant monotherapy can trigger manic switching).",
        ]
    elif result.dominant_state == "mixed":
        recs += [
            "Mixed states carry elevated risk — clinical evaluation is strongly recommended.",
            "DBT-based emotion regulation and distress tolerance skills may be beneficial.",
        ]

    # ── Always include crisis resources ───────────────────────────────
    recs.append(
        "Free resources: iCall 9152987821 | Vandrevala 1860-2662-345 | "
        "NIMHANS 080-46110007 | Emergency 112 | who.int/mhgap"
    )

    return recs