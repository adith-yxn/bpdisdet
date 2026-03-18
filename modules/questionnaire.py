"""
questionnaire.py  ·  bpdisdet v2
══════════════════════════════════
Validated clinical screening questionnaires:
  • MDQ-7  — Mood Disorder Questionnaire (mania screening, Hirschfeld 2000)
  • PHQ-9  — Patient Health Questionnaire (depression severity, Kroenke 2001)
  • ALS-SF — Affective Lability Scale Short Form (instability proxy)

References:
  Hirschfeld RM et al. (2000). Am J Psychiatry 157(11):1873-1875
  Kroenke K et al. (2001). J Gen Intern Med 16(9):606-613
  Harvey PD et al. (2002). Psychol Med 32(4):633-645
"""

from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════════════
# MDQ-7  (Simplified)  — Mania/Hypomania Screening
# ══════════════════════════════════════════════════════════════════════════════
MDQ_QUESTIONS = [
    {
        "id": "mdq1",
        "text": "Has there ever been a period when you were so irritable that you shouted at people, "
                "started fights, or had extreme arguments?",
        "category": "irritability",
    },
    {
        "id": "mdq2",
        "text": "Has there been a period when you felt so good or so hyper that other people thought "
                "you were not your normal self — or you were so hyper that you got into trouble?",
        "category": "elevated_mood",
    },
    {
        "id": "mdq3",
        "text": "Did you need much less sleep than usual and still felt rested?",
        "category": "decreased_sleep",
    },
    {
        "id": "mdq4",
        "text": "Did your thoughts race or come so fast you couldn't slow them down?",
        "category": "racing_thoughts",
    },
    {
        "id": "mdq5",
        "text": "Were you more talkative or spoke faster than usual?",
        "category": "pressured_speech",
    },
    {
        "id": "mdq6",
        "text": "Were you much more active than usual — starting lots of new projects, "
                "feeling much more energetic, or becoming more sexually active?",
        "category": "increased_activity",
    },
    {
        "id": "mdq7",
        "text": "Did you feel particularly special, important, or that you had special powers or abilities?",
        "category": "grandiosity",
    },
]

MDQ_RESPONSE_OPTIONS = ["Never / Not at all", "Rarely", "Sometimes", "Often", "Very often / Always"]
MDQ_WEIGHTS          = [0, 1, 2, 3, 4]   # 0–4 per item → max 28


# ══════════════════════════════════════════════════════════════════════════════
# PHQ-9  — Depression Severity
# ══════════════════════════════════════════════════════════════════════════════
PHQ9_QUESTIONS = [
    {
        "id": "phq1",
        "text": "Little interest or pleasure in doing things",
        "category": "anhedonia",
    },
    {
        "id": "phq2",
        "text": "Feeling down, depressed, or hopeless",
        "category": "depressed_mood",
    },
    {
        "id": "phq3",
        "text": "Trouble falling or staying asleep, or sleeping too much",
        "category": "sleep_disturbance",
    },
    {
        "id": "phq4",
        "text": "Feeling tired or having little energy",
        "category": "fatigue",
    },
    {
        "id": "phq5",
        "text": "Poor appetite or overeating",
        "category": "appetite",
    },
    {
        "id": "phq6",
        "text": "Feeling bad about yourself — that you are a failure or have let yourself or your family down",
        "category": "worthlessness",
    },
    {
        "id": "phq7",
        "text": "Trouble concentrating on things, such as reading or watching TV",
        "category": "concentration",
    },
    {
        "id": "phq8",
        "text": "Moving or speaking so slowly that others could notice. "
                "Or the opposite — being so fidgety or restless that you have been moving around more than usual",
        "category": "psychomotor",
    },
    {
        "id": "phq9",
        "text": "Thoughts that you would be better off dead, or thoughts of hurting yourself in some way",
        "category": "suicidal_ideation",
        "is_safety_item": True,
    },
]

PHQ9_RESPONSE_OPTIONS = [
    "Not at all (0 days)",
    "Several days (1–6 days)",
    "More than half the days (7–11 days)",
    "Nearly every day (12–14 days)",
]
PHQ9_WEIGHTS = [0, 1, 2, 3]   # 0–3 per item → max 27


# ══════════════════════════════════════════════════════════════════════════════
# ALS-SF  — Affective Lability Scale Short Form (6 items)
# ══════════════════════════════════════════════════════════════════════════════
ALS_QUESTIONS = [
    {
        "id": "als1",
        "text": "My mood shifts from feeling fine to feeling sad and blue",
        "category": "depression_shift",
    },
    {
        "id": "als2",
        "text": "My mood shifts from feeling fine to feeling angry or irritable",
        "category": "anger_shift",
    },
    {
        "id": "als3",
        "text": "My mood shifts from feeling fine to feeling very happy or high",
        "category": "elation_shift",
    },
    {
        "id": "als4",
        "text": "My moods change very rapidly — from happy to sad to irritable in a matter of hours",
        "category": "rapid_cycling",
    },
    {
        "id": "als5",
        "text": "I feel anxious and then this suddenly changes to feeling sad or depressed",
        "category": "anxiety_depression",
    },
    {
        "id": "als6",
        "text": "I feel sad or depressed and this suddenly changes to feeling irritable or angry",
        "category": "depression_anger",
    },
]

ALS_RESPONSE_OPTIONS = ["Almost never", "Sometimes", "Often", "Almost always"]
ALS_WEIGHTS          = [0, 1, 2, 3]   # 0–3 per item → max 18


# ══════════════════════════════════════════════════════════════════════════════
# Result dataclass
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class QuestionnaireResult:
    mdq_raw_score:      int   = 0   # 0–28
    mdq_scaled:         float = 0.0 # 0–100
    mdq_positive:       bool  = False
    mdq_severity:       str   = "none"     # none / mild / moderate / severe

    phq9_raw_score:     int   = 0   # 0–27
    phq9_scaled:        float = 0.0 # 0–100
    phq9_positive:      bool  = False
    phq9_severity:      str   = "none"     # none / mild / moderate / moderately-severe / severe
    phq9_safety_flag:   bool  = False

    als_raw_score:      int   = 0   # 0–18
    als_scaled:         float = 0.0 # 0–100
    als_severity:       str   = "low"      # low / moderate / high

    composite_score:    float = 0.0 # 0–100 fused
    dominant_state:     str   = "euthymic"
    overall_risk:       str   = "minimal"

    mdq_answers:        dict  = field(default_factory=dict)
    phq9_answers:       dict  = field(default_factory=dict)
    als_answers:        dict  = field(default_factory=dict)

    category_breakdown: dict  = field(default_factory=dict)
    interpretation:     str   = ""
    recommendations:    list  = field(default_factory=list)


def score_questionnaire(
    mdq_answers:  dict,   # {question_id: response_index (0-4)}
    phq9_answers: dict,   # {question_id: response_index (0-3)}
    als_answers:  dict,   # {question_id: response_index (0-3)}
) -> QuestionnaireResult:
    """Compute validated questionnaire scores and clinical interpretation."""

    result = QuestionnaireResult(
        mdq_answers=mdq_answers,
        phq9_answers=phq9_answers,
        als_answers=als_answers,
    )

    # ── MDQ-7 ─────────────────────────────────────────────────────────
    mdq_total = sum(MDQ_WEIGHTS[ans] for ans in mdq_answers.values()
                    if isinstance(ans, int) and 0 <= ans < len(MDQ_WEIGHTS))
    result.mdq_raw_score = mdq_total
    result.mdq_scaled    = round(mdq_total / 28 * 100, 1)

    if   mdq_total >= 16: result.mdq_severity = "severe";   result.mdq_positive = True
    elif mdq_total >= 10: result.mdq_severity = "moderate"; result.mdq_positive = True
    elif mdq_total >= 5:  result.mdq_severity = "mild"
    else:                 result.mdq_severity = "none"

    # ── PHQ-9 ─────────────────────────────────────────────────────────
    phq9_total = sum(PHQ9_WEIGHTS[ans] for ans in phq9_answers.values()
                     if isinstance(ans, int) and 0 <= ans < len(PHQ9_WEIGHTS))
    result.phq9_raw_score = phq9_total
    result.phq9_scaled    = round(phq9_total / 27 * 100, 1)

    # PHQ-9 severity thresholds (Kroenke 2001)
    if   phq9_total >= 20: result.phq9_severity = "severe";            result.phq9_positive = True
    elif phq9_total >= 15: result.phq9_severity = "moderately-severe"; result.phq9_positive = True
    elif phq9_total >= 10: result.phq9_severity = "moderate";          result.phq9_positive = True
    elif phq9_total >= 5:  result.phq9_severity = "mild"
    else:                  result.phq9_severity = "none"

    # Safety item (PHQ-9 item 9)
    si_ans = phq9_answers.get("phq9", 0)
    result.phq9_safety_flag = isinstance(si_ans, int) and si_ans >= 1

    # ── ALS-SF ────────────────────────────────────────────────────────
    als_total = sum(ALS_WEIGHTS[ans] for ans in als_answers.values()
                    if isinstance(ans, int) and 0 <= ans < len(ALS_WEIGHTS))
    result.als_raw_score = als_total
    result.als_scaled    = round(als_total / 18 * 100, 1)

    if   als_total >= 13: result.als_severity = "high"
    elif als_total >= 7:  result.als_severity = "moderate"
    else:                 result.als_severity = "low"

    # ── Composite score (MDQ 35%, PHQ-9 40%, ALS 25%) ─────────────────
    result.composite_score = round(
        result.mdq_scaled  * 0.35 +
        result.phq9_scaled * 0.40 +
        result.als_scaled  * 0.25,
        1
    )

    # ── Dominant state ────────────────────────────────────────────────
    if result.mdq_positive and result.phq9_positive:
        result.dominant_state = "mixed"
    elif result.mdq_positive:
        result.dominant_state = "manic"
    elif result.phq9_positive:
        result.dominant_state = "depressive"
    else:
        result.dominant_state = "euthymic"

    # ── Overall risk ──────────────────────────────────────────────────
    peak = max(result.mdq_scaled, result.phq9_scaled, result.als_scaled)
    if result.phq9_safety_flag:                      result.overall_risk = "high"
    elif peak >= 65 or result.composite_score >= 60: result.overall_risk = "high"
    elif peak >= 40 or result.composite_score >= 35: result.overall_risk = "moderate"
    elif peak >= 20 or result.composite_score >= 15: result.overall_risk = "low"
    else:                                             result.overall_risk = "minimal"

    # ── Category breakdown ─────────────────────────────────────────────
    result.category_breakdown = _category_breakdown(mdq_answers, phq9_answers, als_answers)

    # ── Interpretation text ────────────────────────────────────────────
    result.interpretation = _interpret(result)

    # ── Recommendations ────────────────────────────────────────────────
    result.recommendations = _recommendations(result)

    return result


def _category_breakdown(mdq_ans, phq9_ans, als_ans) -> dict:
    bd = {}
    for i, q in enumerate(MDQ_QUESTIONS):
        ans = mdq_ans.get(q["id"], 0)
        bd[f"MDQ: {q['category']}"] = MDQ_WEIGHTS[ans] / 4 * 100 if isinstance(ans, int) else 0
    for i, q in enumerate(PHQ9_QUESTIONS):
        ans = phq9_ans.get(q["id"], 0)
        bd[f"PHQ: {q['category']}"] = PHQ9_WEIGHTS[ans] / 3 * 100 if isinstance(ans, int) else 0
    for i, q in enumerate(ALS_QUESTIONS):
        ans = als_ans.get(q["id"], 0)
        bd[f"ALS: {q['category']}"] = ALS_WEIGHTS[ans] / 3 * 100 if isinstance(ans, int) else 0
    return bd


def _interpret(r: QuestionnaireResult) -> str:
    parts = []
    parts.append(
        f"MDQ-7 score: **{r.mdq_raw_score}/28** ({r.mdq_severity} — "
        f"{'positive screen for hypomanic/manic features' if r.mdq_positive else 'below threshold'}). "
    )
    parts.append(
        f"PHQ-9 score: **{r.phq9_raw_score}/27** ({r.phq9_severity} depression severity"
        f"{' — SAFETY FLAG: suicidal ideation item endorsed' if r.phq9_safety_flag else ''}). "
    )
    parts.append(
        f"ALS-SF score: **{r.als_raw_score}/18** ({r.als_severity} affective lability). "
    )
    parts.append(
        f"Composite questionnaire score: **{r.composite_score:.0f}/100**. "
        f"Dominant pattern: **{r.dominant_state.upper()}**. "
        f"Overall risk: **{r.overall_risk.upper()}**."
    )
    return " ".join(parts)


def _recommendations(r: QuestionnaireResult) -> list:
    recs = []
    if r.phq9_safety_flag:
        recs.append("🔴 URGENT: PHQ-9 item 9 (suicidal ideation) was endorsed. "
                    "Contact a crisis helpline or emergency services immediately.")
    if r.overall_risk == "high":
        recs += [
            "Urgent psychiatric evaluation recommended within 24–48 hours.",
            "Share this screening with your mental health provider.",
        ]
    elif r.overall_risk == "moderate":
        recs += [
            "Schedule a psychiatric or psychological evaluation within 1 week.",
            "Begin daily mood tracking (e.g., eMoods, Daylio apps).",
        ]
    elif r.overall_risk == "low":
        recs += [
            "Discuss these results with your GP or a mental health professional.",
            "Monitor mood patterns and sleep for the next 2 weeks.",
        ]
    else:
        recs.append("No urgent action indicated. Continue healthy habits and monitor wellbeing.")

    if r.mdq_positive:
        recs.append("Positive MDQ screen — evaluation for bipolar spectrum disorder warranted.")
    if r.phq9_severity in ("moderate", "moderately-severe", "severe"):
        recs.append("PHQ-9 indicates significant depression — antidepressant therapy should "
                    "only be started after ruling out bipolar disorder (risk of switching).")
    if r.als_severity == "high":
        recs.append("High affective lability — DBT-based emotion regulation skills may be beneficial.")

    recs.append(
        "💡 Free resources: iCall (India) 9152987821 · "
        "Vandrevala 1860-2662-345 · NIMHANS 080-46110007 · WHO mhGAP: who.int/mhgap"
    )
    return recs