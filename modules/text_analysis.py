"""
text_analysis.py  ·  bpdisdet v2
══════════════════════════════════
LLM-powered psycholinguistic analysis using Anthropic Claude.
DSM-5-TR aligned markers for bipolar spectrum disorders.
Full local heuristic fallback — zero internet required.
"""

import re
import json
import time
from dataclasses import dataclass, field
from typing import Optional


# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class LinguisticMarkers:
    # Mania/Hypomania
    pressured_speech:    float = 0.0   # 0–100
    flight_of_ideas:     float = 0.0
    grandiosity:         float = 0.0
    decreased_sleep_ref: float = 0.0
    goal_directed_act:   float = 0.0
    distractibility:     float = 0.0
    # Depression
    anhedonia:           float = 0.0
    hopelessness:        float = 0.0
    worthlessness:       float = 0.0
    psychomotor_slow:    float = 0.0
    somatic_complaints:  float = 0.0
    suicidal_ideation:   float = 0.0
    # Mixed / Trans-diagnostic
    irritability:        float = 0.0
    mixed_dysphoria:     float = 0.0
    cognitive_disruption:float = 0.0
    # Stylometric
    words_per_sentence:  float = 0.0
    lexical_diversity:   float = 0.0
    exclamation_density: float = 0.0
    caps_ratio:          float = 0.0
    sentiment_positive:  float = 0.0
    sentiment_negative:  float = 0.0


@dataclass
class TextAnalysisResult:
    raw_text:          str
    word_count:        int   = 0
    markers:           LinguisticMarkers = field(default_factory=LinguisticMarkers)
    mania_score:       float = 0.0
    depression_score:  float = 0.0
    mixed_score:       float = 0.0
    risk_level:        str   = "minimal"
    dominant_state:    str   = "euthymic"
    key_phrases:       list  = field(default_factory=list)
    clinical_summary:  str   = ""
    recommendations:   list  = field(default_factory=list)
    confidence:        float = 0.0
    analysis_method:   str   = "heuristic"
    suicidal_flag:     bool  = False
    timestamp:         float = field(default_factory=time.time)


# ── Claude system prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert clinical psycholinguist trained in DSM-5-TR criteria
for Bipolar Spectrum Disorders (Bipolar I, Bipolar II, Cyclothymia, and Mixed Features).

Analyse the submitted user text for the following evidence-based linguistic and semantic markers:

═══ MANIA / HYPOMANIA MARKERS ═══
1. pressured_speech      — Run-on sentences, breathless pacing, minimal punctuation breaks,
                           excessive exclamation marks, ALL CAPS usage
2. flight_of_ideas       — Abrupt, loose topic shifts; tangential associations; rhyming or
                           clang associations
3. grandiosity           — Inflated self-esteem language; special destiny/mission; invincibility
                           claims; references to unusual powers, relationships with famous figures
4. decreased_sleep_ref   — Explicit references to not needing sleep, feeling rested after <4h,
                           productivity during typical sleep hours
5. goal_directed_act     — Flood of new plans, projects, businesses started simultaneously;
                           hypersexuality references; excessive social engagement
6. distractibility       — Derailing mid-sentence; parenthetical tangents; inability to finish
                           a thought before starting another

═══ DEPRESSION MARKERS ═══
7. anhedonia             — "Nothing feels good", loss of pleasure, passive references
8. hopelessness          — Future is blocked/closed; "nothing will change"; fatalistic language
9. worthlessness         — Self-blame; "I am a burden"; negative self-concept statements
10. psychomotor_slow     — Very short sentences; ellipses suggesting pauses; slow, effortful
                           expression; incomplete thoughts
11. somatic_complaints   — Fatigue, pain, appetite changes, heaviness, brain fog references
12. suicidal_ideation    — Any references to wanting to die, ending it, being better off dead,
                           self-harm, overdose — FLAG IMMEDIATELY even if vague

═══ MIXED / TRANS-DIAGNOSTIC MARKERS ═══
13. irritability         — Anger at self/others without clear provocation; hostile tone;
                           feeling betrayed or wronged
14. mixed_dysphoria      — Simultaneous elevated AND depressed elements; "I feel awful but
                           can't stop thinking"; racing negative thoughts
15. cognitive_disruption — Reported confusion, memory complaints, concentration difficulties,
                           dissociation hints

SCORING: Each marker 0–100 where:
  0–20  = absent/minimal    40–60 = moderate presence
  20–40 = mild presence     60–80 = prominent          80–100 = severe/florid

OVERALL SCORES (0–100):
  mania_score      = weighted aggregate of markers 1–6
  depression_score = weighted aggregate of markers 7–12
  mixed_score      = weighted aggregate of markers 13–15 PLUS overlap of mania+depression

RISK STRATIFICATION:
  minimal  = No significant markers; appears euthymic
  low      = Sub-threshold markers; monitoring warranted
  moderate = Multiple moderate markers; evaluation recommended within 1 week
  high     = Florid markers, suicidal ideation, or acute episode features — URGENT

RESPOND ONLY with this exact JSON (no markdown, no extra text):
{
  "markers": {
    "pressured_speech": 0,
    "flight_of_ideas": 0,
    "grandiosity": 0,
    "decreased_sleep_ref": 0,
    "goal_directed_act": 0,
    "distractibility": 0,
    "anhedonia": 0,
    "hopelessness": 0,
    "worthlessness": 0,
    "psychomotor_slow": 0,
    "somatic_complaints": 0,
    "suicidal_ideation": 0,
    "irritability": 0,
    "mixed_dysphoria": 0,
    "cognitive_disruption": 0,
    "words_per_sentence": 0,
    "lexical_diversity": 0,
    "exclamation_density": 0,
    "caps_ratio": 0,
    "sentiment_positive": 0,
    "sentiment_negative": 0
  },
  "mania_score": 0,
  "depression_score": 0,
  "mixed_score": 0,
  "risk_level": "minimal",
  "dominant_state": "euthymic",
  "key_phrases": [],
  "clinical_summary": "",
  "recommendations": [],
  "suicidal_flag": false,
  "confidence": 0
}"""


# ── Anthropic API analysis ────────────────────────────────────────────────────
def analyse_with_api(text: str, api_key: str) -> TextAnalysisResult:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        msg = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    "Analyse the following text for bipolar disorder linguistic markers.\n\n"
                    f"--- TEXT ---\n{text}\n--- END TEXT ---\n\n"
                    "Return ONLY the JSON object."
                )
            }]
        )

        raw = msg.content[0].text.strip()
        raw = re.sub(r"^```json?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)

        # Merge stylometric features computed locally (more accurate than LLM guessing)
        local_style = _stylometric_features(text)
        for k in ["words_per_sentence", "lexical_diversity", "exclamation_density",
                  "caps_ratio", "sentiment_positive", "sentiment_negative"]:
            data["markers"][k] = local_style.get(k, 0)

        m = LinguisticMarkers(**{k: float(v) for k, v in data["markers"].items()})
        result = TextAnalysisResult(
            raw_text=text,
            word_count=len(text.split()),
            markers=m,
            mania_score=float(data["mania_score"]),
            depression_score=float(data["depression_score"]),
            mixed_score=float(data["mixed_score"]),
            risk_level=data["risk_level"],
            dominant_state=data["dominant_state"],
            key_phrases=data.get("key_phrases", [])[:12],
            clinical_summary=data.get("clinical_summary", ""),
            recommendations=data.get("recommendations", []),
            confidence=float(data.get("confidence", 75)),
            analysis_method="claude-api",
            suicidal_flag=bool(data.get("suicidal_flag", False)),
        )

        if result.suicidal_flag:
            result.risk_level = "high"
            result.recommendations.insert(0,
                "⚠️ URGENT: Suicidal ideation markers detected. "
                "Contact a mental health professional or crisis line immediately.")
        return result

    except Exception as exc:
        r = analyse_heuristic(text)
        r.clinical_summary += f"  [API error — local analysis used: {str(exc)[:80]}]"
        return r


# ── Local heuristic analyser ──────────────────────────────────────────────────
_MANIA_LEXICON = {
    "amazing", "incredible", "genius", "mission", "destined", "chosen",
    "energy", "invincible", "unstoppable", "powerful", "brilliant",
    "perfect", "superpower", "special", "greatest", "best ever",
    "no sleep", "don't need sleep", "ideas", "projects", "business",
    "opportunity", "universe", "message", "enlightened", "fast",
    "excited", "euphoric", "revolutionary", "changed the world",
}
_DEPRESS_LEXICON = {
    "hopeless", "worthless", "nothing", "empty", "tired", "exhausted",
    "sad", "crying", "alone", "isolated", "pointless", "meaningless",
    "can't", "cannot", "never", "won't get better", "give up",
    "no point", "fail", "failure", "burden", "ugly", "hate myself",
    "numb", "disconnected", "heavy", "slow", "forget", "numb",
    "can't concentrate", "everything hurts", "no energy", "stayed in bed",
}
_SUICIDAL_PHRASES = {
    "want to die", "end it all", "kill myself", "not worth living",
    "better off dead", "suicid", "self-harm", "hurt myself", "overdose",
    "no reason to live", "can't go on", "disappear forever",
}
_IRRITABILITY_LEXICON = {
    "angry", "furious", "rage", "hate", "disgusting", "idiot",
    "stupid", "unfair", "betrayed", "lied", "cheated", "furious",
    "infuriating", "sick of", "fed up",
}
_SOMATIC_LEXICON = {
    "headache", "pain", "ache", "nausea", "stomach", "appetite",
    "weight", "sleep", "fatigue", "heavy", "body", "physical",
}

_POS_SENTIMENT = {"good","great","happy","love","joy","wonderful","excited",
                  "grateful","peaceful","hopeful","energized","fantastic","awesome"}
_NEG_SENTIMENT = {"bad","awful","terrible","hate","sad","horrible","miserable",
                  "depressed","anxious","scared","worthless","hopeless","fail","dark"}


def analyse_heuristic(text: str) -> TextAnalysisResult:
    """Full rule-based analyser — zero API, zero network."""
    tokens    = re.findall(r"\b\w+\b", text.lower())
    token_set = set(tokens)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words     = text.split()

    style = _stylometric_features(text)

    # Lexicon hits
    mania_hits  = sum(1 for t in token_set if t in _MANIA_LEXICON)
    depr_hits   = sum(1 for t in token_set if t in _DEPRESS_LEXICON)
    irrit_hits  = sum(1 for t in token_set if t in _IRRITABILITY_LEXICON)
    somat_hits  = sum(1 for t in token_set if t in _SOMATIC_LEXICON)
    sui_flag    = any(ph in text.lower() for ph in _SUICIDAL_PHRASES)

    # Stylometric signals
    exc_density   = style["exclamation_density"]
    caps_r        = style["caps_ratio"]
    avg_sent_len  = style["words_per_sentence"]
    lex_div       = style["lexical_diversity"]

    # ── Marker scores ──────────────────────────────────────────────────
    pressured    = min(100, exc_density * 35 + caps_r * 50 + max(0, avg_sent_len - 20) * 1.5)
    flight       = min(100, (1 - lex_div) * 40 + exc_density * 25)
    grandiosity  = min(100, mania_hits * 9)
    goal_act     = min(100, mania_hits * 5 + exc_density * 20)
    distract     = min(100, (1 - lex_div) * 30 + caps_r * 30)
    sleep_ref    = min(100, 60 if any(w in text.lower() for w in
                                      ["sleep", "insomnia", "awake all night", "no sleep"]) else 0)

    anhedonia    = min(100, depr_hits * 6)
    hopeless     = min(100, depr_hits * 7)
    worthless    = min(100, sum(1 for t in token_set if t in
                                {"worthless","burden","useless","hate myself","failure"}) * 18)
    psycho_slow  = min(100, max(0, 15 - avg_sent_len) * 4)
    somatic      = min(100, somat_hits * 10)
    sui_score    = min(100, 85 if sui_flag else 0)

    irritability = min(100, irrit_hits * 12)
    mixed_dys    = min(100, (pressured * 0.3 + hopeless * 0.4 + irritability * 0.3)
                       if pressured > 20 and hopeless > 20 else 0)
    cog_disrupt  = min(100, depr_hits * 4 + distract * 0.3)

    # ── Composite scores ───────────────────────────────────────────────
    mania_score = min(100,
        pressured * 0.20 + grandiosity * 0.25 + flight * 0.15 +
        goal_act  * 0.15 + sleep_ref  * 0.10 + distract * 0.15)

    depr_score = min(100,
        anhedonia * 0.20 + hopeless   * 0.20 + worthless  * 0.15 +
        psycho_slow * 0.15 + somatic  * 0.10 + sui_score  * 0.20)

    mixed_score = min(100,
        mixed_dys * 0.5 + irritability * 0.3 +
        min(mania_score, depr_score) * 0.2 if mania_score > 25 and depr_score > 25 else
        mixed_dys * 0.5 + irritability * 0.3)

    # ── Risk level ──────────────────────────────────────────────────────
    peak = max(mania_score, depr_score, mixed_score)
    if sui_flag:          risk_level = "high"
    elif peak >= 65:      risk_level = "high"
    elif peak >= 40:      risk_level = "moderate"
    elif peak >= 20:      risk_level = "low"
    else:                 risk_level = "minimal"

    # ── Dominant state ──────────────────────────────────────────────────
    if mixed_score > 45:                                      dominant = "mixed"
    elif mania_score > depr_score and mania_score > 30:       dominant = "manic"
    elif depr_score > mania_score and depr_score > 30:        dominant = "depressive"
    else:                                                      dominant = "euthymic"

    # ── Key phrases ─────────────────────────────────────────────────────
    key_phrases = []
    for ph in _SUICIDAL_PHRASES:
        if ph in text.lower():
            key_phrases.append(f"⚠️ {ph}")
    for w in list(_MANIA_LEXICON)[:6]:
        if w in text.lower() and w not in key_phrases:
            key_phrases.append(w)
    for w in list(_DEPRESS_LEXICON)[:6]:
        if w in text.lower() and w not in key_phrases:
            key_phrases.append(w)

    # ── Summary & recs ──────────────────────────────────────────────────
    summary = (
        f"Heuristic linguistic screening detected a **{dominant}** pattern. "
        f"Mania indicators: {mania_score:.0f}/100 · "
        f"Depression indicators: {depr_score:.0f}/100 · "
        f"Mixed features: {mixed_score:.0f}/100. "
        f"Text length: {len(words)} words across {len(sentences)} sentences. "
        f"Lexical diversity: {lex_div*100:.0f}%."
    )

    recs = ["Consult a licensed mental health professional for formal evaluation."]
    if sui_flag:
        recs.insert(0, "⚠️ URGENT: Crisis language detected — contact a helpline now.")
    if mania_score > 50:
        recs.append("Evaluation for hypomanic/manic episode recommended.")
    if depr_score > 50:
        recs.append("Evaluation for depressive episode recommended.")
    if mixed_score > 45:
        recs.append("Mixed affective features present — psychiatric consultation advised.")

    markers = LinguisticMarkers(
        pressured_speech=pressured, flight_of_ideas=flight, grandiosity=grandiosity,
        decreased_sleep_ref=sleep_ref, goal_directed_act=goal_act, distractibility=distract,
        anhedonia=anhedonia, hopelessness=hopeless, worthlessness=worthless,
        psychomotor_slow=psycho_slow, somatic_complaints=somatic, suicidal_ideation=sui_score,
        irritability=irritability, mixed_dysphoria=mixed_dys, cognitive_disruption=cog_disrupt,
        words_per_sentence=avg_sent_len, lexical_diversity=lex_div * 100,
        exclamation_density=exc_density * 100, caps_ratio=caps_r * 100,
        sentiment_positive=style["sentiment_positive"],
        sentiment_negative=style["sentiment_negative"],
    )

    return TextAnalysisResult(
        raw_text=text,
        word_count=len(words),
        markers=markers,
        mania_score=mania_score,
        depression_score=depr_score,
        mixed_score=mixed_score,
        risk_level=risk_level,
        dominant_state=dominant,
        key_phrases=key_phrases[:12],
        clinical_summary=summary,
        recommendations=recs,
        confidence=42.0,
        analysis_method="heuristic",
        suicidal_flag=sui_flag,
    )


# ── Stylometric feature extraction ────────────────────────────────────────────
def _stylometric_features(text: str) -> dict:
    words     = text.split()
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    tokens    = re.findall(r"\b\w+\b", text.lower())

    wps = len(words) / max(len(sentences), 1)
    lex_div = len(set(tokens)) / max(len(tokens), 1)
    exc_den = text.count("!") / max(len(sentences), 1)
    caps_r  = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    tok_set = set(tokens)
    pos_s = min(100, len(tok_set & _POS_SENTIMENT) / max(len(tok_set), 1) * 800)
    neg_s = min(100, len(tok_set & _NEG_SENTIMENT) / max(len(tok_set), 1) * 800)

    return {
        "words_per_sentence":   round(wps, 2),
        "lexical_diversity":    round(lex_div, 3),
        "exclamation_density":  round(exc_den, 3),
        "caps_ratio":           round(caps_r, 4),
        "sentiment_positive":   round(pos_s, 2),
        "sentiment_negative":   round(neg_s, 2),
    }