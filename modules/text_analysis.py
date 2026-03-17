"""
text_analysis.py
────────────────
LLM-powered analysis of user text for linguistic markers of mania,
depression, and mixed affective states associated with bipolar disorder.

Uses the Anthropic Claude API.  Falls back to a local rule-based NLP
heuristic when the API key is unavailable.
"""

import re
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class LinguisticMarkers:
    pressured_speech:     float = 0.0   # 0–100
    flight_of_ideas:      float = 0.0
    grandiosity:          float = 0.0
    decreased_sleep_ref:  float = 0.0
    anhedonia_markers:    float = 0.0
    hopelessness:         float = 0.0
    psychomotor_slowdown: float = 0.0
    cognitive_slowing:    float = 0.0
    mixed_dysphoria:      float = 0.0
    irritability:         float = 0.0
    word_per_sentence:    float = 0.0
    lexical_diversity:    float = 0.0
    negative_sentiment:   float = 0.0
    positive_sentiment:   float = 0.0

@dataclass
class TextAnalysisResult:
    raw_text:            str
    markers:             LinguisticMarkers = field(default_factory=LinguisticMarkers)
    mania_score:         float = 0.0
    depression_score:    float = 0.0
    mixed_score:         float = 0.0
    risk_level:          str   = "minimal"       # minimal / low / moderate / high
    dominant_state:      str   = "euthymic"      # euthymic / manic / depressive / mixed
    key_phrases:         list  = field(default_factory=list)
    clinical_summary:    str   = ""
    recommendations:     list  = field(default_factory=list)
    confidence:          float = 0.0
    analysis_method:     str   = "api"            # api / heuristic
    timestamp:           float = field(default_factory=time.time)


# ── System prompt for Claude ───────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a clinical psycholinguistics assistant trained in DSM-5-TR 
criteria for Bipolar Spectrum Disorders. Your task is to analyse user-submitted text 
for linguistic and semantic markers associated with:

1. MANIA / HYPOMANIA markers:
   - Pressured speech (run-on sentences, excessive punctuation, ALL CAPS)
   - Flight of ideas (abrupt topic shifts, loose associations)
   - Grandiosity (inflated self-references, special powers/missions)
   - Decreased need for sleep references
   - Increased goal-directed activity language
   - Elevated/irritable tone, hypersexuality markers

2. DEPRESSION markers:
   - Anhedonia (loss of interest, "nothing matters" phrases)
   - Hopelessness / helplessness language
   - Psychomotor retardation (very short sentences, long pauses indicated by ellipses)
   - Cognitive slowing (confusion, memory complaints)
   - Somatic complaints (fatigue, pain, appetite)
   - Suicidal ideation markers (flagged separately)

3. MIXED STATE markers:
   - Simultaneous dysphoric and elevated elements
   - Agitated depression language
   - Racing negative thoughts

IMPORTANT DISCLAIMERS:
- This is a SCREENING TOOL ONLY, not a diagnostic instrument.
- Results must always be reviewed by a licensed mental health professional.
- Never provide a definitive diagnosis.
- Flag any suicidal or self-harm language immediately and prominently.

Respond ONLY with a valid JSON object (no markdown, no extra text) in this exact schema:
{
  "markers": {
    "pressured_speech": 0-100,
    "flight_of_ideas": 0-100,
    "grandiosity": 0-100,
    "decreased_sleep_ref": 0-100,
    "anhedonia_markers": 0-100,
    "hopelessness": 0-100,
    "psychomotor_slowdown": 0-100,
    "cognitive_slowing": 0-100,
    "mixed_dysphoria": 0-100,
    "irritability": 0-100
  },
  "mania_score": 0-100,
  "depression_score": 0-100,
  "mixed_score": 0-100,
  "risk_level": "minimal|low|moderate|high",
  "dominant_state": "euthymic|manic|depressive|mixed",
  "key_phrases": ["phrase1", "phrase2"],
  "clinical_summary": "2-3 sentence summary",
  "recommendations": ["recommendation1", "recommendation2"],
  "suicidal_ideation_flag": false,
  "confidence": 0-100
}"""


# ── Anthropic API caller ───────────────────────────────────────────────────────
def analyse_with_api(text: str, api_key: str) -> TextAnalysisResult:
    """Call Claude API for psycholinguistic analysis."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        user_prompt = f"""Please analyse the following text for bipolar disorder linguistic markers.
The person wrote this as part of a mental health screening.

--- TEXT BEGIN ---
{text}
--- TEXT END ---

Provide the JSON analysis as specified."""

        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw_json = message.content[0].text.strip()
        # Strip any accidental markdown fences
        raw_json = re.sub(r"^```json?\s*", "", raw_json)
        raw_json = re.sub(r"\s*```$", "", raw_json)
        data = json.loads(raw_json)

        markers = LinguisticMarkers(**data["markers"])
        # Add lexical diversity from local computation
        markers.lexical_diversity = _lexical_diversity(text)
        markers.word_per_sentence = _words_per_sentence(text)
        markers.negative_sentiment, markers.positive_sentiment = _sentiment_ratio(text)

        result = TextAnalysisResult(
            raw_text=text,
            markers=markers,
            mania_score=float(data["mania_score"]),
            depression_score=float(data["depression_score"]),
            mixed_score=float(data["mixed_score"]),
            risk_level=data["risk_level"],
            dominant_state=data["dominant_state"],
            key_phrases=data.get("key_phrases", []),
            clinical_summary=data.get("clinical_summary", ""),
            recommendations=data.get("recommendations", []),
            confidence=float(data.get("confidence", 75)),
            analysis_method="api",
        )

        # Handle suicidal ideation flag
        if data.get("suicidal_ideation_flag", False):
            result.risk_level = "high"
            result.recommendations.insert(0, "⚠️ URGENT: Potential suicidal ideation detected. "
                                           "Please contact a mental health professional immediately "
                                           "or call a crisis helpline.")
        return result

    except Exception as e:
        # Fall back to heuristic
        result = analyse_heuristic(text)
        result.clinical_summary += f" [API error: {str(e)[:60]}; heuristic used]"
        return result


# ── Local heuristic analyser ──────────────────────────────────────────────────
MANIA_WORDS = {
    "amazing", "incredible", "genius", "mission", "destined", "chosen",
    "energy", "invincible", "unstoppable", "powerful", "brilliant",
    "perfect", "superpower", "special", "greatest", "best ever",
    "no sleep", "don't need sleep", "ideas", "plan", "project", "business",
    "opportunity", "God", "universe", "message", "everything", "fast",
}
DEPRESSION_WORDS = {
    "hopeless", "worthless", "nothing", "empty", "tired", "exhausted",
    "sad", "crying", "alone", "isolated", "pointless", "meaningless",
    "can't", "cannot", "never", "won't get better", "give up",
    "no point", "fail", "failure", "burden", "ugly", "hate myself",
    "numb", "disconnected", "heavy", "slow", "forget", "can't concentrate",
}
SUICIDAL_PHRASES = {
    "want to die", "end it all", "kill myself", "not worth living",
    "better off dead", "suicide", "self-harm", "hurt myself", "overdose",
}
IRRITABILITY_WORDS = {
    "angry", "furious", "rage", "hate", "disgusting", "idiot",
    "stupid", "unfair", "betrayed", "lied", "cheated",
}


def analyse_heuristic(text: str) -> TextAnalysisResult:
    """Rule-based linguistic marker analysis (no API required)."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    sentences = re.split(r"[.!?]+", text)
    word_set = set(tokens)

    # Feature extraction
    exclamation_ratio = text.count("!") / max(len(sentences), 1)
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    avg_sentence_len = len(tokens) / max(len(sentences), 1)
    lex_div = _lexical_diversity(text)

    mania_hits  = len(word_set & MANIA_WORDS)
    depr_hits   = len(word_set & DEPRESSION_WORDS)
    irrit_hits  = len(word_set & IRRITABILITY_WORDS)
    sui_flag    = any(phrase in text.lower() for phrase in SUICIDAL_PHRASES)

    pressured = min(100, exclamation_ratio * 40 + caps_ratio * 60 + max(0, avg_sentence_len - 20) * 2)
    grandiosity = min(100, mania_hits * 8)
    hopelessness = min(100, depr_hits * 7)
    irritability = min(100, irrit_hits * 12)
    psychomotor_slow = min(100, max(0, 20 - avg_sentence_len) * 3)
    neg_sent, pos_sent = _sentiment_ratio(text)

    mania_score = min(100, pressured * 0.3 + grandiosity * 0.4 + irritability * 0.15 + pos_sent * 0.15)
    depression_score = min(100, hopelessness * 0.5 + psychomotor_slow * 0.2 + neg_sent * 0.3)
    mixed_score = min(100, (mania_score * 0.5 + depression_score * 0.5)
                       if mania_score > 30 and depression_score > 30 else
                       min(mania_score, depression_score) * 0.3)

    if sui_flag:
        risk_level = "high"
    elif max(mania_score, depression_score) > 65:
        risk_level = "moderate"
    elif max(mania_score, depression_score) > 35:
        risk_level = "low"
    else:
        risk_level = "minimal"

    if mixed_score > 45:
        dominant_state = "mixed"
    elif mania_score > depression_score and mania_score > 35:
        dominant_state = "manic"
    elif depression_score > mania_score and depression_score > 35:
        dominant_state = "depressive"
    else:
        dominant_state = "euthymic"

    markers = LinguisticMarkers(
        pressured_speech=pressured,
        flight_of_ideas=min(100, (1 - lex_div) * 50 + exclamation_ratio * 30),
        grandiosity=grandiosity,
        decreased_sleep_ref=min(100, 40 if "sleep" in text.lower() else 0),
        anhedonia_markers=min(100, depr_hits * 5),
        hopelessness=hopelessness,
        psychomotor_slowdown=psychomotor_slow,
        cognitive_slowing=min(100, depr_hits * 4),
        mixed_dysphoria=mixed_score,
        irritability=irritability,
        word_per_sentence=avg_sentence_len,
        lexical_diversity=lex_div * 100,
        negative_sentiment=neg_sent,
        positive_sentiment=pos_sent,
    )

    key_phrases = []
    for phrase in SUICIDAL_PHRASES:
        if phrase in text.lower():
            key_phrases.append(f"⚠️ {phrase}")
    for word in list(MANIA_WORDS)[:5]:
        if word in text.lower():
            key_phrases.append(word)
    for word in list(DEPRESSION_WORDS)[:5]:
        if word in text.lower():
            key_phrases.append(word)

    summary = (
        f"Heuristic analysis detected {dominant_state} linguistic features. "
        f"Mania indicators: {mania_score:.0f}/100. "
        f"Depression indicators: {depression_score:.0f}/100."
    )

    recs = ["Consult a licensed mental health professional for clinical evaluation."]
    if sui_flag:
        recs.insert(0, "⚠️ URGENT: Potential crisis language detected. "
                       "Contact crisis helpline immediately (e.g. iCall: 9152987821).")
    if mania_score > 50:
        recs.append("Consider evaluation for hypomanic/manic episodes.")
    if depression_score > 50:
        recs.append("Consider evaluation for depressive episodes.")

    return TextAnalysisResult(
        raw_text=text,
        markers=markers,
        mania_score=mania_score,
        depression_score=depression_score,
        mixed_score=mixed_score,
        risk_level=risk_level,
        dominant_state=dominant_state,
        key_phrases=key_phrases[:10],
        clinical_summary=summary,
        recommendations=recs,
        confidence=45.0,
        analysis_method="heuristic",
    )


# ── Utility functions ──────────────────────────────────────────────────────────
def _lexical_diversity(text: str) -> float:
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return 0.5
    return len(set(tokens)) / len(tokens)

def _words_per_sentence(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return 0.0
    return sum(len(s.split()) for s in sentences) / len(sentences)

def _sentiment_ratio(text: str) -> tuple[float, float]:
    """Returns (negative%, positive%) based on simple keyword presence."""
    POS = {"good", "great", "happy", "love", "joy", "wonderful", "excited",
           "grateful", "peaceful", "content", "hopeful", "energized"}
    NEG = {"bad", "awful", "terrible", "hate", "sad", "horrible", "miserable",
           "depressed", "anxious", "scared", "worthless", "hopeless", "fail"}
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    pos = len(tokens & POS) / max(len(tokens), 1) * 100
    neg = len(tokens & NEG) / max(len(tokens), 1) * 100
    return min(100, neg * 5), min(100, pos * 5)


# ── Composite multi-entry scorer ──────────────────────────────────────────────
def analyse_journal_entries(entries: list[str], api_key: Optional[str] = None) -> list[TextAnalysisResult]:
    """Analyse a list of journal/diary entries for longitudinal pattern."""
    results = []
    for entry in entries:
        if api_key:
            results.append(analyse_with_api(entry, api_key))
        else:
            results.append(analyse_heuristic(entry))
    return results