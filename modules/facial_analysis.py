"""
facial_analysis.py
──────────────────
Real-time facial landmark & emotion detection using OpenCV + DeepFace.
Tracks affective instability by measuring emotional volatility over time.
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import time


# ── Emotion categories mapped to valence/arousal ──────────────────────────────
EMOTION_VALENCE = {
    "happy":     (+1.0,  0.6),
    "surprise":  (+0.3,  0.9),
    "neutral":   ( 0.0,  0.0),
    "sad":       (-0.8, -0.4),
    "fear":      (-0.7,  0.8),
    "disgust":   (-0.6,  0.3),
    "angry":     (-0.5,  0.9),
}

MANIA_EMOTIONS   = {"happy", "surprise", "angry"}
DEPRESS_EMOTIONS = {"sad", "fear", "disgust"}
MIXED_EMOTIONS   = {"angry", "fear"}


@dataclass
class EmotionFrame:
    timestamp:   float
    dominant:    str
    scores:      dict
    valence:     float
    arousal:     float
    face_box:    Optional[tuple] = None
    landmarks:   Optional[np.ndarray] = None


@dataclass
class FacialSession:
    frames:               list = field(default_factory=list)
    start_time:           float = field(default_factory=time.time)
    affective_instability: float = 0.0
    mania_score:          float = 0.0
    depression_score:     float = 0.0
    mixed_state_score:    float = 0.0
    dominant_pattern:     str  = "neutral"
    emotion_transitions:  int  = 0
    valence_history:      list = field(default_factory=list)
    arousal_history:      list = field(default_factory=list)


# ── Haar Cascade face detector (no internet required) ─────────────────────────
class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

    def detect(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        results = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            results.append({
                "box": (x, y, w, h),
                "has_eyes": len(eyes) > 0,
                "eye_positions": eyes.tolist() if len(eyes) > 0 else [],
            })
        return results


# ── DeepFace emotion analyser (falls back to heuristic if unavailable) ────────
def _try_deepface_analyse(face_img: np.ndarray) -> Optional[dict]:
    """Attempt DeepFace analysis; return None on any failure."""
    try:
        from deepface import DeepFace  # type: ignore
        result = DeepFace.analyze(
            face_img,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]
        return result.get("emotion", {})
    except Exception:
        return None


def _heuristic_emotion(face_img: np.ndarray) -> dict:
    """
    Lightweight heuristic emotion estimator based on grey-level facial region
    statistics.  Used as fallback when DeepFace is unavailable.
    Returns a dict of {emotion: probability_0_to_100}.
    """
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Region-of-interest statistics
    upper  = gray[:h//2, :]          # forehead / brow
    lower  = gray[h//2:, :]          # mouth / chin region
    left   = gray[:, :w//2]
    right  = gray[:, w//2:]

    upper_var  = float(np.var(upper))
    lower_var  = float(np.var(lower))
    brightness = float(np.mean(gray))
    contrast   = float(np.std(gray))
    asymmetry  = abs(float(np.mean(left)) - float(np.mean(right)))

    # Very rough heuristics (research-inspired, not clinically validated)
    happy_p    = max(0, min(100, 20 + (lower_var - upper_var) * 0.05 + (brightness - 100) * 0.1))
    sad_p      = max(0, min(100, 20 + (upper_var - lower_var) * 0.03 + (100 - brightness) * 0.1))
    angry_p    = max(0, min(100, 10 + asymmetry * 0.3 + contrast * 0.05))
    neutral_p  = max(0, 100 - happy_p - sad_p - angry_p * 0.5)
    surprise_p = max(0, min(100, upper_var * 0.02))
    fear_p     = max(0, min(100, contrast * 0.03 + asymmetry * 0.1))
    disgust_p  = max(0, min(100, lower_var * 0.01))

    raw = {
        "happy": happy_p, "sad": sad_p, "angry": angry_p,
        "neutral": neutral_p, "surprise": surprise_p,
        "fear": fear_p, "disgust": disgust_p,
    }
    total = sum(raw.values()) or 1
    return {k: v / total * 100 for k, v in raw.items()}


# ── Frame analyser ─────────────────────────────────────────────────────────────
def analyse_frame(frame: np.ndarray, detector: FaceDetector) -> tuple[np.ndarray, Optional[EmotionFrame]]:
    """
    Detect faces, run emotion analysis, annotate frame.
    Returns (annotated_frame, EmotionFrame | None).
    """
    faces = detector.detect(frame)
    annotated = frame.copy()
    emotion_frame = None

    for face in faces[:1]:   # process first detected face
        x, y, w, h = face["box"]
        face_img = frame[y:y+h, x:x+w]

        # --- Emotion scores ------------------------------------------------
        scores = _try_deepface_analyse(face_img) or _heuristic_emotion(face_img)
        dominant = max(scores, key=scores.get)
        valence, arousal = EMOTION_VALENCE.get(dominant, (0.0, 0.0))

        emotion_frame = EmotionFrame(
            timestamp=time.time(),
            dominant=dominant,
            scores={k: round(v, 2) for k, v in scores.items()},
            valence=valence,
            arousal=arousal,
            face_box=(x, y, w, h),
        )

        # --- Annotation ----------------------------------------------------
        COLORS = {
            "happy": (50, 220, 80), "surprise": (255, 200, 0),
            "neutral": (180, 180, 180), "sad": (80, 100, 220),
            "fear": (200, 80, 220), "disgust": (80, 200, 200),
            "angry": (50, 50, 220),
        }
        color = COLORS.get(dominant, (200, 200, 200))

        # Bounding box
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)

        # Emotion label + confidence
        conf = round(scores.get(dominant, 0), 1)
        label = f"{dominant.upper()}  {conf:.0f}%"
        cv2.rectangle(annotated, (x, y-32), (x+w, y), color, -1)
        cv2.putText(annotated, label, (x+4, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Eye indicators
        if face["has_eyes"]:
            cv2.circle(annotated, (x + w//4, y + h//3), 4, (0, 255, 200), -1)
            cv2.circle(annotated, (x + 3*w//4, y + h//3), 4, (0, 255, 200), -1)

        # Valence / Arousal bars
        bar_x = x + w + 10
        if bar_x + 80 < annotated.shape[1]:
            _draw_bar(annotated, bar_x, y,      valence, "V", (100, 220, 100))
            _draw_bar(annotated, bar_x, y + 30, arousal, "A", (100, 100, 220))

    # Overlay session watermark
    cv2.putText(annotated, "bpdisdet | emotion monitor",
                (8, annotated.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
    return annotated, emotion_frame


def _draw_bar(img, x, y, value, label, color):
    cv2.putText(img, label, (x, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    bar_w = int(40 * abs(value))
    bar_color = color if value >= 0 else (80, 80, 200)
    cv2.rectangle(img, (x+14, y+2), (x+14+bar_w, y+16), bar_color, -1)
    cv2.rectangle(img, (x+14, y+2), (x+54, y+16), (80, 80, 80), 1)


# ── Session analytics ──────────────────────────────────────────────────────────
def compute_session_metrics(session: FacialSession) -> FacialSession:
    """
    Compute affective instability, mania/depression scores from session frames.
    Uses MSSD (Mean Square of Successive Differences) for instability.
    """
    if len(session.frames) < 2:
        return session

    valences = [f.valence for f in session.frames]
    arousals = [f.arousal for f in session.frames]
    emotions = [f.dominant for f in session.frames]

    session.valence_history = valences
    session.arousal_history = arousals

    # Affective instability (MSSD of valence)
    diffs = np.diff(valences)
    session.affective_instability = float(np.mean(diffs ** 2)) * 10

    # Mania score: proportion of manic emotions × mean arousal of those frames
    manic_frames = [f for f in session.frames if f.dominant in MANIA_EMOTIONS]
    session.mania_score = min(100, (len(manic_frames) / len(session.frames)) * 100
                               * (1 + np.mean([f.arousal for f in manic_frames]) if manic_frames else 0))

    # Depression score: proportion of depressive emotions × abs(mean valence)
    depr_frames = [f for f in session.frames if f.dominant in DEPRESS_EMOTIONS]
    session.depression_score = min(100, (len(depr_frames) / len(session.frames)) * 100
                                    * (1 + abs(np.mean([f.valence for f in depr_frames])) if depr_frames else 0))

    # Mixed state: rapid oscillation between mania/depression clusters
    mixed = 0
    for i in range(1, len(emotions)):
        prev, curr = emotions[i-1], emotions[i]
        if ((prev in MANIA_EMOTIONS and curr in DEPRESS_EMOTIONS) or
                (prev in DEPRESS_EMOTIONS and curr in MANIA_EMOTIONS)):
            mixed += 1
    session.mixed_state_score = min(100, mixed / max(len(emotions)-1, 1) * 200)

    # Emotional transitions
    session.emotion_transitions = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])

    # Dominant pattern
    from collections import Counter
    counts = Counter(emotions)
    most_common = counts.most_common(1)[0][0]
    if session.mania_score > 55:
        session.dominant_pattern = "manic"
    elif session.depression_score > 55:
        session.dominant_pattern = "depressive"
    elif session.mixed_state_score > 40:
        session.dominant_pattern = "mixed"
    else:
        session.dominant_pattern = "stable"

    return session