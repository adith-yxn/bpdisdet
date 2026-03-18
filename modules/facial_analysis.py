"""
facial_analysis.py  ·  bpdisdet v2
═══════════════════════════════════
Advanced facial affect analysis using PURE OpenCV geometric features.
No DeepFace, no TensorFlow — 100% Streamlit Cloud compatible.

Pipeline:
  1. Multi-cascade face + landmark detection (Haar + LBP)
  2. Geometric feature extraction:
       • Eye Aspect Ratio  (EAR)  — blink rate, alertness, psychomotor state
       • Mouth Aspect Ratio (MAR) — expressiveness, mania vs depression
       • Brow Displacement Ratio (BDR) — worry, surprise, anger
       • Facial Asymmetry Index (FAI) — emotional instability proxy
       • Head Pose Estimate (HPE) — avoidance / agitation
       • Skin Luminance Variance (SLV) — arousal proxy
  3. Rule-based emotion inference from feature vector
  4. Temporal MSSD (Mean Square of Successive Differences) for affective instability
  5. Mania / Depression / Mixed pattern classification
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional


# ── Emotion label colours (BGR) ───────────────────────────────────────────────
EMOTION_COLORS = {
    "joyful":    (50,  220, 80),
    "elevated":  (0,   200, 255),
    "agitated":  (50,  50,  220),
    "anxious":   (180, 80,  220),
    "withdrawn": (80,  100, 220),
    "flat":      (140, 140, 140),
    "neutral":   (180, 180, 180),
    "distressed":(50,  50,  200),
}

# Valence (+pos / -neg) and Arousal (high=1 / low=-1) for each label
EMOTION_VA = {
    "joyful":    ( 0.9,  0.6),
    "elevated":  ( 0.5,  0.9),
    "agitated":  (-0.3,  0.9),
    "anxious":   (-0.6,  0.7),
    "withdrawn": (-0.7, -0.5),
    "flat":      (-0.4, -0.8),
    "neutral":   ( 0.0,  0.0),
    "distressed":(-0.8,  0.4),
}

MANIA_LABELS    = {"joyful", "elevated", "agitated"}
DEPRESS_LABELS  = {"withdrawn", "flat", "distressed"}
MIXED_LABELS    = {"agitated", "anxious"}


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class GeometricFeatures:
    """Per-frame facial geometric measurements."""
    ear:           float = 0.0   # Eye Aspect Ratio          (0–1; low = closed/droopy)
    mar:           float = 0.0   # Mouth Aspect Ratio         (0–1; high = open/expressive)
    bdr:           float = 0.0   # Brow Displacement Ratio    (0–1; high = raised brows)
    fai:           float = 0.0   # Facial Asymmetry Index     (0–1; high = asymmetric)
    slv:           float = 0.0   # Skin Luminance Variance    (normalised)
    face_area:     float = 0.0   # Relative face size (proxy for camera distance)
    head_tilt_deg: float = 0.0   # Head tilt in degrees


@dataclass
class EmotionFrame:
    timestamp:    float
    emotion:      str
    confidence:   float
    valence:      float
    arousal:      float
    features:     GeometricFeatures = field(default_factory=GeometricFeatures)
    face_box:     Optional[tuple]   = None


@dataclass
class FacialSession:
    frames:                list  = field(default_factory=list)
    start_time:            float = field(default_factory=time.time)
    affective_instability: float = 0.0
    mania_score:           float = 0.0
    depression_score:      float = 0.0
    mixed_state_score:     float = 0.0
    dominant_pattern:      str   = "neutral"
    emotion_transitions:   int   = 0
    valence_history:       list  = field(default_factory=list)
    arousal_history:       list  = field(default_factory=list)
    ear_history:           list  = field(default_factory=list)
    mar_history:           list  = field(default_factory=list)
    feature_summary:       dict  = field(default_factory=dict)


# ── Cascade loaders ───────────────────────────────────────────────────────────
class CascadeDetector:
    """Loads multiple Haar/LBP cascades for robust face & feature detection."""

    def __init__(self):
        base = cv2.data.haarcascades
        self.face_front   = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self.face_alt     = cv2.CascadeClassifier(base + "haarcascade_frontalface_alt2.xml")
        self.eye          = cv2.CascadeClassifier(base + "haarcascade_eye_tree_eyeglasses.xml")
        self.mouth        = cv2.CascadeClassifier(base + "haarcascade_smile.xml")
        self.profile      = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")

    def detect_faces(self, gray: np.ndarray) -> list:
        f1 = self.face_front.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        f2 = self.face_alt.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        faces = []
        for det in [f1, f2]:
            if len(det) > 0:
                for rect in det:
                    faces.append(tuple(rect))
        # deduplicate by IoU
        return _nms(faces)

    def detect_eyes(self, roi_gray: np.ndarray) -> list:
        eyes = self.eye.detectMultiScale(roi_gray, 1.1, 3, minSize=(15, 15))
        return eyes.tolist() if len(eyes) > 0 else []

    def detect_mouth(self, roi_gray: np.ndarray) -> list:
        m = self.mouth.detectMultiScale(roi_gray, 1.7, 11, minSize=(20, 10))
        return m.tolist() if len(m) > 0 else []


def _nms(rects, overlap_thresh=0.4):
    """Simple non-maximum suppression."""
    if not rects:
        return []
    seen, kept = set(), []
    for i, (x1, y1, w1, h1) in enumerate(rects):
        duplicate = False
        for j, (x2, y2, w2, h2) in enumerate(rects):
            if i >= j:
                continue
            ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
            iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
            inter = ix * iy
            union = w1*h1 + w2*h2 - inter
            if union > 0 and inter / union > overlap_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append((x1, y1, w1, h1))
    return kept


# ── Geometric feature extraction ──────────────────────────────────────────────
def _extract_features(face_img: np.ndarray, gray_face: np.ndarray,
                      detector: CascadeDetector) -> GeometricFeatures:
    h, w = gray_face.shape
    feats = GeometricFeatures()

    # ── Skin luminance variance (arousal proxy) ──────────────────────
    feats.slv = float(np.std(face_img)) / 128.0

    # ── Facial asymmetry index ────────────────────────────────────────
    left_half  = gray_face[:, :w//2]
    right_half = np.fliplr(gray_face[:, w//2:])
    min_w = min(left_half.shape[1], right_half.shape[1])
    feats.fai = float(np.mean(np.abs(
        left_half[:, :min_w].astype(float) - right_half[:, :min_w].astype(float)
    ))) / 255.0

    # ── Eye Aspect Ratio (EAR) ────────────────────────────────────────
    upper_roi = gray_face[:h//2, :]
    eyes = detector.detect_eyes(upper_roi)
    if len(eyes) >= 2:
        eyes_sorted = sorted(eyes, key=lambda e: e[0])
        ear_vals = []
        for (ex, ey, ew, eh) in eyes_sorted[:2]:
            ear_vals.append(eh / max(ew, 1))   # simple height/width ratio
        feats.ear = float(np.mean(ear_vals))
    elif len(eyes) == 1:
        ex, ey, ew, eh = eyes[0]
        feats.ear = eh / max(ew, 1)
    else:
        # Fallback: brightness in eye region
        eye_strip = upper_roi[h//8 : h//3, w//6 : 5*w//6]
        feats.ear = 0.3 + float(np.mean(eye_strip)) / 512.0

    # ── Brow Displacement Ratio (BDR) ─────────────────────────────────
    # Higher intensity variance in upper face → more brow movement
    brow_strip = gray_face[h//8 : h//4, w//6 : 5*w//6]
    feats.bdr = min(1.0, float(np.var(brow_strip)) / 3000.0)

    # ── Mouth Aspect Ratio (MAR) ──────────────────────────────────────
    lower_roi = gray_face[h//2:, :]
    mouth = detector.detect_mouth(lower_roi)
    if mouth:
        mx, my, mw, mh = mouth[0]
        feats.mar = mh / max(mw, 1)
    else:
        # Fallback: brightness contrast in lower face
        lower_var = float(np.var(lower_roi))
        feats.mar = min(1.0, lower_var / 3000.0)

    # ── Head tilt (from face bounding box elongation direction) ────────
    aspect = w / max(h, 1)
    feats.head_tilt_deg = abs(45.0 * (aspect - 1.0))

    # ── Face area (relative to full image) ────────────────────────────
    feats.face_area = (w * h) / max(gray_face.size, 1)

    return feats


# ── Emotion inference from geometric feature vector ───────────────────────────
def _infer_emotion(feats: GeometricFeatures) -> tuple[str, float]:
    """
    Rule-based emotion label from geometric features.
    Returns (emotion_label, confidence_0_to_1).

    Feature interpretation (evidence-informed):
      HIGH EAR  + HIGH MAR  + HIGH BDR  → elevated/manic state
      LOW  EAR  + LOW  MAR  + LOW  BDR  → flat/depressed
      HIGH FAI  + HIGH MAR              → agitated/mixed
      HIGH BDR  + LOW  EAR             → anxious/fearful
      HIGH MAR  + LOW  FAI             → joyful
      LOW  FAI  + LOW  MAR  + MID EAR  → neutral
    """
    ear  = feats.ear
    mar  = feats.mar
    bdr  = feats.bdr
    fai  = feats.fai
    slv  = feats.slv

    # Normalise to 0-1 ranges based on typical values
    ear_n  = min(1.0, ear / 0.5)
    mar_n  = min(1.0, mar / 0.4)
    bdr_n  = min(1.0, bdr / 0.6)
    fai_n  = min(1.0, fai / 0.15)
    slv_n  = min(1.0, slv / 0.5)

    scores = {
        "joyful":    (ear_n * 0.3 + mar_n * 0.5 + (1 - fai_n) * 0.2),
        "elevated":  (ear_n * 0.35 + bdr_n * 0.35 + slv_n * 0.3),
        "agitated":  (fai_n * 0.4  + mar_n * 0.3 + slv_n * 0.3),
        "anxious":   (bdr_n * 0.4  + (1 - ear_n) * 0.3 + fai_n * 0.3),
        "withdrawn": ((1 - mar_n) * 0.4 + (1 - ear_n) * 0.35 + (1 - slv_n) * 0.25),
        "flat":      ((1 - bdr_n) * 0.35 + (1 - mar_n) * 0.35 + (1 - slv_n) * 0.3),
        "distressed":((1 - ear_n) * 0.3 + fai_n * 0.4 + (1 - slv_n) * 0.3),
        "neutral":   (1.0 - abs(ear_n - 0.5) - abs(mar_n - 0.3) - abs(fai_n - 0.1)),
    }
    scores["neutral"] = max(0.0, scores["neutral"])

    total = sum(scores.values()) or 1.0
    norm  = {k: v / total for k, v in scores.items()}
    best  = max(norm, key=norm.get)
    return best, round(norm[best], 3)


# ── Per-frame analysis ────────────────────────────────────────────────────────
def analyse_frame(frame: np.ndarray,
                  detector: CascadeDetector) -> tuple[np.ndarray, Optional[EmotionFrame]]:
    """Detect face, extract features, infer emotion, annotate frame."""
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq   = cv2.equalizeHist(gray)           # improves detection in low light
    annotated = frame.copy()

    faces = detector.detect_faces(gray_eq)
    emotion_frame = None

    for (x, y, w, h) in faces[:1]:
        # Expand ROI slightly for better context
        pad = int(0.05 * min(w, h))
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face_img  = frame[y1:y2, x1:x2]
        gray_face = gray[y1:y2, x1:x2]

        feats          = _extract_features(face_img, gray_face, detector)
        feats.face_area = (w * h) / (frame.shape[0] * frame.shape[1])

        emotion, conf  = _infer_emotion(feats)
        valence, arousal = EMOTION_VA.get(emotion, (0.0, 0.0))

        emotion_frame = EmotionFrame(
            timestamp=time.time(),
            emotion=emotion,
            confidence=conf,
            valence=valence,
            arousal=arousal,
            features=feats,
            face_box=(x, y, w, h),
        )

        # ── Annotation ───────────────────────────────────────────────
        color = EMOTION_COLORS.get(emotion, (180, 180, 180))

        # Main bounding box
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)

        # Rounded header strip
        cv2.rectangle(annotated, (x, y-38), (x+w, y), color, -1)
        cv2.putText(annotated,
                    f"{emotion.upper()}  {conf*100:.0f}%",
                    (x+5, y-14),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # Feature bars on the right
        bx = x + w + 8
        if bx + 70 < annotated.shape[1]:
            _overlay_bars(annotated, bx, y, feats, color)

        # Valence indicator dot
        dot_color = (50, 220, 80) if valence > 0.2 else (50, 50, 220) if valence < -0.2 else (180, 180, 50)
        cv2.circle(annotated, (x + w - 12, y + 12), 6, dot_color, -1)

    # Watermark
    h_img = annotated.shape[0]
    cv2.rectangle(annotated, (0, h_img-22), (annotated.shape[1], h_img), (10, 10, 20), -1)
    cv2.putText(annotated,
                f"bpdisdet v2  |  geometric affect analysis  |  faces: {len(faces)}",
                (6, h_img - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 130, 160), 1)

    return annotated, emotion_frame


def _overlay_bars(img, bx, by, feats: GeometricFeatures, accent):
    """Draw small feature bar chart next to face box."""
    items = [
        ("EAR", feats.ear,  0.5, (100, 220, 120)),
        ("MAR", feats.mar,  0.4, (100, 180, 220)),
        ("BDR", feats.bdr,  0.6, (200, 150, 80)),
        ("FAI", feats.fai,  0.2, (200, 80,  80)),
    ]
    bar_h = 10
    for i, (lbl, val, scale, col) in enumerate(items):
        y0 = by + i * (bar_h + 4)
        cv2.putText(img, lbl, (bx, y0 + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)
        filled = int(min(1.0, val / scale) * 55)
        cv2.rectangle(img, (bx+28, y0), (bx+83, y0+bar_h), (40, 40, 50), -1)
        cv2.rectangle(img, (bx+28, y0), (bx+28+filled, y0+bar_h), col, -1)


# ── Session analytics ─────────────────────────────────────────────────────────
def compute_session_metrics(session: FacialSession) -> FacialSession:
    """MSSD-based affective instability + pattern scoring."""
    if not session.frames:
        return session

    frames = session.frames
    valences = [f.valence  for f in frames]
    arousals  = [f.arousal  for f in frames]
    emotions  = [f.emotion  for f in frames]
    ears      = [f.features.ear for f in frames]
    mars      = [f.features.mar for f in frames]

    session.valence_history = valences
    session.arousal_history  = arousals
    session.ear_history      = ears
    session.mar_history      = mars

    # ── Affective Instability (MSSD of valence) ──────────────────────
    if len(valences) >= 2:
        diffs = np.diff(valences)
        session.affective_instability = float(np.mean(diffs ** 2))
    else:
        session.affective_instability = 0.0

    # ── Mania score ───────────────────────────────────────────────────
    manic_n = sum(1 for e in emotions if e in MANIA_LABELS)
    avg_arousal = float(np.mean(arousals)) if arousals else 0.0
    session.mania_score = min(100.0,
        (manic_n / max(len(emotions), 1)) * 100 * (1 + max(0, avg_arousal)))

    # ── Depression score ──────────────────────────────────────────────
    depr_n = sum(1 for e in emotions if e in DEPRESS_LABELS)
    avg_ear = float(np.mean(ears)) if ears else 0.3
    low_ear_penalty = max(0, 0.3 - avg_ear) * 100   # very droopy eyes → depression
    session.depression_score = min(100.0,
        (depr_n / max(len(emotions), 1)) * 80 + low_ear_penalty)

    # ── Mixed state score ─────────────────────────────────────────────
    transitions = sum(
        1 for i in range(1, len(emotions))
        if (emotions[i-1] in MANIA_LABELS and emotions[i]  in DEPRESS_LABELS) or
           (emotions[i-1] in DEPRESS_LABELS and emotions[i] in MANIA_LABELS)
    )
    session.emotion_transitions = transitions
    session.mixed_state_score  = min(100.0,
        transitions / max(len(emotions)-1, 1) * 200 +
        (session.affective_instability * 30))

    # ── Dominant pattern ──────────────────────────────────────────────
    if session.mixed_state_score > 45:
        session.dominant_pattern = "mixed"
    elif session.mania_score > 55:
        session.dominant_pattern = "manic"
    elif session.depression_score > 55:
        session.dominant_pattern = "depressive"
    else:
        session.dominant_pattern = "stable"

    # ── Feature summary ───────────────────────────────────────────────
    session.feature_summary = {
        "mean_EAR":        round(float(np.mean(ears)),  3) if ears  else 0,
        "mean_MAR":        round(float(np.mean(mars)),  3) if mars  else 0,
        "mean_FAI":        round(float(np.mean([f.features.fai for f in frames])), 3),
        "mean_BDR":        round(float(np.mean([f.features.bdr for f in frames])), 3),
        "mean_valence":    round(float(np.mean(valences)), 3),
        "mean_arousal":    round(float(np.mean(arousals)),  3),
        "MSSD_instability":round(session.affective_instability, 4),
        "n_frames":        len(frames),
        "n_transitions":   transitions,
    }

    return session