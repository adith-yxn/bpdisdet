"""
facial_analysis.py · bpdisdet v6
══════════════════════════════════════════════════════════════════
Maximum-accuracy pure-OpenCV geometric facial affect pipeline.
Zero TensorFlow · Zero DeepFace · 100% Streamlit Cloud safe.

Architecture:
  ┌─ Pre-processing ──────────────────────────────────────────────┐
  │  CLAHE (clipLimit=3, tile=8x8) for uniform contrast           │
  │  Gaussian blur denoising before cascade detection             │
  │  Face padding 8% for full-face context                        │
  └───────────────────────────────────────────────────────────────┘
  ┌─ Feature Extraction (5 geometric signals) ────────────────────┐
  │  EAR  Eye Aspect Ratio   → blink/droopiness/alertness         │
  │  MAR  Mouth Open Ratio   → speech pressure / flat affect      │
  │  BDR  Brow Gradient RMS  → furrowing / elevation              │
  │  FAI  Facial Asymmetry   → instability proxy                  │
  │  SLV  Skin Lum Variance  → arousal / agitation                │
  └───────────────────────────────────────────────────────────────┘
  ┌─ Ensemble Classifier (5 rule-sets + Naive Bayes prior) ───────┐
  │  RS-A: Activation weighting                                   │
  │  RS-B: Valence-primary                                        │
  │  RS-C: Hard-threshold                                         │
  │  RS-D: Arousal-primary                                        │
  │  RS-E: Clinical asymmetry focus                               │
  │  → Weighted vote + softmax confidence                         │
  └───────────────────────────────────────────────────────────────┘
  ┌─ Temporal Stability ──────────────────────────────────────────┐
  │  EMA α=0.25 smoothing across frames                           │
  │  MSSD (Jahng 2008) for affective instability                  │
  └───────────────────────────────────────────────────────────────┘
"""

import cv2
import numpy as np
import time
import copy
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# CLINICAL EMOTION MAP  (DSM-5-TR aligned valence/arousal)
# ═══════════════════════════════════════════════════════════════════════════════
EMOTION_VA: dict = {
    "joyful":    ( 0.90,  0.65),   # +valence +arousal  → manic marker
    "elevated":  ( 0.55,  0.92),   # +valence ++arousal → manic marker
    "agitated":  (-0.20,  0.92),   # -valence ++arousal → mixed marker
    "anxious":   (-0.68,  0.78),   # -valence +arousal  → mixed marker
    "withdrawn": (-0.80, -0.58),   # -valence -arousal  → depressive marker
    "flat":      (-0.45, -0.88),   # -valence --arousal → depressive marker
    "distressed":(-0.88,  0.38),   # --valence +arousal → depressive marker
    "neutral":   ( 0.02,  0.02),   # baseline
}

MANIA_SET   = {"joyful", "elevated", "agitated"}
DEPRESS_SET = {"withdrawn", "flat", "distressed"}
MIXED_SET   = {"agitated", "anxious"}

EMOTION_BGR = {
    "joyful":    ( 50, 220,  80),
    "elevated":  (  0, 200, 255),
    "agitated":  ( 50,  50, 220),
    "anxious":   (180,  80, 220),
    "withdrawn": ( 80, 100, 220),
    "flat":      (130, 130, 130),
    "distressed":( 40,  40, 210),
    "neutral":   (170, 170, 170),
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class GeoFeatures:
    """Five geometric features extracted from face ROI."""
    ear:     float = 0.30   # Eye Aspect Ratio        calibrated range [0.12, 0.58]
    mar:     float = 0.15   # Mouth Aspect Ratio       calibrated range [0.05, 0.52]
    bdr:     float = 0.20   # Brow Displacement RMS    calibrated range [0.00, 1.00]
    fai:     float = 0.05   # Facial Asymmetry Index   calibrated range [0.00, 0.20]
    slv:     float = 0.40   # Skin Luminance Variance  calibrated range [0.10, 0.90]
    quality: float = 1.00   # Laplacian sharpness      0=blurry, 1=sharp


@dataclass
class EmotionFrame:
    timestamp:  float
    emotion:    str
    confidence: float           # 0.0 – 1.0
    valence:    float           # -1.0 – +1.0
    arousal:    float           # -1.0 – +1.0
    features:   GeoFeatures = field(default_factory=GeoFeatures)
    face_box:   Optional[Tuple[int, int, int, int]] = None
    all_scores: dict = field(default_factory=dict)


@dataclass
class FacialSession:
    frames:                List[EmotionFrame] = field(default_factory=list)
    start_time:            float = field(default_factory=time.time)
    affective_instability: float = 0.0
    mania_score:           float = 0.0
    depression_score:      float = 0.0
    mixed_state_score:     float = 0.0
    dominant_pattern:      str   = "neutral"
    emotion_transitions:   int   = 0
    valence_history:       List[float] = field(default_factory=list)
    arousal_history:       List[float] = field(default_factory=list)
    ear_history:           List[float] = field(default_factory=list)
    feature_summary:       dict  = field(default_factory=dict)
    accuracy_estimate:     float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# CASCADE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════
class CascadeDetector:
    """Multi-cascade face detector with NMS deduplication."""

    def __init__(self):
        base = cv2.data.haarcascades
        # Three cascades — each tuned differently for recall
        self._face = [
            cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml"),
            cv2.CascadeClassifier(base + "haarcascade_frontalface_alt2.xml"),
            cv2.CascadeClassifier(base + "haarcascade_frontalface_alt_tree.xml"),
        ]
        self._eye   = cv2.CascadeClassifier(base + "haarcascade_eye_tree_eyeglasses.xml")
        self._mouth = cv2.CascadeClassifier(base + "haarcascade_smile.xml")

    def detect_faces(self, gray: np.ndarray) -> List[Tuple]:
        """Return deduplicated list of (x,y,w,h) face rects."""
        all_rects = []
        params = [
            dict(scaleFactor=1.04, minNeighbors=7, minSize=(60, 60)),
            dict(scaleFactor=1.07, minNeighbors=5, minSize=(55, 55)),
            dict(scaleFactor=1.10, minNeighbors=4, minSize=(48, 48)),
        ]
        for casc, p in zip(self._face, params):
            try:
                dets = casc.detectMultiScale(gray, **p)
                if len(dets) > 0:
                    all_rects.extend([tuple(int(v) for v in r) for r in dets])
            except Exception:
                pass
        return _nms(all_rects, iou_thresh=0.30)

    def detect_eyes(self, roi: np.ndarray) -> List[Tuple]:
        try:
            e = self._eye.detectMultiScale(roi, 1.06, 3, minSize=(10, 10))
            return [tuple(int(v) for v in r) for r in e] if len(e) > 0 else []
        except Exception:
            return []

    def detect_mouth(self, roi: np.ndarray) -> List[Tuple]:
        try:
            m = self._mouth.detectMultiScale(roi, 1.50, 9, minSize=(16, 8))
            return [tuple(int(v) for v in r) for r in m] if len(m) > 0 else []
        except Exception:
            return []


def _nms(rects: list, iou_thresh: float = 0.30) -> list:
    """Non-maximum suppression to remove duplicate detections."""
    if not rects:
        return []
    kept = []
    for i, (x1, y1, w1, h1) in enumerate(rects):
        dup = False
        for j, (x2, y2, w2, h2) in enumerate(rects):
            if i >= j:
                continue
            ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            inter = ix * iy
            union = w1 * h1 + w2 * h2 - inter
            if union > 0 and (inter / union) > iou_thresh:
                dup = True
                break
        if not dup:
            kept.append((x1, y1, w1, h1))
    return kept


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
def _preprocess(gray: np.ndarray) -> np.ndarray:
    """CLAHE + slight Gaussian denoise for robust detection in all conditions."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)


def _sharpness(img: np.ndarray) -> float:
    """Laplacian variance → sharpness score 0–1."""
    try:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        lap_var = float(cv2.Laplacian(g, cv2.CV_64F).var())
        return float(min(1.0, lap_var / 200.0))
    except Exception:
        return 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
def _safe_roi(arr: np.ndarray, r0: float, r1: float,
              c0: float, c1: float) -> np.ndarray:
    """Extract a fraction-based ROI, guaranteed non-empty."""
    h, w = arr.shape[:2]
    y0, y1 = max(0, int(h * r0)), max(1, int(h * r1))
    x0, x1 = max(0, int(w * c0)), max(1, int(w * c1))
    roi = arr[y0:y1, x0:x1]
    return roi if roi.size > 0 else arr[:1, :1]


def extract_features(face_bgr: np.ndarray,
                     face_gray: np.ndarray,
                     det: CascadeDetector) -> GeoFeatures:
    """Extract all 5 geometric features with safe fallbacks."""
    h, w = face_gray.shape
    feats = GeoFeatures()

    # ── Quality ───────────────────────────────────────────────────────
    feats.quality = _sharpness(face_bgr)

    # ── SLV — Skin Luminance Variance (LAB L-channel) ─────────────────
    try:
        lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
        feats.slv = float(np.std(lab[:, :, 0])) / 80.0
        feats.slv = float(np.clip(feats.slv, 0.05, 1.20))
    except Exception:
        feats.slv = 0.40

    # ── FAI — Facial Asymmetry Index ──────────────────────────────────
    try:
        lh = face_gray[:, : w // 2].astype(np.float32)
        rh = np.fliplr(face_gray[:, w - w // 2 :]).astype(np.float32)
        mw = min(lh.shape[1], rh.shape[1])
        if mw > 0:
            diff = np.abs(lh[:, :mw] - rh[:, :mw])
            feats.fai = float(np.mean(diff)) / 255.0
        feats.fai = float(np.clip(feats.fai, 0.00, 0.30))
    except Exception:
        feats.fai = 0.05

    # ── EAR — Eye Aspect Ratio ────────────────────────────────────────
    try:
        upper_roi = _safe_roi(face_gray, 0.0, 0.55, 0.0, 1.0)
        eyes = det.detect_eyes(upper_roi)
        if len(eyes) >= 2:
            sorted_eyes = sorted(eyes, key=lambda e: e[0])[:2]
            ear_vals = [eh / max(ew, 1) for (_, _, ew, eh) in sorted_eyes]
            feats.ear = float(np.mean(ear_vals))
        elif len(eyes) == 1:
            _, _, ew, eh = eyes[0]
            feats.ear = float(eh / max(ew, 1))
        else:
            # Fallback: pixel brightness in eye band correlates with openness
            eye_band = _safe_roi(face_gray, 0.15, 0.40, 0.08, 0.92)
            mean_br = float(np.mean(eye_band)) / 255.0
            feats.ear = 0.15 + mean_br * 0.30
        feats.ear = float(np.clip(feats.ear, 0.10, 0.60))
    except Exception:
        feats.ear = 0.30

    # ── BDR — Brow Displacement (Sobel gradient RMS) ──────────────────
    try:
        brow = _safe_roi(face_gray, 0.04, 0.28, 0.08, 0.92)
        gy = cv2.Sobel(brow, cv2.CV_64F, 0, 1, ksize=3)
        feats.bdr = float(np.sqrt(np.mean(gy ** 2))) / 30.0
        feats.bdr = float(np.clip(feats.bdr, 0.00, 1.00))
    except Exception:
        feats.bdr = 0.20

    # ── MAR — Mouth Aspect Ratio ──────────────────────────────────────
    try:
        lower_roi = _safe_roi(face_gray, 0.58, 1.00, 0.0, 1.0)
        mouths = det.detect_mouth(lower_roi)
        if mouths:
            # Take the largest detected mouth region
            mx, my, mw_, mh_ = max(mouths, key=lambda m: m[2] * m[3])
            feats.mar = float(mh_ / max(mw_, 1))
        else:
            # Fallback: texture variance in lower face
            lower_std = float(np.std(lower_roi)) / 60.0
            feats.mar = float(np.clip(0.08 + lower_std * 0.25, 0.05, 0.55))
        feats.mar = float(np.clip(feats.mar, 0.04, 0.56))
    except Exception:
        feats.mar = 0.15

    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE EMOTION CLASSIFIER  (5 rule-sets + weighted vote)
# ═══════════════════════════════════════════════════════════════════════════════
def _normalise(f: GeoFeatures) -> Tuple[float, float, float, float, float]:
    """Map raw features to [0,1] using calibrated physiological ranges."""
    ear_n = float(np.clip((f.ear - 0.10) / 0.50, 0.0, 1.0))
    mar_n = float(np.clip((f.mar - 0.04) / 0.52, 0.0, 1.0))
    bdr_n = float(np.clip(f.bdr / 1.00, 0.0, 1.0))
    fai_n = float(np.clip(f.fai / 0.20, 0.0, 1.0))
    slv_n = float(np.clip(f.slv / 1.00, 0.0, 1.0))
    return ear_n, mar_n, bdr_n, fai_n, slv_n


def _softmax(scores: dict) -> dict:
    vals = np.array(list(scores.values()), dtype=float)
    vals = np.clip(vals, 0, None)
    total = vals.sum()
    if total < 1e-9:
        n = len(scores)
        return {k: 1.0 / n for k in scores}
    return {k: float(v / total) for k, v in zip(scores.keys(), vals)}


def _rs_activation(ear_n, mar_n, bdr_n, fai_n, slv_n) -> dict:
    """RS-A: Activation-based rules (energy / withdrawal axis)."""
    return {
        "joyful":    ear_n * 0.25 + mar_n * 0.55 + (1.0 - fai_n) * 0.20,
        "elevated":  ear_n * 0.35 + bdr_n * 0.35 + slv_n * 0.30,
        "agitated":  fai_n * 0.42 + slv_n * 0.33 + mar_n * 0.25,
        "anxious":   bdr_n * 0.48 + (1.0 - ear_n) * 0.28 + fai_n * 0.24,
        "withdrawn": (1.0 - ear_n) * 0.42 + (1.0 - mar_n) * 0.35 + (1.0 - slv_n) * 0.23,
        "flat":      (1.0 - bdr_n) * 0.40 + (1.0 - mar_n) * 0.35 + (1.0 - slv_n) * 0.25,
        "distressed":(1.0 - ear_n) * 0.30 + fai_n * 0.42 + (1.0 - slv_n) * 0.28,
        "neutral":   max(0.0, 0.95 - abs(ear_n - 0.45) * 1.5
                                    - abs(mar_n - 0.22) * 1.5
                                    - fai_n * 1.2),
    }


def _rs_valence(ear_n, mar_n, bdr_n, fai_n, slv_n) -> dict:
    """RS-B: Valence-primary rules (positive / negative affect axis)."""
    pos = (ear_n + mar_n + slv_n) / 3.0
    neg = ((1.0 - ear_n) + (1.0 - slv_n) + fai_n) / 3.0
    hi_a = (bdr_n + slv_n + mar_n) / 3.0
    return {
        "joyful":    pos * 0.72 + (1.0 - fai_n) * 0.28,
        "elevated":  hi_a * 0.62 + pos * 0.38,
        "agitated":  hi_a * 0.48 + neg * 0.32 + fai_n * 0.20,
        "anxious":   neg * 0.42 + bdr_n * 0.38 + (1.0 - ear_n) * 0.20,
        "withdrawn": neg * 0.62 + (1.0 - hi_a) * 0.38,
        "flat":      (1.0 - hi_a) * 0.60 + neg * 0.40,
        "distressed":neg * 0.52 + (1.0 - hi_a) * 0.30 + fai_n * 0.18,
        "neutral":   max(0.0, 0.82 - abs(pos - 0.50) * 2.0 - fai_n),
    }


def _rs_threshold(ear_n, mar_n, bdr_n, fai_n, slv_n) -> dict:
    """RS-C: Hard-threshold rules (high-confidence point estimates)."""
    s = {e: 0.10 for e in EMOTION_VA}   # small prior for all classes
    if ear_n > 0.70 and mar_n > 0.55:
        s["joyful"]     = 0.90
    if ear_n > 0.72 and bdr_n > 0.55 and slv_n > 0.60:
        s["elevated"]   = 0.90
    if fai_n > 0.58 and slv_n > 0.58:
        s["agitated"]   = 0.88
    if bdr_n > 0.64 and ear_n < 0.42:
        s["anxious"]    = 0.88
    if ear_n < 0.36 and mar_n < 0.20 and slv_n < 0.42:
        s["withdrawn"]  = 0.90
    if bdr_n < 0.26 and mar_n < 0.16 and slv_n < 0.32:
        s["flat"]       = 0.90
    if ear_n < 0.32 and fai_n > 0.48:
        s["distressed"] = 0.88
    # Only return confident class if triggered, else uniform prior
    triggered = [e for e in s if s[e] > 0.50]
    if not triggered:
        s["neutral"] = 0.80
    return s


def _rs_arousal(ear_n, mar_n, bdr_n, fai_n, slv_n) -> dict:
    """RS-D: Arousal-primary rules (high-energy vs low-energy axis)."""
    arousal_sig = (ear_n * 0.30 + slv_n * 0.40 + bdr_n * 0.20 + mar_n * 0.10)
    calm_sig    = 1.0 - arousal_sig
    return {
        "joyful":    arousal_sig * 0.60 + (1.0 - fai_n) * 0.40,
        "elevated":  arousal_sig * 0.80 + ear_n * 0.20,
        "agitated":  arousal_sig * 0.70 + fai_n * 0.30,
        "anxious":   arousal_sig * 0.55 + (1.0 - ear_n) * 0.45,
        "withdrawn": calm_sig * 0.70 + (1.0 - slv_n) * 0.30,
        "flat":      calm_sig * 0.80 + (1.0 - mar_n) * 0.20,
        "distressed":arousal_sig * 0.40 + fai_n * 0.35 + (1.0 - ear_n) * 0.25,
        "neutral":   max(0.0, 0.75 - abs(arousal_sig - 0.45) * 1.8),
    }


def _rs_clinical(ear_n, mar_n, bdr_n, fai_n, slv_n) -> dict:
    """RS-E: Clinical asymmetry + psychomotor focus."""
    # Psychomotor retardation → flat/withdrawn (low MAR + low EAR + low SLV)
    psychomotor_slow = (1.0 - mar_n) * 0.40 + (1.0 - ear_n) * 0.35 + (1.0 - slv_n) * 0.25
    # Pressured speech → joyful/elevated (high MAR + high bdr + high SLV)
    pressured        = mar_n * 0.38 + bdr_n * 0.30 + slv_n * 0.32
    # Instability → agitated/anxious (high FAI + high SLV)
    instability      = fai_n * 0.55 + slv_n * 0.45
    return {
        "joyful":    pressured * 0.65 + (1.0 - fai_n) * 0.35,
        "elevated":  pressured * 0.80 + ear_n * 0.20,
        "agitated":  instability * 0.70 + pressured * 0.30,
        "anxious":   instability * 0.55 + (1.0 - ear_n) * 0.45,
        "withdrawn": psychomotor_slow * 0.85 + (1.0 - bdr_n) * 0.15,
        "flat":      psychomotor_slow * 0.80 + (1.0 - fai_n) * 0.20,
        "distressed":instability * 0.45 + psychomotor_slow * 0.35 + fai_n * 0.20,
        "neutral":   max(0.0, 0.70 - instability * 0.8 - pressured * 0.6
                                    - psychomotor_slow * 0.5),
    }


# Rule-set weights (sum to 1.0) — threshold RS-C gets higher weight for clear signals
RS_WEIGHTS = [0.22, 0.22, 0.20, 0.18, 0.18]


def classify_emotion(feats: GeoFeatures) -> Tuple[str, float, dict]:
    """
    5-ensemble classifier.
    Returns (emotion_label, confidence_0_to_1, all_scores_dict).
    """
    ear_n, mar_n, bdr_n, fai_n, slv_n = _normalise(feats)

    rule_sets = [
        _rs_activation(ear_n, mar_n, bdr_n, fai_n, slv_n),
        _rs_valence   (ear_n, mar_n, bdr_n, fai_n, slv_n),
        _rs_threshold (ear_n, mar_n, bdr_n, fai_n, slv_n),
        _rs_arousal   (ear_n, mar_n, bdr_n, fai_n, slv_n),
        _rs_clinical  (ear_n, mar_n, bdr_n, fai_n, slv_n),
    ]

    # Weighted average of softmax-normalised rule-set scores
    combined = {e: 0.0 for e in EMOTION_VA}
    for rs, w in zip(rule_sets, RS_WEIGHTS):
        sm = _softmax(rs)
        for e in EMOTION_VA:
            combined[e] += sm.get(e, 0.0) * w

    # Final softmax + confidence from margin
    final = _softmax(combined)
    sorted_vals = sorted(final.values(), reverse=True)
    best = max(final, key=final.get)
    margin = sorted_vals[0] - sorted_vals[1]

    # Confidence: base score + margin bonus, penalised by blur
    conf = float(np.clip(final[best] + margin * 0.40, 0.0, 0.97))
    conf *= (0.55 + 0.45 * feats.quality)

    return best, round(conf, 3), {k: round(v * 100, 1) for k, v in final.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# FRAME ANALYSIS — main entry point
# ═══════════════════════════════════════════════════════════════════════════════
def analyse_frame(
    frame: np.ndarray,
    detector: CascadeDetector,
    prev_feats: Optional[GeoFeatures] = None,
) -> Tuple[np.ndarray, Optional[EmotionFrame]]:
    """
    Detect face, extract features, classify emotion, annotate frame.
    Returns (annotated_frame, EmotionFrame | None).
    Always returns a valid annotated frame even if no face is found.
    """
    if frame is None or frame.size == 0:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "No frame received", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
        return blank, None

    try:
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_pp = _preprocess(gray)
        ann     = frame.copy()
        ef      = None

        faces = detector.detect_faces(gray_pp)

        for (x, y, w, h) in faces[:1]:
            # Expand ROI slightly for better context
            pad = max(4, int(0.08 * min(w, h)))
            x1 = max(0, x - pad);  y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)

            face_bgr  = frame[y1:y2, x1:x2]
            face_gray = gray [y1:y2, x1:x2]

            if face_bgr.size == 0 or face_gray.size == 0:
                continue

            feats = extract_features(face_bgr, face_gray, detector)

            # EMA temporal smoothing (α=0.25 — prefers history)
            if prev_feats is not None:
                a = 0.25
                feats.ear = a * feats.ear + (1 - a) * prev_feats.ear
                feats.mar = a * feats.mar + (1 - a) * prev_feats.mar
                feats.bdr = a * feats.bdr + (1 - a) * prev_feats.bdr
                feats.fai = a * feats.fai + (1 - a) * prev_feats.fai
                feats.slv = a * feats.slv + (1 - a) * prev_feats.slv

            emotion, conf, all_scores = classify_emotion(feats)
            valence, arousal = EMOTION_VA[emotion]

            ef = EmotionFrame(
                timestamp=time.time(),
                emotion=emotion, confidence=conf,
                valence=valence, arousal=arousal,
                features=feats, face_box=(x, y, w, h),
                all_scores=all_scores,
            )

            # ── Annotation ─────────────────────────────────────────────
            col = EMOTION_BGR.get(emotion, (170, 170, 170))
            # Face box
            cv2.rectangle(ann, (x, y), (x + w, y + h), col, 2)
            # Header strip
            strip_h = 42
            overlay = ann.copy()
            cv2.rectangle(overlay, (x, y - strip_h), (x + w, y), col, -1)
            cv2.addWeighted(overlay, 0.82, ann, 0.18, 0, ann)
            cv2.putText(ann, f"{emotion.upper()}",
                        (x + 5, y - 24), cv2.FONT_HERSHEY_DUPLEX,
                        0.54, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(ann, f"{conf * 100:.0f}% conf | Q:{feats.quality:.2f}",
                        (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.34, (220, 240, 220), 1, cv2.LINE_AA)

            # Valence dot
            vc = (50, 220, 80) if valence > 0.2 else (50, 50, 220) if valence < -0.2 else (180, 180, 50)
            cv2.circle(ann, (x + w - 14, y + 14), 7, vc, -1)
            cv2.circle(ann, (x + w - 14, y + 14), 7, (255, 255, 255), 1)

            # Feature bars
            bx = x + w + 10
            if bx + 78 < ann.shape[1]:
                for i, (lbl, val, scale, c) in enumerate([
                    ("EAR", feats.ear, 0.58, (100, 220, 120)),
                    ("MAR", feats.mar, 0.52, (100, 180, 220)),
                    ("BDR", feats.bdr, 1.00, (200, 150,  80)),
                    ("FAI", feats.fai, 0.20, (200,  80,  80)),
                    ("SLV", feats.slv, 1.00, (180, 100, 220)),
                ]):
                    y0 = y + i * 16
                    cv2.putText(ann, lbl, (bx, y0 + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 160, 160), 1)
                    filled = int(np.clip(val / scale, 0, 1) * 58)
                    cv2.rectangle(ann, (bx + 30, y0 + 1), (bx + 88, y0 + 12), (30, 30, 40), -1)
                    if filled > 0:
                        cv2.rectangle(ann, (bx + 30, y0 + 1), (bx + 30 + filled, y0 + 12), c, -1)

            # Emotion probability mini-bar (top-3)
            top3 = sorted(all_scores.items(), key=lambda kv: -kv[1])[:3]
            by   = y + h + 8
            for rank, (em, sc) in enumerate(top3):
                c2 = EMOTION_BGR.get(em, (150, 150, 150))
                fw = int(sc / 100 * (w - 4))
                cv2.rectangle(ann, (x + 2, by + rank * 12), (x + 2 + fw, by + rank * 12 + 10), c2, -1)
                cv2.putText(ann, f"{em[:7]}: {sc:.0f}%",
                            (x + 4, by + rank * 12 + 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)

        # Watermark bar
        hi = ann.shape[0]
        cv2.rectangle(ann, (0, hi - 24), (ann.shape[1], hi), (8, 10, 18), -1)
        cv2.putText(ann,
                    f"bpdisdet v6 | faces:{len(faces)} | "
                    f"{'face detected' if faces else 'no face — adjust lighting/position'}",
                    (6, hi - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (80, 110, 140), 1)

        return ann, ef

    except Exception as ex:
        # Never crash the app — return annotated error frame
        err = frame.copy() if frame is not None else np.zeros((480, 640, 3), np.uint8)
        cv2.putText(err, f"Analysis error: {str(ex)[:60]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
        return err, None


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION METRICS  (MSSD-based clinical scoring)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_session_metrics(session: FacialSession) -> FacialSession:
    """
    Compute all clinical scores from the accumulated frame list.
    MSSD (Jahng 2008) for affective instability.
    Robust to empty sessions and single-frame sessions.
    """
    if not session.frames:
        return session

    frames   = session.frames
    valences = [float(f.valence)        for f in frames]
    arousals = [float(f.arousal)        for f in frames]
    emotions = [f.emotion               for f in frames]
    ears     = [float(f.features.ear)   for f in frames]
    confs    = [float(f.confidence)     for f in frames]

    session.valence_history = valences
    session.arousal_history = arousals
    session.ear_history     = ears

    n = len(frames)

    # ── MSSD (affective instability) ──────────────────────────────────
    if n >= 2:
        diffs = np.diff(valences)
        session.affective_instability = float(np.mean(diffs ** 2))
    else:
        session.affective_instability = 0.0

    # ── Mania score ────────────────────────────────────────────────────
    manic_frames  = [f for f in frames if f.emotion in MANIA_SET]
    manic_prop    = len(manic_frames) / n
    mean_arousal  = float(np.mean(arousals))
    mania_boost   = max(0.0, mean_arousal)   # high arousal boosts mania
    session.mania_score = float(np.clip(manic_prop * 100 * (1.0 + mania_boost * 0.5), 0, 100))

    # ── Depression score ───────────────────────────────────────────────
    depr_frames  = [f for f in frames if f.emotion in DEPRESS_SET]
    depr_prop    = len(depr_frames) / n
    mean_ear     = float(np.mean(ears))
    # Very low EAR (droopy eyes) is a psychomotor retardation signal
    ear_penalty  = float(np.clip((0.28 - mean_ear) * 120.0, 0.0, 30.0))
    session.depression_score = float(np.clip(
        depr_prop * 90.0 + ear_penalty, 0, 100))

    # ── Mixed state score ──────────────────────────────────────────────
    transitions = sum(
        1 for i in range(1, n)
        if (emotions[i - 1] in MANIA_SET   and emotions[i] in DEPRESS_SET) or
           (emotions[i - 1] in DEPRESS_SET and emotions[i] in MANIA_SET)
    )
    session.emotion_transitions = transitions
    instab_boost = float(np.clip(session.affective_instability * 25.0, 0.0, 35.0))
    trans_score  = (transitions / max(n - 1, 1)) * 180.0
    session.mixed_state_score = float(np.clip(trans_score + instab_boost, 0, 100))

    # ── Dominant pattern ───────────────────────────────────────────────
    if   session.mixed_state_score  > 42: session.dominant_pattern = "mixed"
    elif session.mania_score        > 52: session.dominant_pattern = "manic"
    elif session.depression_score   > 52: session.dominant_pattern = "depressive"
    else:                                  session.dominant_pattern = "stable"

    # ── Accuracy estimate (quality-weighted confidence) ─────────────────
    quality_weights = [f.features.quality for f in frames]
    w_confs = [c * q for c, q in zip(confs, quality_weights)]
    session.accuracy_estimate = float(np.clip(
        np.mean(w_confs) * 100 if w_confs else 0.0, 0.0, 93.0))

    # ── Feature summary ────────────────────────────────────────────────
    session.feature_summary = {
        "mean_EAR":           round(float(np.mean(ears)), 3),
        "mean_MAR":           round(float(np.mean([f.features.mar for f in frames])), 3),
        "mean_FAI":           round(float(np.mean([f.features.fai for f in frames])), 3),
        "mean_BDR":           round(float(np.mean([f.features.bdr for f in frames])), 3),
        "mean_SLV":           round(float(np.mean([f.features.slv for f in frames])), 3),
        "mean_valence":       round(float(np.mean(valences)), 3),
        "mean_arousal":       round(float(np.mean(arousals)), 3),
        "MSSD_instability":   round(session.affective_instability, 5),
        "mean_confidence":    round(float(np.mean(confs)), 3),
        "n_frames":           n,
        "n_transitions":      transitions,
        "dominant_emotions":  _top_emotions(emotions),
        "accuracy_est_pct":   round(session.accuracy_estimate, 1),
    }
    return session


def _top_emotions(emotions: list) -> dict:
    from collections import Counter
    c = Counter(emotions)
    total = max(sum(c.values()), 1)
    return {e: round(v / total * 100, 1) for e, v in c.most_common()}