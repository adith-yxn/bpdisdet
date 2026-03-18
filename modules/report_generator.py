"""
report_generator.py  ·  bpdisdet v2
Generates a comprehensive clinical PDF screening report.
"""

import re
from datetime import datetime
from typing import Optional
from modules.screening_engine import ScreeningResult


def generate_pdf_report(
    result: ScreeningResult,
    patient_name: str = "Anonymous",
    dob: str = "Not provided",
    clinician: str = "Not specified",
) -> bytes:
    try:
        from fpdf import FPDF
    except ImportError:
        return b""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    RISK_RGB = {
        "high":     (220, 70,  70),
        "moderate": (210, 140, 30),
        "low":      (60,  175, 110),
        "minimal":  (80,  140, 210),
    }
    r_col, g_col, b_col = RISK_RGB.get(result.overall_risk, (150, 150, 150))

    # ── Header ──────────────────────────────────────────────────────────
    pdf.set_fill_color(18, 26, 48)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_text_color(45, 212, 191)
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_xy(10, 5)
    pdf.cell(0, 9, "bpdisdet  v2  |  Multimodal Mental Health Screening", ln=True)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(140, 160, 180)
    pdf.set_x(10)
    pdf.cell(0, 6, "Bipolar Spectrum Disorder Early Detection  ·  SDG-3 Aligned  ·  FOR CLINICAL REVIEW ONLY", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    # ── Disclaimer ────────────────────────────────────────────────────
    pdf.set_fill_color(255, 240, 200)
    pdf.set_draw_color(210, 160, 20)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_x(10)
    pdf.multi_cell(190, 5,
        "DISCLAIMER: This report is produced by an AI-assisted screening tool and does NOT "
        "constitute a clinical diagnosis. All results require review by a licensed mental health "
        "professional. This tool supports — it does not replace — clinical judgment.",
        border=1, fill=True)
    pdf.ln(5)

    # ── Patient Info ──────────────────────────────────────────────────
    _section_header(pdf, "Patient & Session Information")
    modalities = []
    if result.has_facial:        modalities.append("Facial Affect")
    if result.has_text:          modalities.append("Linguistic")
    if result.has_questionnaire: modalities.append("Questionnaires")

    _info_row(pdf, "Patient Name",     patient_name)
    _info_row(pdf, "Date of Birth",    dob)
    _info_row(pdf, "Clinician",        clinician)
    _info_row(pdf, "Session ID",       result.session_id)
    _info_row(pdf, "Report Date",      datetime.fromtimestamp(result.timestamp).strftime("%Y-%m-%d %H:%M"))
    _info_row(pdf, "Modalities Used",  " + ".join(modalities) or "None")
    pdf.ln(4)

    # ── Risk Banner ───────────────────────────────────────────────────
    pdf.set_fill_color(r_col, g_col, b_col)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_x(10)
    pdf.cell(190, 11,
             f"  RISK LEVEL: {result.overall_risk.upper()}   |   "
             f"DOMINANT PATTERN: {result.dominant_state.upper()}   |   "
             f"CONFIDENCE: {result.confidence_pct:.0f}%",
             ln=True, fill=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # ── Composite Scores ──────────────────────────────────────────────
    _section_header(pdf, "Composite Screening Scores")
    _score_bar(pdf, "Mania Index",              result.composite_mania)
    _score_bar(pdf, "Depression Index",         result.composite_depression)
    _score_bar(pdf, "Mixed State Index",        result.composite_mixed)
    _score_bar(pdf, "Affective Instability",    result.affective_instability)
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_x(10)
    pdf.cell(0, 5,
             f"Confidence: {result.confidence_pct:.0f}% (capped at 88% — screening tool limitation)", ln=True)
    pdf.ln(4)

    # ── Questionnaire Subscores ───────────────────────────────────────
    if result.has_questionnaire and result.questionnaire_result:
        qr = result.questionnaire_result
        _section_header(pdf, "Validated Questionnaire Scores")
        _score_bar(pdf, f"MDQ-7 (Mania Screen)  [{qr.mdq_raw_score}/28 — {qr.mdq_severity}]",
                   qr.mdq_scaled)
        _score_bar(pdf, f"PHQ-9 (Depression)    [{qr.phq9_raw_score}/27 — {qr.phq9_severity}]",
                   qr.phq9_scaled)
        _score_bar(pdf, f"ALS-SF (Lability)     [{qr.als_raw_score}/18 — {qr.als_severity}]",
                   qr.als_scaled)
        if qr.phq9_safety_flag:
            pdf.set_fill_color(255, 220, 220)
            pdf.set_x(10)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(190, 6, "  !! PHQ-9 ITEM 9 (SUICIDAL IDEATION) ENDORSED — IMMEDIATE EVALUATION REQUIRED",
                     ln=True, fill=True, border=1)
        pdf.ln(4)

    # ── Text Analysis ─────────────────────────────────────────────────
    if result.has_text and result.text_results:
        _section_header(pdf, "Linguistic Analysis Summary")
        tr = result.text_results[-1]
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_x(10)
        clean = re.sub(r"\*\*(.*?)\*\*", r"\1", tr.clinical_summary)
        pdf.multi_cell(190, 5, clean)
        pdf.ln(3)
        if tr.key_phrases:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_x(10)
            pdf.cell(0, 5, "Key linguistic markers: " + ", ".join(
                [re.sub(r"[^\x00-\x7F]+","", p) for p in tr.key_phrases[:8]]
            ), ln=True)
        pdf.ln(3)

    # ── Clinical Summary ──────────────────────────────────────────────
    _section_header(pdf, "Overall Clinical Summary")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(10)
    clean_sum = re.sub(r"\*\*(.*?)\*\*", r"\1", result.clinical_summary)
    pdf.multi_cell(190, 6, clean_sum)
    pdf.ln(4)

    # ── Red Flags ─────────────────────────────────────────────────────
    if result.red_flags:
        pdf.set_fill_color(255, 225, 225)
        pdf.set_draw_color(200, 60, 60)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_x(10)
        pdf.cell(190, 7, "  CRITICAL FLAGS", ln=True, fill=True, border=1)
        pdf.set_font("Helvetica", "", 9)
        for flag in result.red_flags:
            pdf.set_x(14)
            pdf.multi_cell(186, 5, "• " + re.sub(r"[^\x00-\x7F]+", "!", flag))
        pdf.ln(3)

    # ── Recommendations ───────────────────────────────────────────────
    _section_header(pdf, "Recommended Actions")
    pdf.set_font("Helvetica", "", 9.5)
    for i, rec in enumerate(result.recommendations, 1):
        clean_r = re.sub(r"[^\x00-\x7F]+", "", rec).strip()
        clean_r = re.sub(r"^[🔴💡⚠️✅]\s*", "", clean_r)
        pdf.set_x(10)
        pdf.multi_cell(190, 5.5, f"{i}. {clean_r}")
        pdf.ln(1)

    # ── Footer ────────────────────────────────────────────────────────
    pdf.set_y(-18)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 5,
             f"bpdisdet v2  |  SDG-3 Mental Health Screening  |  Session {result.session_id}  |  "
             f"{datetime.now().strftime('%Y-%m-%d')}  |  Not a clinical diagnosis",
             align="C")
    return pdf.output()


def _section_header(pdf, title: str):
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(20, 80, 120)
    pdf.cell(0, 7, title, ln=True)
    pdf.set_draw_color(180, 200, 220)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def _info_row(pdf, label: str, value: str):
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_x(10)
    pdf.cell(55, 6, f"{label}:")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, str(value), ln=True)


def _score_bar(pdf, label: str, score: float):
    from fpdf import FPDF
    if score < 33:   r, g, b = 60, 175, 110
    elif score < 65: r, g, b = 210, 140, 30
    else:            r, g, b = 220, 70, 70

    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_x(10)
    pdf.cell(80, 6, label)
    bar_y = pdf.get_y() + 1
    pdf.set_fill_color(220, 220, 220)
    pdf.rect(92, bar_y, 90, 4, "F")
    filled = int(90 * min(score, 100) / 100)
    pdf.set_fill_color(r, g, b)
    if filled > 0:
        pdf.rect(92, bar_y, filled, 4, "F")
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_x(185)
    pdf.cell(20, 6, f"{score:.0f}/100", ln=True)