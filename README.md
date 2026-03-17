# 🧠 bpdisdet — Multimodal Mental Health Screening Tool

**Bipolar Disorder Detection System** · Supporting UN SDG-3: Good Health and Well-Being

---

## Overview

**bpdisdet** is an open-source, privacy-first mental health screening tool that combines two analytical modalities to identify potential markers of bipolar spectrum disorders:

| Modality | Technology | What It Detects |
|----------|-----------|----------------|
| 🎭 **Facial Affect** | OpenCV + DeepFace | Emotional patterns, affective instability (MSSD), valence/arousal dynamics |
| 📝 **Linguistic Analysis** | Anthropic Claude API (+ local fallback) | Pressured speech, grandiosity, anhedonia, hopelessness, mixed dysphoria |

> ⚠️ **Medical Disclaimer:** bpdisdet is a **screening support tool only**. It does **not** provide clinical diagnoses. All results must be reviewed by a licensed mental health professional.

---

## Features

- ✅ Real-time facial emotion detection (7 emotions) via webcam or uploaded image/video
- ✅ Affective instability quantification using MSSD (Mean Square of Successive Differences)
- ✅ LLM-powered psycholinguistic analysis (Claude API) with local heuristic fallback
- ✅ Multi-entry journal analysis for longitudinal mood pattern detection
- ✅ Composite multimodal risk scoring (mania / depression / mixed state)
- ✅ Printable PDF reports for clinical providers
- ✅ JSON export for research integration
- ✅ Crisis resource integration (SDG-3 equity focus)
- ✅ Works offline with local heuristic fallback (no API key required)

---

## Architecture

```
bpdisdet/
├── app.py                      # Streamlit frontend + UI
├── modules/
│   ├── facial_analysis.py      # OpenCV + DeepFace emotion pipeline
│   ├── text_analysis.py        # LLM + rule-based linguistic analysis
│   ├── screening_engine.py     # Multimodal fusion & risk scoring
│   └── report_generator.py     # PDF report generation (fpdf2)
├── requirements.txt
└── README.md
```

### Multimodal Fusion Weights
```
Composite Score = 0.40 × Facial Affect Score  +  0.60 × Linguistic Score
```
(Text receives higher weight as it contains richer semantic information)

---

## Installation

### Prerequisites
- Python 3.10+
- Webcam (optional, for live capture)

### Setup

```bash
# 1. Clone / download the project
cd bpdisdet

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."   # Linux/macOS
set ANTHROPIC_API_KEY=sk-ant-...       # Windows
```

### Run

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## Usage Guide

### 1. Text Analysis Tab
1. Enter your thoughts / journal entries in the text area
2. Use **Multi-entry journal** mode for longitudinal analysis (up to 3 entries)
3. Try the **Demo** buttons (Manic / Depressed / Mixed) to see sample analysis
4. Click **Analyse Text** — Claude API will provide deep psycholinguistic analysis, or the local heuristic engine will run if no API key is set

### 2. Facial Affect Tab
1. **Upload image**: Upload a photo (JPG/PNG) or short video clip
2. **Live webcam**: Use the camera input for instant capture
3. The system detects faces, classifies 7 emotions, and computes valence/arousal
4. Multiple frames (from video) enable MSSD-based affective instability computation

### 3. Screening Results Tab
1. After completing one or both modalities, click **Compute Composite Screening Result**
2. View the unified risk assessment, composite scores, and clinical summary
3. Review recommendations and any critical flags

### 4. Report & Export Tab
1. Enter patient/clinician details
2. Click **Generate PDF Report** for a printable clinical report
3. Export raw results as JSON for research integration

---

## Clinical Interpretation Guide

| Score Range | Interpretation | Recommended Action |
|-------------|---------------|-------------------|
| 0–20 (Minimal) | No significant markers detected | Routine monitoring |
| 21–44 (Low) | Some markers present, sub-threshold | GP consultation, mood diary |
| 45–69 (Moderate) | Multiple markers present | Psychiatric evaluation within 1 week |
| 70–100 (High) | Strong markers; possible acute episode | Urgent evaluation within 24–48 hours |

### Linguistic Markers Detected

**Mania/Hypomania:**
- Pressured speech (run-on sentences, excessive exclamation)
- Grandiosity (inflated self-concept, special mission language)
- Flight of ideas (abrupt topic shifts, loose associations)
- Decreased sleep references
- Elevated arousal tone

**Depression:**
- Anhedonia markers ("nothing matters", loss of enjoyment)
- Hopelessness/helplessness language
- Psychomotor retardation indicators (short, slow sentences)
- Cognitive slowing complaints
- Somatic symptom references

**Mixed States:**
- Simultaneous dysphoric and elevated elements
- Agitated depression language
- Racing negative thoughts

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit 1.32+ |
| Facial Detection | OpenCV 4.9 (Haar Cascades) |
| Emotion Recognition | DeepFace (FER+ model) |
| Fallback Emotion | Custom heuristic analyser |
| LLM Analysis | Anthropic Claude (`claude-opus-4-5`) |
| Fallback NLP | Rule-based lexical marker engine |
| Visualization | Plotly 5.19 |
| PDF Reports | fpdf2 2.7 |
| Data | NumPy, Pandas |

---

## Privacy & Ethics

- **No data persistence**: All session data is stored in memory only and cleared on browser refresh
- **No external transmission**: When using the local heuristic, no text leaves your device
- **API mode**: Text is sent to Anthropic's API only when an API key is provided
- **Anonymisation**: Patient name fields default to "Anonymous"
- **Research only**: Not validated for clinical deployment

---

## SDG-3 Alignment

bpdisdet is designed to support **UN Sustainable Development Goal 3** (Good Health and Well-Being), specifically Target 3.4 (mental health promotion). Key design principles:

- **Low-resource compatible**: Runs on modest hardware; heuristic fallback requires no internet
- **Low-cost**: Open source; Anthropic API tier provides affordable access
- **Equitable access**: Designed for use in community health settings, telepsychiatry, and low-income contexts
- **Local language adaptable**: Linguistic markers can be extended for regional languages
- **Crisis resource integration**: Built-in helpline references for India and globally

---

## Crisis Resources

| Resource | Contact |
|----------|---------|
| iCall (India) | 9152987821 |
| Vandrevala Foundation | 1860-2662-345 |
| NIMHANS Helpline | 080-46110007 |
| Snehi | 044-24640050 |
| International Association for Suicide Prevention | https://www.iasp.info/resources/Crisis_Centres/ |
| Emergency | 112 |

---

## References

1. Marwaha et al. (2013). Affective instability in bipolar disorder. *British Journal of Psychiatry*.
2. Guntuku et al. (2019). Detecting depression and mental illness on social media. *Current Opinion in Behavioral Sciences*.
3. Ekman, P. (1992). An argument for basic emotions. *Cognition & Emotion*.
4. DSM-5-TR (2022). Bipolar and Related Disorders. *APA*.
5. WHO mhGAP Intervention Guide (2016). Mental health in low- and middle-income countries.

---

## License

MIT License — Free for research, educational, and humanitarian use.

---

*bpdisdet is not a substitute for professional mental health care.*
