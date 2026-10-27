import streamlit as st
import numpy as np
import joblib
import json
import os
import sys
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ── Add src folder to path so utils_preprocess is found ─────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_preprocess import preprocess_batch

# ══════════════════════════════════════════════════════════
#  MODEL PATHS — full absolute paths
# ══════════════════════════════════════════════════════════
# MODEL PATHS — Streamlit Cloud compatible
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

FUSION_MODEL = MODELS_DIR / "fusion_maternal_model.pkl"
SCALER       = MODELS_DIR / "fusion_scaler.pkl"
CLF_MODEL    = MODELS_DIR / "fetal_classification_model.h5"
ANO_MODEL    = MODELS_DIR / "fetal_anomaly_model.keras"
CLASSES_JSON = MODELS_DIR / "fetal_classification_model_classes.json"

# ══════════════════════════════════════════════════════════
#  Load Models
# ══════════════════════════════════════════════════════════
@st.cache_resource
# Load Models
@st.cache_resource
def load_models():
    # Check that all required model files exist
    required_files = [
        FUSION_MODEL,
        SCALER,
        CLF_MODEL,
        ANO_MODEL,
        CLASSES_JSON
    ]

    missing_files = [
        str(file) for file in required_files
        if not file.exists()
    ]

    if missing_files:
        st.error("The following model files are missing:")
        for file in missing_files:
            st.write(file)
        st.stop()

    fusion = joblib.load(FUSION_MODEL)
    scaler = joblib.load(SCALER)
    clf = load_model(CLF_MODEL, compile=False)
    ano = load_model(ANO_MODEL, compile=False)

    with open(CLASSES_JSON, "r") as f:
        classes = json.load(f)

    return fusion, scaler, clf, ano, classes


fusion_model, scaler, clf_model, ano_model, class_indices = load_models()

idx_to_class = {v: k for k, v in class_indices.items()}

# ══════════════════════════════════════════════════════════
#  Page Config & CSS
# ══════════════════════════════════════════════════════════
st.set_page_config(page_title="Maternal Health AI", layout="wide", page_icon="🤰")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #fdf8f4;
        color: #2d1f1f;
    }
    .stApp { background: linear-gradient(135deg, #fdf8f4 0%, #fef3ec 100%); }
    #MainMenu, footer, header { visibility: hidden; }

    .hero {
        background: linear-gradient(135deg, #8b1a4a 0%, #c0395a 50%, #e8607a 100%);
        border-radius: 20px; padding: 2.5rem 2rem; margin-bottom: 2rem;
        position: relative; overflow: hidden;
    }
    .hero h1 { font-family: 'DM Serif Display', serif; font-size: 2.4rem; color: #fff; margin: 0 0 0.4rem 0; }
    .hero p  { color: rgba(255,255,255,0.85); font-size: 0.95rem; font-weight: 300; margin: 0; }

    .section-title {
        font-family: 'DM Serif Display', serif; font-size: 1.4rem; color: #8b1a4a;
        margin: 1.8rem 0 1rem 0; display: flex; align-items: center;
    }
    .section-title::after {
        content: ''; flex: 1; height: 1px;
        background: linear-gradient(to right, #e8607a44, transparent); margin-left: 10px;
    }

    .risk-low  { background: linear-gradient(135deg,#d4f4e2,#a8e6c3); border: 1.5px solid #52c77d; border-radius: 16px; padding: 1.5rem 2rem; text-align: center; }
    .risk-mod  { background: linear-gradient(135deg,#fff8e1,#fff3cd); border: 1.5px solid #f0a500; border-radius: 16px; padding: 1.5rem 2rem; text-align: center; }
    .risk-high { background: linear-gradient(135deg,#ffe4e9,#ffc8d0); border: 1.5px solid #e8607a; border-radius: 16px; padding: 1.5rem 2rem; text-align: center; }
    .risk-title { font-family: 'DM Serif Display', serif; font-size: 1.6rem; margin: 0 0 0.3rem 0; }
    .risk-sub   { font-size: 0.9rem; font-weight: 300; opacity: 0.8; margin: 0; }

    .pill-row { display: flex; flex-wrap: wrap; gap: 10px; margin: 0.5rem 0 1rem 0; }
    .pill { background: #fff; border: 1px solid rgba(139,26,74,0.15); border-radius: 50px;
            padding: 6px 16px; font-size: 0.82rem; color: #8b1a4a; font-weight: 500; }

    .flag     { background: #fff8f0; border-left: 4px solid #e8a020;
                border-radius: 0 10px 10px 0; padding: 10px 14px; margin: 6px 0; font-size: 0.88rem; color: #7a4a00; }
    .flag-red { background: #fff0f3; border-left: 4px solid #e8607a;
                border-radius: 0 10px 10px 0; padding: 10px 14px; margin: 6px 0; font-size: 0.88rem; color: #8b1a4a; }

    .report-block { background: #fff; border-radius: 16px; padding: 1.5rem;
                    box-shadow: 0 2px 20px rgba(139,26,74,0.06);
                    border: 1px solid rgba(139,26,74,0.08); margin-bottom: 1rem; }
    .report-block h4 { font-family: 'DM Serif Display', serif; color: #8b1a4a; font-size: 1.1rem; margin: 0 0 0.6rem 0; }
    .report-block p, .report-block li { font-size: 0.9rem; color: #4a3030; line-height: 1.7; margin: 0.2rem 0; }

    .scan-badge { display: inline-block; background: linear-gradient(135deg,#8b1a4a,#c0395a);
                  color: white; border-radius: 20px; padding: 4px 14px; font-size: 0.82rem; font-weight: 600; margin: 4px; }
    .anomaly-normal   { background: #edfaf3; border: 1.5px solid #52c77d; border-radius: 12px; padding: 12px 16px; }
    .anomaly-abnormal { background: #fff0f3; border: 1.5px solid #e8607a; border-radius: 12px; padding: 12px 16px; }

    .avoid-card  { background: #fff; border: 1.5px solid #ffc8d0; border-radius: 14px; padding: 14px; text-align: center; margin-bottom: 8px; }
    .avoid-card .emoji { font-size: 1.6rem; margin-bottom: 4px; }
    .avoid-card .name  { font-size: 0.82rem; font-weight: 600; color: #8b1a4a; margin-bottom: 2px; }
    .avoid-card .reason{ font-size: 0.73rem; color: #888; line-height: 1.4; }
    .reduce-card { background: #fff; border: 1.5px solid #ffe0a8; border-radius: 14px; padding: 14px; text-align: center; margin-bottom: 8px; }
    .reduce-card .emoji { font-size: 1.6rem; margin-bottom: 4px; }
    .reduce-card .name  { font-size: 0.82rem; font-weight: 600; color: #7a4a00; margin-bottom: 2px; }
    .reduce-card .reason{ font-size: 0.73rem; color: #888; line-height: 1.4; }

    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.83rem !important; font-weight: 700 !important;
        color: #8b1a4a !important; text-transform: uppercase !important;
    }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] > div > div {
        background: #ffffff !important;
        border: 1.5px solid rgba(139,26,74,0.35) !important;
        border-radius: 10px !important;
        color: #1a1a1a !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stNumberInput"] input:focus {
        border-color: #8b1a4a !important;
        box-shadow: 0 0 0 2px rgba(139,26,74,0.15) !important;
        color: #1a1a1a !important;
        outline: none !important;
    }
    div[data-testid="stSelectbox"] > div > div > div {
        color: #1a1a1a !important;
    }
    div[data-testid="stNumberInput"] button {
        background: #f5f0f4 !important;
        color: #8b1a4a !important;
        border-color: rgba(139,26,74,0.2) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #8b1a4a, #c0395a) !important;
        color: white !important; border: none !important; border-radius: 50px !important;
        font-weight: 600 !important; width: 100% !important;
        box-shadow: 0 4px 18px rgba(139,26,74,0.3) !important;
        font-size: 1rem !important; padding: 0.65rem 2rem !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 24px rgba(139,26,74,0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  Hero
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <h1>🤰 Maternal Health AI</h1>
    <p>Dual AI analysis — clinical data + ultrasound imaging —
    for accurate pregnancy risk assessment and complication detection.</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  Patient Inputs
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-title">👤 Patient Information</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    age    = st.number_input("👤 Age (years)",    min_value=15,   max_value=50,    value=26)
    weight = st.number_input("⚖️ Weight (kg)",    min_value=35.0, max_value=120.0, value=60.0)
    height = st.number_input("📏 Height (ft)",    min_value=4.0,  max_value=7.0,   value=5.2)

with c2:
    gest_week  = st.number_input("🗓️ Gestational Age (wks)", min_value=1,  max_value=42,  value=28)
    bp_sys_anc = st.number_input("💉 Systolic BP - ANC",     min_value=80, max_value=200, value=110)
    bp_dia_anc = st.number_input("💉 Diastolic BP - ANC",    min_value=50, max_value=140, value=70)

with c3:
    blood_sugar   = st.number_input("🩸 Blood Sugar (mmol/L)", min_value=3.0, max_value=20.0, value=7.0)
    heart_rate    = st.number_input("❤️ Heart Rate (bpm)",     min_value=50,  max_value=140,  value=74)
    tetanus_doses = st.number_input("💉 Tetanus Vaccine Doses", min_value=1,  max_value=3,    value=2)

with c4:
    hrsag_test = st.selectbox("🧬 HBsAg Test", ["Negative", "Positive"])
    anaemia    = st.selectbox("🩸 Anaemia",    [0, 1], format_func=lambda x: "Yes ✓" if x else "No ✗")
    jaundice   = st.selectbox("🟡 Jaundice",   [0, 1], format_func=lambda x: "Yes ✓" if x else "No ✗")

# Auto BMI
bmi = round(weight / ((height * 0.3048) ** 2), 1)
bmi_status = "✅ Normal" if bmi < 25 else "⚠️ Overweight" if bmi < 30 else "🔴 Obese"
st.markdown(f"""
<div style="background:#fff0f8;border:1px solid rgba(139,26,74,0.2);border-radius:10px;
     padding:8px 16px;margin:4px 0;font-size:0.85rem;color:#8b1a4a;font-weight:600;">
  📊 Auto-calculated BMI: <b>{bmi}</b> kg/m²  &nbsp;|&nbsp; {bmi_status}
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  Ultrasound Upload
# ══════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🖼️ Ultrasound Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Ultrasound Image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    file_kb = round(uploaded_file.size / 1024, 1)
    pc, ic  = st.columns([1, 2])
    with pc:
        st.image(uploaded_file, use_column_width=True)
    with ic:
        st.markdown(f"""
        <div style="background:#edfaf3;border:1.5px solid #52c77d;border-radius:14px;
             padding:1rem 1.4rem;margin-top:0.3rem;">
          <b style="color:#1a7a4a;font-size:1rem;">✅ Image Uploaded Successfully</b><br>
          <span style="font-size:0.85rem;color:#2d5040;line-height:1.8;">
            📁 <b>File:</b> {uploaded_file.name}<br>
            📦 <b>Size:</b> {file_kb} KB<br>
            🤖 <b>Stage 1:</b> Scan type classification (MobileNetV2)<br>
            🔬 <b>Stage 2:</b> Complication detection (EfficientNet-B0)
          </span>
        </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background:#fff8f0;border:1.5px dashed #e8a020;border-radius:14px;
         padding:0.9rem 1.3rem;display:flex;align-items:center;gap:12px;">
      <div style="font-size:1.5rem;">📂</div>
      <div style="font-size:0.85rem;color:#7a4a00;">
        <b>No image uploaded.</b> Upload an ultrasound to enable
        scan classification + complication detection.
        Without it, risk is based on clinical data only.
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  Generate Report
# ══════════════════════════════════════════════════════════
if st.button("✨ Generate AI Health Report"):

    # ── Tabular features (top 10, exact order matching fusion model training) ──
    hrsag_enc = 1 if hrsag_test == "Positive" else 0
    tabular   = np.array([[
        weight,        # weight_kg
        bp_dia_anc,    # diastolic_bp_anc
        tetanus_doses, # tetanus_vaccine
        bp_sys_anc,    # systolic_bp_anc
        bmi,           # bmi
        hrsag_enc,     # hrsag_test
        blood_sugar,   # blood_sugar
        age,           # age
        gest_week,     # gestational_age_week
        heart_rate     # heart_rate
    ]])

    # ── Process ultrasound ──────────────────────────────────
    image_uploaded = uploaded_file is not None
    scan_probs_6   = np.array([1/6]*6)
    anomaly_prob   = 0.0
    scan_type      = "Not provided"
    scan_conf      = 0.0
    top_class_idx  = 0

    if image_uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        try:
            img_batch     = preprocess_batch(tmp_path)
            scan_probs_6  = clf_model.predict(img_batch, verbose=0)[0]
            top_class_idx = int(np.argmax(scan_probs_6))
            scan_type     = idx_to_class[top_class_idx].replace("_"," ").title()
            scan_conf     = float(scan_probs_6[top_class_idx]) * 100
            anomaly_prob  = float(ano_model.predict(img_batch, verbose=0)[0][0])
        except Exception as e:
            st.warning(f"⚠️ Image processing issue: {e}. Using clinical data only.")
            image_uploaded = False
        finally:
            try: os.remove(tmp_path)
            except: pass

    anomaly_2 = np.array([1 - anomaly_prob, anomaly_prob])

    # ── Combine 10 tabular + 8 image = 18 features ─────────
    img_8      = np.concatenate([scan_probs_6, anomaly_2]).reshape(1, -1)
    X_combined = np.concatenate([tabular, img_8], axis=1)
    X_scaled   = scaler.transform(X_combined)

    # ── Predict ─────────────────────────────────────────────
    if hasattr(fusion_model, "predict_proba"):
        risk_prob = float(fusion_model.predict_proba(X_scaled)[0][1])
    else:
        risk_prob = float(fusion_model.predict(X_scaled)[0])

    # ── Boost if anomaly detected ────────────────────────────
    if image_uploaded and anomaly_prob > 0.5454:
        risk_prob = min(1.0, risk_prob + 0.15)

    # ── 3-level risk classification ──────────────────────────
    warning_count = sum([
        bp_sys_anc >= 130,
        bp_dia_anc >= 85,
        age >= 30,
        anaemia == 1,
        jaundice == 1,
        gest_week >= 36,
        bmi >= 30,
        blood_sugar >= 7.8,
        heart_rate >= 100,
        image_uploaded and anomaly_prob > 0.45
    ])

    if bp_sys_anc >= 140 or bp_dia_anc >= 90 or age >= 40 or (anaemia and jaundice):
        risk = 2
        risk_prob = max(risk_prob, 0.80)
    elif bp_sys_anc >= 130 or bp_dia_anc >= 85 or age >= 35 or warning_count >= 2:
        risk = 1
        risk_prob = max(0.45, min(risk_prob, 0.72))
    else:
        risk = 0
        risk_prob = min(risk_prob, 0.28)

    risk_pct = int(risk_prob * 100)

    # ══════════════════════════════════════════════════════
    #  PATIENT SNAPSHOT
    # ══════════════════════════════════════════════════════
    st.markdown('<div class="section-title">📋 Patient Snapshot</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="pill-row">
        <span class="pill">👤 Age: {age} yrs</span>
        <span class="pill">⚖️ Weight: {weight} kg</span>
        <span class="pill">📏 BMI: {bmi}</span>
        <span class="pill">🗓️ Week: {gest_week}</span>
        <span class="pill">💉 BP: {bp_sys_anc}/{bp_dia_anc} mmHg</span>
        <span class="pill">🩸 Sugar: {blood_sugar} mmol/L</span>
        <span class="pill">❤️ HR: {heart_rate} bpm</span>
        <span class="pill">🧬 HBsAg: {hrsag_test}</span>
        <span class="pill">🩸 Anaemia: {"Yes" if anaemia else "No"}</span>
        <span class="pill">🟡 Jaundice: {"Yes" if jaundice else "No"}</span>
    </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    #  RISK INDICATOR
    # ══════════════════════════════════════════════════════
    st.markdown('<div class="section-title">🎯 Pregnancy Risk Indicator</div>', unsafe_allow_html=True)

    if risk == 0:
        st.markdown(f"""<div class="risk-low">
            <p class="risk-title" style="color:#1a7a4a;">🟢 LOW RISK PREGNANCY</p>
            <p class="risk-sub" style="color:#2d6050;">AI Risk Score: ~{risk_pct}% &nbsp;·&nbsp; Routine check-ups sufficient</p>
        </div>""", unsafe_allow_html=True)
    elif risk == 1:
        st.markdown(f"""<div class="risk-mod">
            <p class="risk-title" style="color:#b36b00;">🟡 MODERATE RISK PREGNANCY</p>
            <p class="risk-sub" style="color:#7a5000;">AI Risk Score: ~{risk_pct}% &nbsp;·&nbsp; Precautions & closer monitoring advised</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="risk-high">
            <p class="risk-title" style="color:#8b1a4a;">🔴 HIGH RISK PREGNANCY</p>
            <p class="risk-sub" style="color:#7a2030;">AI Risk Score: ~{risk_pct}% &nbsp;·&nbsp; Immediate medical attention required</p>
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    #  ULTRASOUND ANALYSIS
    # ══════════════════════════════════════════════════════
    st.markdown('<div class="section-title">🔬 Ultrasound Image Analysis</div>', unsafe_allow_html=True)

    if image_uploaded:
        st.markdown(f"""
        <div class="report-block">
            <h4>🖼️ Stage 1 — Scan Type Classification (MobileNetV2)</h4>
            <p>Detected: <span class="scan-badge">{scan_type}</span>
               with <b>{round(scan_conf,1)}%</b> confidence</p>
            <p style="font-size:0.82rem;color:#666;margin-top:8px;">All class probabilities:</p>
        </div>""", unsafe_allow_html=True)

        prob_cols = st.columns(len(class_indices))
        for cls_name, cls_idx in sorted(class_indices.items(), key=lambda x: x[1]):
            pval  = float(scan_probs_6[cls_idx]) * 100
            color = "#8b1a4a" if cls_idx == top_class_idx else "#aaa"
            with prob_cols[cls_idx]:
                st.markdown(f"""
                <div style="text-align:center;padding:8px;background:#fdf8f4;
                     border-radius:10px;border:1px solid rgba(139,26,74,0.1);">
                  <div style="font-size:1.1rem;font-weight:700;color:{color};">{round(pval,1)}%</div>
                  <div style="font-size:0.68rem;color:#888;">{cls_name.replace('_',' ').title()}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Stage 2 — ADJUSTED THRESHOLDS: >0.45 = red, >0.25 = yellow, else green
        # Stage 2 — FIXED THRESHOLDS
        st.markdown("<br>", unsafe_allow_html=True)

        # Stage 2 — FIXED THRESHOLDS
        ap = round(anomaly_prob * 100, 1)

        if anomaly_prob > 0.45:
            st.markdown(f"""
            <div class="anomaly-abnormal">
              <b style="color:#8b1a4a;">🔴 Stage 2 — Complication Detected (EfficientNet-B0)</b><br>
              <span style="font-size:0.88rem;color:#7a2030;">
              Anomaly probability: <b>{ap}%</b> — Visual irregularities in this {scan_type} scan.
              Could indicate placental issues, fluid abnormalities, or growth concerns.
              Specialist review recommended immediately.
              </span>
            </div>""", unsafe_allow_html=True)

        elif anomaly_prob > 0.25:
            st.markdown(f"""
            <div style="background:#fff8e1;border:1.5px solid #f0a500;border-radius:12px;padding:12px 16px;">
              <b style="color:#b36b00;">🟡 Stage 2 — Borderline Pattern (EfficientNet-B0)</b><br>
              <span style="font-size:0.88rem;color:#7a5000;">
              Anomaly probability: <b>{ap}%</b> — Minor irregularities detected.
              Monitor at next scheduled scan.
              </span>
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="anomaly-normal">
              <b style="color:#1a7a4a;">✅ Stage 2 — No Complications Detected (EfficientNet-B0)</b><br>
              <span style="font-size:0.88rem;color:#2d6050;">
              Anomaly probability: <b>{ap}%</b> —
              {scan_type} scan appears within normal range for gestational week {gest_week}.
              </span>
            </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#f8f4ff;border:1.5px dashed #c4b0d8;border-radius:14px;
             padding:1.2rem 1.5rem;display:flex;align-items:center;gap:14px;">
          <div style="font-size:2rem;">🖼️</div>
          <div>
            <b style="color:#5a3a7a;">No Ultrasound Image Provided</b><br>
            <span style="font-size:0.83rem;color:#888;">
            Risk based on clinical data only.
            Upload an ultrasound to enable Stage 1 + Stage 2 analysis.
            </span>
          </div>
        </div>""", unsafe_allow_html=True)
    # ══════════════════════════════════════════════════════
    #  RISK FACTOR FLAGS
    # ══════════════════════════════════════════════════════
    flags = []
    if age >= 35:
        flags.append(("⚠️","flag",f"<b>Advanced maternal age ({age} yrs)</b> — higher complication risk"))
    if bp_sys_anc >= 140 or bp_dia_anc >= 90:
        flags.append(("⚠️","flag-red","<b>Blood pressure critically elevated</b> — preeclampsia risk"))
    elif bp_sys_anc >= 130 or bp_dia_anc >= 85:
        flags.append(("⚠️","flag","<b>Blood pressure borderline high</b> — monitor closely"))
    if blood_sugar >= 7.8:
        flags.append(("🩸","flag-red",f"<b>Blood sugar elevated ({blood_sugar} mmol/L)</b> — gestational diabetes risk"))
    if heart_rate >= 100:
        flags.append(("❤️","flag",f"<b>Heart rate elevated ({heart_rate} bpm)</b> — cardiac stress"))
    if bmi >= 30:
        flags.append(("⚖️","flag",f"<b>BMI {bmi} — Obese</b> — increases delivery complications"))
    elif bmi >= 25:
        flags.append(("⚖️","flag",f"<b>BMI {bmi} — Overweight</b> — monitor weight gain"))
    if hrsag_test == "Positive":
        flags.append(("🧬","flag-red","<b>HBsAg Positive</b> — Hepatitis B detected, specialist consult needed"))
    if anaemia:
        flags.append(("🩸","flag","<b>Anaemia detected</b> — iron & folate supplement needed"))
    if jaundice:
        flags.append(("🟡","flag","<b>Jaundice detected</b> — liver function monitoring advised"))
    if image_uploaded and anomaly_prob > 0.45:
        flags.append(("🔬","flag-red",
            f"<b>Ultrasound anomaly ({round(anomaly_prob*100)}%)</b> — complication in {scan_type} scan"))

    if flags:
        st.markdown('<div class="section-title">⚠️ Risk Factor Flags</div>', unsafe_allow_html=True)
        for ico, cls, msg in flags:
            st.markdown(f'<div class="{cls}">{ico} {msg}</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    #  AI HEALTH REPORT
    # ══════════════════════════════════════════════════════
    st.markdown('<div class="section-title">📄 AI Maternal Health Report</div>', unsafe_allow_html=True)

    if risk == 0:
        st.markdown("""
        <div class="report-block"><h4>Current Condition</h4>
        <p>Pregnancy is progressing within a healthy range. Vital signs are stable and no major risk factors detected.</p></div>
        <div class="report-block"><h4>✅ Health Status</h4><ul>
            <li>Maternal vital signs appear stable</li>
            <li>No major risk factors detected at this time</li>
            <li>Pregnancy development appears on track</li>
            <li>Routine monthly check-ups are sufficient</li>
        </ul></div>""", unsafe_allow_html=True)

    elif risk == 1:
        st.markdown("""
        <div class="report-block"><h4>Current Condition</h4>
        <p>Borderline indicators detected. Not immediately dangerous but require closer attention and lifestyle adjustments.</p></div>
        <div class="report-block"><h4>⚡ Possible Concerns</h4><ul>
            <li>Borderline BP or age-related risk factors present</li>
            <li>Risk of gestational hypertension if unmonitored</li>
            <li>Nutritional deficiencies may develop</li>
            <li>Stress or overexertion could worsen condition</li>
        </ul></div>
        <div class="report-block"><h4>🛡️ Recommended Precautions</h4><ul>
            <li>Doctor visit every 2 weeks instead of monthly</li>
            <li>Monitor blood pressure at home every 2–3 days</li>
            <li>Reduce salt, sugar, and processed food intake</li>
            <li>Light exercise only — avoid heavy lifting</li>
            <li>Ensure 8 hrs sleep and manage stress actively</li>
            <li>Report any swelling, headaches or vision changes immediately</li>
        </ul></div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="report-block"><h4>Current Condition</h4>
        <p>Significant indicators detected. Immediate medical attention is critical for both mother and baby.</p></div>
        <div class="report-block"><h4>⚡ Possible Effects</h4><ul>
            <li>Risk of preeclampsia or gestational hypertension</li>
            <li>Possible complications for fetal development</li>
            <li>Higher chance of preterm labour if untreated</li>
            <li>Maternal organ stress — kidney, liver function at risk</li>
        </ul></div>
        <div class="report-block"><h4>🚨 Urgent Precautions</h4><ul>
            <li>Consult OB-GYN immediately — do not delay</li>
            <li>Weekly hospital check-ups are strongly advised</li>
            <li>Monitor blood pressure twice daily</li>
            <li>Strict bed rest if BP remains elevated</li>
            <li>Avoid all salt, fried food, caffeine and stress</li>
            <li>Emergency contact ready at all times</li>
        </ul></div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    #  DIET — compact donut
    # ══════════════════════════════════════════════════════
    st.markdown('<div class="section-title">🥗 Recommended Pregnancy Diet</div>', unsafe_allow_html=True)
    diet_col, legend_col = st.columns([1, 1])

    if risk == 2:   sizes = [30, 20, 28, 12, 10]
    elif risk == 1: sizes = [27, 25, 26, 12, 10]
    else:           sizes = [25, 30, 20, 15, 10]

    labels = ["Proteins","Carbohydrates","Fruits & Veg","Dairy","Healthy Fats"]
    colors = ["#c0395a","#e8607a","#f5a0b0","#ffc8d0","#ffe0e8"]

    with diet_col:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        fig.patch.set_facecolor("none")
        _, _, autotexts = ax.pie(sizes, labels=None, autopct='%1.0f%%',
            startangle=90, colors=colors, pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2))
        for at in autotexts:
            at.set_fontsize(8); at.set_color("#2d1f1f"); at.set_fontweight("600")
        ax.axis("equal"); ax.set_facecolor("none")
        st.pyplot(fig, use_container_width=False)

    with legend_col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        for label, color, size in zip(labels, colors, sizes):
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                <div style="width:14px;height:14px;border-radius:4px;background:{color};flex-shrink:0;"></div>
                <span style="font-size:0.88rem;font-weight:600;color:#3d1f2f;">{label}</span>
                <span style="font-size:0.78rem;color:#888;">{size}%</span>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    #  FOODS TO AVOID
    # ══════════════════════════════════════════════════════
    st.markdown('<div class="section-title">🚫 Foods to Avoid Completely</div>', unsafe_allow_html=True)
    avoid_foods = [
        ("🍣","Raw Seafood","Listeria & mercury"),
        ("🥩","Undercooked Meat","Toxoplasmosis"),
        ("🍷","Alcohol","Fetal alcohol syndrome"),
        ("🚬","Tobacco / Pan","Low birth weight"),
        ("🧀","Unpasteurised Cheese","Listeria risk"),
        ("🥚","Raw Eggs","Salmonella infection"),
        ("🫙","Excess Vitamin A","Birth defect risk"),
        ("☕","High Caffeine >200mg","Miscarriage risk"),
    ]
    if risk >= 1:
        avoid_foods += [("🧂","Excess Salt","Worsens BP"),("🍟","Fried Foods","Increases BP")]
    if risk == 2:
        avoid_foods += [("🥛","Full Fat Dairy","Raises cholesterol"),("🍔","Fast Food","High sodium")]

    cols = st.columns(4)
    for i, (emoji, name, reason) in enumerate(avoid_foods):
        with cols[i % 4]:
            st.markdown(
                f'<div class="avoid-card"><div class="emoji">{emoji}</div>'
                f'<div class="name">{name}</div><div class="reason">{reason}</div></div>',
                unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    #  FOODS TO REDUCE
    # ══════════════════════════════════════════════════════
    st.markdown('<div class="section-title">⚠️ Foods to Reduce / Limit</div>', unsafe_allow_html=True)
    reduce_foods = [
        ("🍬","Sugar & Sweets","Diabetes risk"),
        ("🍕","Processed Foods","High sodium"),
        ("🥤","Sugary Drinks","Empty calories"),
        ("🌶️","Spicy Foods","Heartburn"),
        ("🧃","Packaged Juices","Hidden sugars"),
        ("🍫","Excess Chocolate","Caffeine"),
    ]
    if anaemia:
        reduce_foods += [("☕","Tea with meals","Blocks iron")]
    if bp_sys_anc >= 130 or risk >= 1:
        reduce_foods += [("🧂","Salt / Soy Sauce","Raises BP")]
    if blood_sugar >= 7.0:
        reduce_foods += [("🍚","White Rice / Maida","Spikes blood sugar")]

    cols2 = st.columns(4)
    for i, (emoji, name, reason) in enumerate(reduce_foods):
        with cols2[i % 4]:
            st.markdown(
                f'<div class="reduce-card"><div class="emoji">{emoji}</div>'
                f'<div class="name">{name}</div><div class="reason">{reason}</div></div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#fff;border-radius:14px;padding:1rem 1.5rem;
         border:1px solid rgba(139,26,74,0.1);font-size:0.8rem;color:#888;text-align:center;">
        ⚕️ AI-generated report for informational purposes only.
        Always consult a qualified healthcare professional for medical decisions.
    </div>""", unsafe_allow_html=True)
