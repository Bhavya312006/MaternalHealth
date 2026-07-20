# AI-Based Multimodal Maternal Health Risk Prediction System
## Using MobileNetV2 + EfficientNet-B0 Fusion with CLAHE-Enhanced Fetal Ultrasound

---

## Project Structure

```
maternal_project/
├── src/
│   ├── utils_preprocess.py          ← CLAHE preprocessing pipeline (shared)
│   ├── prepare_image_dataset.py     ← Step 1: Organize + preprocess images
│   ├── train_fetal_model.py         ← Step 2: Train MobileNetV2 (scan classification)
│   ├── smart_label_anomaly.py       ← Step 3A: Label normal/abnormal from masks
│   ├── train_anomaly_model.py       ← Step 3B: Train EfficientNet-B0 (anomaly)
│   ├── extract_image_features.py   ← Step 4: Extract 8 image features
│   ├── fusion_model.py              ← Step 5: Train fusion model (18 features)
│   ├── update_app_thresholds.py     ← Step 6: Calibrate anomaly thresholds
│   └── app.py                       ← Step 7: Run Streamlit app
├── data/
│   ├── tabular/
│   │   └── maternal_cleaned.csv     ← Clinical dataset (1,204 patients)
│   ├── images/
│   │   ├── FETAL_PLANES_ZENODO/     ← Download from zenodo.org/records/3904280
│   │   └── Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images/
│   │                                ← Download from data.mendeley.com/datasets/4gcpm9dsc3/1
│   └── prepared/                    ← Auto-created by prepare_image_dataset.py
├── models/                          ← Auto-created, stores .keras + .pkl files
├── docs/
│   ├── maternal_health_report.docx  ← Full project report
│   └── model_comparison.docx        ← Model comparison tables
└── requirements.txt

```

---

## Datasets Required

### 1. Zenodo FETAL_PLANES_DB
- **URL:** https://zenodo.org/records/3904280
- **Download:** FETAL_PLANES_ZENODO.zip (2.1 GB)
- **Extract to:** `D:\maternal_project\data\images\FETAL_PLANES_ZENODO\`
- **Contents:** 12,400 PNG images + FETAL_PLANES_DB_data.xlsx

### 2. Mendeley Fetal Abdominal Segmentation
- **URL:** https://data.mendeley.com/datasets/4gcpm9dsc3/1
- **Download:** All files
- **Extract to:** `D:\maternal_project\data\images\Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images\`
- **Contents:** IMAGES/ folder (1,588 PNG) + ARRAY_FORMAT/ folder (.npy masks)

### 3. Tabular Clinical Dataset
- Already included: `data/tabular/maternal_cleaned.csv`
- 1,204 patient records, 119 features
- Source: ANC register + NHANES combined dataset

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run Order (Complete Pipeline)

```bash
cd D:\maternal_project\src

# Step 1: Organize and preprocess all images (CLAHE pipeline)
python prepare_image_dataset.py

# Step 2: Train scan type classifier (MobileNetV2, 6 classes)
python train_fetal_model.py
# Expected: ~86% accuracy, saves fetal_classification_model.keras

# Step 3A: Create proper normal/abnormal labels from Mendeley
python smart_label_anomaly.py

# Step 3B: Train anomaly detector (EfficientNet-B0)
python train_anomaly_model.py
# Expected: ~90% AUC, saves fetal_anomaly_model.keras + anomaly_threshold_stats.json

# Step 4: Extract 8 image features using both trained models
python extract_image_features.py
# Output: image_features_8.npy (shape: n_images x 8)

# Step 5: Train fusion model (10 tabular + 8 image = 18 features)
python fusion_model.py
# Expected: ~97-99% accuracy, saves fusion_maternal_model.pkl + fusion_scaler.pkl

# Step 6: Auto-calibrate anomaly thresholds in app.py
python update_app_thresholds.py

# Step 7: Launch the app
streamlit run app.py
```

---

## System Architecture

```
Clinical Inputs (10 features)          Ultrasound Image
       ↓                                      ↓
StandardScaler normalization         CLAHE Preprocessing Pipeline
       ↓                              (Median + CLAHE + Bilateral)
10-feature tabular vector                     ↓
                                   MobileNetV2 (Stage 1)
                                   → 6 scan-type probabilities
                                          +
                                   EfficientNet-B0 (Stage 2)
                                   → 2 anomaly probabilities
                                          ↓
                              8 image features
                                   ↓
       ←──────── concatenate ────────────────────────→
                    18-feature fusion vector
                          ↓
              Random Forest Classifier
                          ↓
         🟢 Low / 🟡 Moderate / 🔴 High Risk
```

---

## Top 10 Input Features (by importance)

| # | Feature | Importance |
|---|---|---|
| 1 | Weight (kg) | 22.22% |
| 2 | Diastolic BP - ANC | 13.50% |
| 3 | Tetanus Vaccine Doses | 11.37% |
| 4 | Systolic BP - ANC | 6.46% |
| 5 | BMI (auto-calculated) | 3.79% |
| 6 | HBsAg Test | 3.69% |
| 7 | Blood Sugar (mmol/L) | 3.68% |
| 8 | Age | 3.36% |
| 9 | Gestational Age (weeks) | 3.16% |
| 10 | Heart Rate (bpm) | 2.90% |

---

## Test Cases

### 🟢 LOW RISK
Age=22, Weight=52, Height=5.2ft, Gest=20wks, SysBP=110, DiaBP=55,
Sugar=6.4, HR=72, Tetanus=3, HBsAg=Negative, Anaemia=No, Jaundice=No
Image: any fetal_brain scan from prepared/classification/train/fetal_brain/

### 🟡 MODERATE RISK
Age=33, Weight=78, Height=5.1ft, Gest=34wks, SysBP=128, DiaBP=83,
Sugar=7.8, HR=88, Tetanus=2, HBsAg=Negative, Anaemia=Yes, Jaundice=No
Image: any maternal_cervix scan

### 🔴 HIGH RISK
Age=38, Weight=88, Height=5.0ft, Gest=37wks, SysBP=145, DiaBP=92,
Sugar=9.5, HR=102, Tetanus=1, HBsAg=Positive, Anaemia=Yes, Jaundice=Yes
Image: any scan from prepared/anomaly/abnormal/

---

## Models Saved After Training

```
D:\maternal_project\models\
├── fetal_classification_model.keras      ← MobileNetV2 scan classifier
├── fetal_classification_model_classes.json
├── fetal_anomaly_model.keras             ← EfficientNet-B0 anomaly detector
├── anomaly_threshold_stats.json          ← Calibrated thresholds
├── fusion_maternal_model.pkl             ← Random Forest fusion model
└── fusion_scaler.pkl                     ← StandardScaler for 18 features
```

---

## Key Results

| Component | Model | Accuracy |
|---|---|---|
| Tabular Risk | Random Forest | 94.61% |
| Scan Classification | MobileNetV2 | 86.15% |
| Anomaly Detection | EfficientNet-B0 | ~92% AUC |
| **Multimodal Fusion** | **Random Forest (18-feat)** | **97-99%** |

---

## Project Title
**"Multimodal Maternal Health Risk Prediction Using MobileNetV2–EfficientNet-B0 Fusion with CLAHE-Enhanced Fetal Ultrasound Analysis"**

---

## Team
- Bhavya (1BG23CS086)
- Dept. of CSE, BNMIT, Bengaluru

