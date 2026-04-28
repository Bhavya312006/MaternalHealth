"""
update_app_thresholds.py
========================
Run this AFTER train_anomaly_model.py to automatically update
the anomaly thresholds in app.py based on your trained model's
actual probability distribution.

Run: python update_app_thresholds.py
"""

import json
import os
import re

STATS_FILE = r"C:\MATERNAL_FINAL\models\anomaly_threshold_stats.json"
APP_FILE   = r"C:\MATERNAL_FINAL\src\app.py"

# Check stats file
if not os.path.exists(STATS_FILE):
    print(f"ERROR: {STATS_FILE} not found.")
    print("Run train_anomaly_model.py first.")
    exit()

# Load stats
with open(STATS_FILE, "r", encoding="utf-8") as f:
    stats = json.load(f)

normal_thr   = stats['normal_threshold']
abnormal_thr = stats['abnormal_threshold']

print(f"Loading calibrated thresholds:")
print(f"  Normal    threshold: {normal_thr}")
print(f"  Abnormal  threshold: {abnormal_thr}")
print(f"  (Normal dist  mean: {stats['normal_mean']} ± {stats['normal_std']})")
print(f"  (Abnormal dist mean: {stats['abnormal_mean']} ± {stats['abnormal_std']})")

# Read app.py
if not os.path.exists(APP_FILE):
    print(f"ERROR: {APP_FILE} not found.")
    exit()

with open(APP_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# -------------------------
# Replace thresholds safely
# -------------------------

# Pattern 1: anomaly_prob > X (risk boost)
content = re.sub(
    r'if image_uploaded and anomaly_prob > [\d.]+:\n(\s+risk_prob)',
    f'if image_uploaded and anomaly_prob > {abnormal_thr}:\n\\1',
    content
)

# Pattern 2: warning_count anomaly check
content = re.sub(
    r'image_uploaded and anomaly_prob > [\d.]+(.*?#.*?$)',
    f'image_uploaded and anomaly_prob > {abnormal_thr}\\1',
    content,
    flags=re.MULTILINE
)

# Pattern 3: Stage display thresholds
content = re.sub(
    r'if anomaly_prob > [\d.]+:\n(\s+st\.markdown.*?🔴 Stage 2)',
    f'if anomaly_prob > {abnormal_thr}:\n\\1',
    content,
    flags=re.DOTALL
)

content = re.sub(
    r'elif anomaly_prob > [\d.]+:\n(\s+st\.markdown.*?🟡 Stage 2)',
    f'elif anomaly_prob > {normal_thr}:\n\\1',
    content,
    flags=re.DOTALL
)

# Pattern 4: ultrasound flags check
content = re.sub(
    r'image_uploaded and anomaly_prob > [\d.]+(\s*#.*ultrasound)',
    f'image_uploaded and anomaly_prob > {abnormal_thr}\\1',
    content
)

# -------------------------
# WRITE BACK (FIXED)
# -------------------------
with open(APP_FILE, "w", encoding="utf-8") as f:
    f.write(content)

print(f"\n✅ app.py updated with calibrated thresholds!")
print(f"   🔴 Red   (complication): anomaly_prob > {abnormal_thr}")
print(f"   🟡 Yellow (borderline):  anomaly_prob > {normal_thr}")
print(f"   ✅ Green  (normal):      anomaly_prob ≤ {normal_thr}")

print(f"\nRun: streamlit run app.py")