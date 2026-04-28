"""
FILE 4: extract_image_features.py
REPLACES old extract_image_features.py (which used ResNet50)

WHAT IT DOES:
  Loads both trained models
  Extracts 8 features per image:
    [0-5] = scan-type probabilities (MobileNetV2)  6 values
    [6]   = normal probability      (EfficientNet)  1 value
    [7]   = abnormal probability    (EfficientNet)  1 value
  Saves: image_features_8.npy + image_labels.npy

OLD: ResNet50 → 2048 random useless features
NEW: 2 trained models → 8 medically meaningful features

RUN: python extract_image_features.py
"""
import os, sys, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_preprocess import preprocess_batch

CLF_MODEL    = r"C:\MATERNAL_FINAL\models\fetal_classification_model.h5"
ANO_MODEL    = r"C:\MATERNAL_FINAL\models\fetal_anomaly_model.keras"
CLASSES_JSON = r"C:\MATERNAL_FINAL\models\fetal_classification_model_classes.json"
IMAGE_DIR    = r"C:\MATERNAL_FINAL\data\prepared\classification\train"
OUT_FEATURES = r"C:\MATERNAL_FINAL\data\image_features_8.npy"
OUT_LABELS   = r"C:\MATERNAL_FINAL\data\image_labels.npy"

print("Loading models...")
clf_model = load_model(CLF_MODEL)
ano_model = load_model(ANO_MODEL)

with open(CLASSES_JSON) as f:
    class_indices = json.load(f)
idx_to_class = {v:k for k,v in class_indices.items()}
print(f"Scan classes: {list(class_indices.keys())}")

def extract_8_features(img_path):
    """
    Returns 8 features: [scan_probs x6, normal_prob, abnormal_prob]
    """
    img_batch  = preprocess_batch(img_path)                          # CLAHE pipeline
    scan_probs = clf_model.predict(img_batch, verbose=0)[0]          # (6,)
    ano_prob   = float(ano_model.predict(img_batch, verbose=0)[0][0])
    ano_probs  = np.array([1-ano_prob, ano_prob])                    # (2,)
    return np.concatenate([scan_probs, ano_probs])                   # (8,)

features_list, labels_list, skipped = [], [], 0

for class_name in sorted(os.listdir(IMAGE_DIR)):
    cls_path = os.path.join(IMAGE_DIR, class_name)
    if not os.path.isdir(cls_path): continue
    files = [f for f in os.listdir(cls_path) if f.lower().endswith((".png",".jpg",".jpeg"))]
    print(f"  {class_name}: {len(files)} images")
    for fname in files:
        try:
            feats = extract_8_features(os.path.join(cls_path, fname))
            features_list.append(feats); labels_list.append(class_name)
        except: skipped += 1

features_arr = np.array(features_list)
labels_arr   = np.array(labels_list)
np.save(OUT_FEATURES, features_arr)
np.save(OUT_LABELS, labels_arr)

print(f"\n✅ Features shape: {features_arr.shape}  (n_images x 8)")
print(f"   Skipped: {skipped}")
print(f"   [0-5] scan probs | [6] normal | [7] abnormal")
print("Next: python fusion_model.py")
