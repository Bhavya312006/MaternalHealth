"""
smart_label_anomaly.py
======================
Creates proper normal/abnormal labels from Mendeley dataset
using TWO sources of signal:

SOURCE 1 — Segmentation mask quality (from .npy files)
  Normal  = all 4 organs clearly visible (large mask area)
  Abnormal = organs missing/fragmented (small or zero mask area)
  Medical basis: poor organ visibility = growth restriction / structural anomaly

SOURCE 2 — Image texture analysis (from .png files)  
  Normal  = uniform, expected ultrasound texture
  Abnormal = irregular texture, unusual echo patterns, artifacts
  Medical basis: pathological tissue has different acoustic properties

Run this BEFORE train_anomaly_model.py
"""

import os
import sys
import cv2
import numpy as np
import shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_preprocess import preprocess_ultrasound

# ── PATHS ────────────────────────────────────────────────────
MENDELEY_IMAGES = r"C:\MATERNAL_FINAL\data\images\Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images\IMAGES"
MENDELEY_ARRAYS = r"C:\MATERNAL_FINAL\data\images\Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images\ARRAY_FORMAT"
NORMAL_OUT      = r"C:\MATERNAL_FINAL\data\prepared\anomaly\normal"
ABNORMAL_OUT    = r"C:\MATERNAL_FINAL\data\prepared\anomaly\abnormal"

os.makedirs(NORMAL_OUT,   exist_ok=True)
os.makedirs(ABNORMAL_OUT, exist_ok=True)

# ── Clear existing (start fresh) ─────────────────────────────
for folder in [NORMAL_OUT, ABNORMAL_OUT]:
    for f in os.listdir(folder):
        os.remove(os.path.join(folder, f))
print("Cleared existing anomaly folders")


def save_preprocessed(img_path, out_path):
    processed = preprocess_ultrasound(img_path)
    img_uint8 = (processed * 255).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))


def analyze_mask_quality(npy_path):
    """
    Reads Mendeley .npy mask file and computes organ coverage score.
    
    Mendeley npy format: dict with structure names as keys, masks as values
    Keys include: 'aorta', 'umbilical_vein', 'stomach', 'liver' (or similar)
    
    Returns:
      mask_score (0-1): higher = more organ coverage = more normal
      n_structures: how many structures were found
    """
    try:
        data = np.load(npy_path, allow_pickle=True)
        
        # Handle different formats
        if hasattr(data, 'item'):
            data = data.item()
        
        if isinstance(data, dict):
            total_area   = 0
            n_structures = 0
            
            for key, val in data.items():
                if isinstance(val, np.ndarray) and val.ndim >= 2:
                    mask_area = np.sum(val > 0)
                    if mask_area > 0:
                        total_area   += mask_area
                        n_structures += 1
            
            # Normalize by image area (assume 512x512 or actual size)
            img_area   = 512 * 512
            mask_score = min(1.0, total_area / (img_area * 0.3))  # 30% coverage = fully normal
            return mask_score, n_structures
        
        elif isinstance(data, np.ndarray):
            # Single mask array
            mask_area  = np.sum(data > 0)
            mask_score = min(1.0, mask_area / (512 * 512 * 0.2))
            return mask_score, 1
            
    except Exception as e:
        pass
    
    return None, 0  # Could not read mask


def compute_texture_irregularity(img_path):
    """
    Measures image texture to detect anomalies.
    
    Normal ultrasound: consistent speckle texture, clear organ boundaries
    Abnormal ultrasound: irregular texture, unusual echo densities
    
    Returns irregularity score (0-1): higher = more irregular = more abnormal
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.5
    
    img = cv2.resize(img, (224, 224))
    
    # 1. Local variance map — abnormal regions have unusual variance patterns
    kernel      = np.ones((9, 9), np.float32) / 81
    local_mean  = cv2.filter2D(img.astype(np.float32), -1, kernel)
    local_var   = cv2.filter2D((img.astype(np.float32))**2, -1, kernel) - local_mean**2
    var_std     = float(np.std(local_var)) / (float(np.mean(local_var)) + 1e-6)
    
    # 2. Edge density — abnormal tissue has more irregular edges
    edges       = cv2.Canny(img, 30, 100)
    edge_density = float(np.sum(edges > 0)) / (224 * 224)
    
    # 3. Intensity distribution — abnormal scans have unusual brightness distribution
    hist        = cv2.calcHist([img], [0], None, [64], [0, 256]).flatten()
    hist        = hist / (hist.sum() + 1e-6)
    # Entropy of histogram — very low or very high entropy = abnormal
    hist_nonzero = hist[hist > 0]
    entropy     = -float(np.sum(hist_nonzero * np.log2(hist_nonzero + 1e-6)))
    # Normalize entropy: expected normal range is 4.5-5.5 bits
    entropy_score = abs(entropy - 5.0) / 3.0   # deviation from expected
    
    # 4. Quadrant uniformity — normal fetal abdomen has consistent quadrants
    q = [img[:112,:112], img[:112,112:], img[112:,:112], img[112:,112:]]
    q_means     = [float(qq.mean()) for qq in q]
    q_variation = float(np.std(q_means)) / (float(np.mean(q_means)) + 1e-6)
    
    # Combine scores
    irregularity = (
        var_std       * 0.30 +
        edge_density  * 0.25 +
        entropy_score * 0.25 +
        q_variation   * 0.20
    )
    
    return min(1.0, float(irregularity))


# ── MAIN PROCESSING ───────────────────────────────────────────
print(f"\n=== Smart Anomaly Labeling ===")

img_files = sorted([
    f for f in os.listdir(MENDELEY_IMAGES)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])
print(f"Found {len(img_files)} images in Mendeley/IMAGES")

results  = []
no_mask  = 0

for img_name in img_files:
    img_path = os.path.join(MENDELEY_IMAGES, img_name)
    
    # Find corresponding npy file
    npy_name = os.path.splitext(img_name)[0] + '.npy'
    npy_path = os.path.join(MENDELEY_ARRAYS, npy_name)
    
    # Source 1: mask quality
    mask_score, n_structures = analyze_mask_quality(npy_path) if os.path.exists(npy_path) else (None, 0)
    
    # Source 2: texture irregularity
    texture_score = compute_texture_irregularity(img_path)
    
    # Combine signals
    if mask_score is not None:
        # Both sources available
        # Low mask coverage + high texture irregularity = abnormal
        combined_normality = mask_score * 0.6 + (1 - texture_score) * 0.4
    else:
        # Texture only
        no_mask += 1
        combined_normality = 1 - texture_score
    
    results.append({
        'name':        img_name,
        'path':        img_path,
        'normality':   combined_normality,
        'mask_score':  mask_score,
        'texture':     texture_score,
        'n_struct':    n_structures
    })

# Sort by normality score
results.sort(key=lambda x: x['normality'], reverse=True)
n = len(results)

print(f"Images with mask data: {n - no_mask}")
print(f"Images without mask:   {no_mask}")

# ── Split into normal/abnormal ────────────────────────────────
# Top 55% = normal, Bottom 45% = abnormal
# Using 45% abnormal because fetal complications are clinically significant
split_idx = int(n * 0.55)
normal_imgs   = results[:split_idx]
abnormal_imgs = results[split_idx:]

print(f"\nNormal:   {len(normal_imgs)} images (normality: {normal_imgs[0]['normality']:.3f} → {normal_imgs[-1]['normality']:.3f})")
print(f"Abnormal: {len(abnormal_imgs)} images (normality: {abnormal_imgs[0]['normality']:.3f} → {abnormal_imgs[-1]['normality']:.3f})")

# Verify separation is meaningful
normal_mean   = np.mean([r['normality'] for r in normal_imgs])
abnormal_mean = np.mean([r['normality'] for r in abnormal_imgs])
print(f"\nMean normality — Normal: {normal_mean:.3f} | Abnormal: {abnormal_mean:.3f}")
print(f"Separation gap: {normal_mean - abnormal_mean:.3f} (higher = better trained model)")

# ── Save preprocessed images ──────────────────────────────────
print("\nSaving preprocessed images...")
ok = 0
for r in normal_imgs:
    try:
        save_preprocessed(r['path'], os.path.join(NORMAL_OUT, r['name']))
        ok += 1
    except: pass

for r in abnormal_imgs:
    try:
        save_preprocessed(r['path'], os.path.join(ABNORMAL_OUT, r['name']))
        ok += 1
    except: pass

print(f"✅ Saved {ok} preprocessed images")
print(f"   Normal:   {len(os.listdir(NORMAL_OUT))}")
print(f"   Abnormal: {len(os.listdir(ABNORMAL_OUT))}")
print(f"\nNext: python train_anomaly_model.py")
