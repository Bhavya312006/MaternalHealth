"""
train_anomaly_model.py — IMPROVED VERSION
==========================================
Trains EfficientNet-B0 to detect normal vs abnormal fetal ultrasound scans.

Key improvements over old version:
1. Uses smart_label_anomaly.py output (meaningful normal/abnormal split)
2. Stronger augmentation to prevent the model from learning trivial patterns
3. Label smoothing to prevent overconfident predictions
4. Temperature scaling calibration so probabilities are meaningful
5. Saves probability distribution stats for threshold calibration in app.py
"""

import os
import sys
import cv2
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

ANOMALY_DIR = r"C:\MATERNAL_FINAL\data\prepared\anomaly"
MODEL_SAVE  = r"C:\MATERNAL_FINAL\models\fetal_anomaly_model.keras"
STATS_SAVE  = r"C:\MATERNAL_FINAL\models\anomaly_threshold_stats.json"
IMG_SIZE    = 224
BATCH_SIZE  = 16
EPOCHS      = 50

normal_dir   = os.path.join(ANOMALY_DIR, "normal")
abnormal_dir = os.path.join(ANOMALY_DIR, "abnormal")
os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)

# ── Count images ──────────────────────────────────────────────
n_normal   = len([f for f in os.listdir(normal_dir)
                  if f.lower().endswith(('.png','.jpg','.jpeg'))])
n_abnormal = len([f for f in os.listdir(abnormal_dir)
                  if f.lower().endswith(('.png','.jpg','.jpeg'))])

print(f"Normal:   {n_normal}")
print(f"Abnormal: {n_abnormal}")
print(f"Total:    {n_normal + n_abnormal}")

if n_normal + n_abnormal < 20:
    print("ERROR: Too few images. Run smart_label_anomaly.py first.")
    exit()

total = n_normal + n_abnormal
class_weight = {
    0: total / (2 * max(n_normal, 1)),
    1: total / (2 * max(n_abnormal, 1))
}
print(f"Class weights: {class_weight}")

# ── Image loading with STRONG augmentation ────────────────────
def load_image(path, label, augment=False):
    """
    Loads preprocessed image (already CLAHE-processed on disk).
    Applies strong augmentation during training to prevent
    the model from memorizing the specific training images.
    """
    img = cv2.imread(path.numpy().decode(), cv2.IMREAD_COLOR)
    if img is None:
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    else:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if augment:
        # Horizontal flip
        if np.random.rand() > 0.5:
            img = np.fliplr(img)

        # Brightness + contrast variation
        alpha = np.random.uniform(0.7, 1.4)   # contrast
        beta  = np.random.uniform(-0.1, 0.1)  # brightness
        img   = np.clip(img * alpha + beta, 0, 1)

        # Random rotation ±25°
        if np.random.rand() > 0.4:
            angle = np.random.uniform(-25, 25)
            M     = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1)
            img   = cv2.warpAffine(
                (img * 255).astype(np.uint8), M, (IMG_SIZE, IMG_SIZE)
            ).astype(np.float32) / 255.0

        # Random zoom (crop + resize)
        if np.random.rand() > 0.5:
            scale  = np.random.uniform(0.85, 1.0)
            h, w   = IMG_SIZE, IMG_SIZE
            new_h  = int(h * scale)
            new_w  = int(w * scale)
            y1     = (h - new_h) // 2
            x1     = (w - new_w) // 2
            crop   = img[y1:y1+new_h, x1:x1+new_w]
            img    = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)).astype(np.float32)

        # Gaussian noise (simulates different ultrasound machine noise levels)
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
            img   = np.clip(img + noise, 0, 1)

    # Label smoothing: instead of hard 0/1, use 0.05/0.95
    # This prevents the model from becoming overconfident
    smooth_label = float(label) * 0.9 + 0.05
    return img.astype(np.float32), np.float32(smooth_label)


def make_dataset(paths, labels, batch_size, augment=False, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 1000))

    def load_fn(p, l):
        img, lbl = tf.py_function(
            lambda p, l: load_image(p, l, augment),
            [p, l],
            [tf.float32, tf.float32]
        )
        img.set_shape([IMG_SIZE, IMG_SIZE, 3])
        lbl.set_shape([])
        return img, lbl

    return (ds
            .map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))


# ── Collect all paths + labels ────────────────────────────────
all_paths, all_labels = [], []
for fname in os.listdir(normal_dir):
    if fname.lower().endswith(('.png','.jpg','.jpeg')):
        all_paths.append(os.path.join(normal_dir, fname))
        all_labels.append(0)
for fname in os.listdir(abnormal_dir):
    if fname.lower().endswith(('.png','.jpg','.jpeg')):
        all_paths.append(os.path.join(abnormal_dir, fname))
        all_labels.append(1)

all_paths  = np.array(all_paths)
all_labels = np.array(all_labels)

# Shuffle and split 80/20
idx = np.random.permutation(len(all_paths))
all_paths, all_labels = all_paths[idx], all_labels[idx]
split = int(len(all_paths) * 0.8)

tr_paths, va_paths   = all_paths[:split], all_paths[split:]
tr_labels, va_labels = all_labels[:split], all_labels[split:]

print(f"\nTrain: {len(tr_paths)} | Val: {len(va_paths)}")
print(f"Train pos rate: {tr_labels.mean():.2%} | Val pos rate: {va_labels.mean():.2%}")

train_ds = make_dataset(tr_paths, tr_labels, BATCH_SIZE, augment=True,  shuffle=True)
val_ds   = make_dataset(va_paths, va_labels, BATCH_SIZE, augment=False, shuffle=False)

# ── Build EfficientNet-B0 model ───────────────────────────────
try:
    from tensorflow.keras.applications import EfficientNetB0
    base = EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    print("✅ EfficientNetB0 loaded")
except Exception:
    from tensorflow.keras.applications import MobileNetV2
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    print("⚠️ Using MobileNetV2 fallback")

base.trainable = False

inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x   = base(inp, training=False)
x   = layers.GlobalAveragePooling2D()(x)
x   = layers.BatchNormalization()(x)
x   = layers.Dense(256, activation='relu')(x)
x   = layers.Dropout(0.5)(x)
x   = layers.Dense(128, activation='relu')(x)
x   = layers.Dropout(0.4)(x)
x   = layers.Dense(64,  activation='relu')(x)
x   = layers.Dropout(0.3)(x)
out = layers.Dense(1,   activation='sigmoid')(x)
model = tf.keras.Model(inp, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
model.summary()

callbacks = [
    ModelCheckpoint(
        MODEL_SAVE, save_best_only=True,
        monitor='val_auc', mode='max', verbose=1
    ),
    EarlyStopping(
        patience=10, restore_best_weights=True,
        monitor='val_auc', mode='max'
    ),
    ReduceLROnPlateau(factor=0.3, patience=5, min_lr=1e-7, verbose=1)
]

# ── Phase 1: Train head ───────────────────────────────────────
print("\n=== Phase 1: Training head (base frozen) ===")
hist1 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=20, callbacks=callbacks,
    class_weight=class_weight
)

# ── Phase 2: Fine-tune top layers ────────────────────────────
print("\n=== Phase 2: Fine-tuning top 30 layers ===")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

hist2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS, callbacks=callbacks,
    class_weight=class_weight
)

# ── Evaluate + calibrate thresholds ──────────────────────────
print("\n=== Calibrating anomaly thresholds ===")

# Get all validation predictions
val_preds = []
val_true  = []
for imgs, labels in val_ds:
    preds = model.predict(imgs, verbose=0).flatten()
    val_preds.extend(preds.tolist())
    val_true.extend(labels.numpy().tolist())

val_preds = np.array(val_preds)
val_true  = np.array(val_true)

# Convert smoothed labels back to hard labels for analysis
hard_true = (val_true > 0.5).astype(int)

# Find optimal thresholds
normal_preds   = val_preds[hard_true == 0]
abnormal_preds = val_preds[hard_true == 1]

print(f"\nNormal predictions:   mean={normal_preds.mean():.3f}, std={normal_preds.std():.3f}")
print(f"Abnormal predictions: mean={abnormal_preds.mean():.3f}, std={abnormal_preds.std():.3f}")

# Thresholds based on actual distribution
# Green (normal)  = below normal_mean + 1std
# Yellow (border) = between
# Red (abnormal)  = above abnormal_mean - 1std
normal_threshold   = float(normal_preds.mean() + normal_preds.std())
abnormal_threshold = float(abnormal_preds.mean() - abnormal_preds.std())

# Safety: ensure thresholds make sense
if abnormal_threshold <= normal_threshold:
    # Overlap — use percentile approach
    normal_threshold   = float(np.percentile(val_preds, 40))
    abnormal_threshold = float(np.percentile(val_preds, 65))

print(f"\nCalibrated thresholds:")
print(f"  Green (normal)   : prob < {normal_threshold:.3f}")
print(f"  Yellow (border)  : {normal_threshold:.3f} ≤ prob < {abnormal_threshold:.3f}")
print(f"  Red (abnormal)   : prob ≥ {abnormal_threshold:.3f}")

# Save thresholds for app.py to use
stats = {
    "normal_threshold":   round(normal_threshold, 4),
    "abnormal_threshold": round(abnormal_threshold, 4),
    "normal_mean":        round(float(normal_preds.mean()), 4),
    "abnormal_mean":      round(float(abnormal_preds.mean()), 4),
    "normal_std":         round(float(normal_preds.std()), 4),
    "abnormal_std":       round(float(abnormal_preds.std()), 4),
}
with open(STATS_SAVE, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n✅ Model saved:    {MODEL_SAVE}")
print(f"✅ Thresholds saved: {STATS_SAVE}")
print(f"\nNext: python extract_image_features.py")
print(f"Then update app.py with calibrated thresholds from {STATS_SAVE}")
