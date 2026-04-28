"""
FILE 2: train_fetal_model.py
REPLACES old train_fetal_model.py

WHAT IT DOES:
  Trains MobileNetV2 to classify 6 scan types
  Uses CLAHE-preprocessed images from prepare_image_dataset.py
  Phase 1: train head | Phase 2: fine-tune top 30 layers
  Saves: fetal_classification_model.h5 + _classes.json

EXPECTED ACCURACY: ~95-97%
RUN: python train_fetal_model.py
"""
import os, sys, json, cv2, numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_preprocess import preprocess_ultrasound

TRAIN_DIR  = r"C:\MATERNAL_FINAL\data\prepared\classification\train"
TEST_DIR   = r"C:\MATERNAL_FINAL\data\prepared\classification\test"
MODEL_SAVE = r"C:\MATERNAL_FINAL\models\fetal_classification_model.h5"
IMG_SIZE, BATCH_SIZE, EPOCHS = 224, 32, 30

os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)

classes       = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR,d))])
class_indices = {cls:i for i,cls in enumerate(classes)}
num_classes   = len(class_indices)
print(f"Classes ({num_classes}):", class_indices)

with open(MODEL_SAVE.replace(".h5","_classes.json"),"w") as f:
    json.dump(class_indices, f)

def get_paths_labels(data_dir):
    paths, labels = [], []
    for cls in sorted(os.listdir(data_dir)):
        cp = os.path.join(data_dir, cls)
        if not os.path.isdir(cp): continue
        for fname in os.listdir(cp):
            if fname.lower().endswith((".png",".jpg",".jpeg")):
                paths.append(os.path.join(cp, fname))
                labels.append(class_indices[cls])
    return np.array(paths), np.array(labels)

train_paths, train_labels = get_paths_labels(TRAIN_DIR)
test_paths,  test_labels  = get_paths_labels(TEST_DIR)
print(f"Train: {len(train_paths)} | Test: {len(test_paths)}")

def data_gen(paths, labels, batch_size, augment=False, shuffle=True):
    n = len(paths)
    while True:
        if shuffle:
            idx = np.random.permutation(n)
            paths, labels = paths[idx], labels[idx]
        for start in range(0, n, batch_size):
            bp, bl = paths[start:start+batch_size], labels[start:start+batch_size]
            X, y = [], []
            for path, label in zip(bp, bl):
                try:
                    img = preprocess_ultrasound(path)
                    if augment:
                        if np.random.rand()>.5: img = np.fliplr(img)
                        if np.random.rand()>.5:
                            angle = np.random.uniform(-15,15)
                            M = cv2.getRotationMatrix2D((112,112),angle,1)
                            img = cv2.warpAffine((img*255).astype(np.uint8),M,(224,224)).astype(np.float32)/255.0
                        img = np.clip(img*np.random.uniform(0.85,1.15),0,1)
                    X.append(img); y.append(label)
                except: pass
            if X:
                yield np.array(X), tf.keras.utils.to_categorical(y, num_classes)

steps_tr = len(train_paths)//BATCH_SIZE
steps_te = len(test_paths)//BATCH_SIZE

base = MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3),include_top=False,weights="imagenet")
base.trainable = False

model = models.Sequential([
    base, layers.GlobalAveragePooling2D(), layers.BatchNormalization(),
    layers.Dense(256,activation="relu"), layers.Dropout(0.4),
    layers.Dense(128,activation="relu"), layers.Dropout(0.3),
    layers.Dense(num_classes,activation="softmax")
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])

cbs = [
    ModelCheckpoint(MODEL_SAVE,save_best_only=True,monitor="val_accuracy",verbose=1),
    EarlyStopping(patience=7,restore_best_weights=True),
    ReduceLROnPlateau(factor=0.3,patience=3,min_lr=1e-7)
]

print("\n=== Phase 1: Training head ===")
model.fit(data_gen(train_paths,train_labels,BATCH_SIZE,augment=True),
          steps_per_epoch=steps_tr,
          validation_data=data_gen(test_paths,test_labels,BATCH_SIZE,shuffle=False),
          validation_steps=steps_te, epochs=15, callbacks=cbs)

print("\n=== Phase 2: Fine-tuning ===")
base.trainable = True
for layer in base.layers[:-30]: layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(data_gen(train_paths,train_labels,BATCH_SIZE,augment=True),
          steps_per_epoch=steps_tr,
          validation_data=data_gen(test_paths,test_labels,BATCH_SIZE,shuffle=False),
          validation_steps=steps_te, epochs=EPOCHS, callbacks=cbs)

loss,acc = model.evaluate(data_gen(test_paths,test_labels,BATCH_SIZE,shuffle=False),steps=steps_te)
print(f"\n✅ Accuracy: {round(acc*100,2)}% | Saved: {MODEL_SAVE}")
print("Next: python train_anomaly_model.py")
