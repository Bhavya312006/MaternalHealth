"""
FILE 1: prepare_image_dataset.py
REPLACES old prepare_image_dataset.py

YOUR FOLDER (from screenshot):
  D:/maternal_project/data/images/FETAL_PLANES_ZENODO/
    Images/                      <- all 12,400 PNG images
    FETAL_PLANES_DB_data.xlsx    <- labels (Image_name, Plane, Train columns)
    FETAL_PLANES_DB_data.xls
    README.md

WHAT IT DOES:
  Reads xlsx → organizes images into train/test by class
  Applies CLAHE preprocessing to every image before saving

RUN: python prepare_image_dataset.py
"""
import os, sys, cv2, numpy as np, pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_preprocess import preprocess_ultrasound

ZENODO_DIR = r"C:\MATERNAL_FINAL\data\images\FETAL_PLANES_ZENODO"
OUTPUT_DIR = r"C:\MATERNAL_FINAL\data\prepared"

def save_preprocessed(img_float, dst_path):
    img_u8 = (img_float * 255).astype(np.uint8)
    cv2.imwrite(dst_path, cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))

def prepare_zenodo():
    print("=== Preparing Zenodo Dataset ===")

    # Read xlsx
    xlsx_path = os.path.join(ZENODO_DIR, "FETAL_PLANES_DB_data.xlsx")
    if not os.path.exists(xlsx_path):
        xlsx_path = os.path.join(ZENODO_DIR, "FETAL_PLANES_DB_data.xls")
    print(f"Reading: {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    print(f"Columns: {list(df.columns)}")
    print(f"First row: {df.iloc[0].to_dict()}")
    print(f"Total rows: {len(df)}")

    # Auto-detect column names
    col_image = col_plane = col_train = None
    for col in df.columns:
        cl = col.lower().strip()
        if "image" in cl and "name" in cl:          col_image = col
        elif cl == "image_name":                     col_image = col
        elif "plane" in cl and "brain" not in cl:   col_plane = col
        elif cl == "train":                          col_train = col

    # Fallback positional
    if col_image is None: col_image = df.columns[0]
    if col_plane is None: col_plane = df.columns[2]
    if col_train is None: col_train = df.columns[-1]

    print(f"Using columns → image:{col_image}, plane:{col_plane}, train:{col_train}")

    images_dir = os.path.join(ZENODO_DIR, "Images")
    if not os.path.exists(images_dir):
        print(f"ERROR: Images folder not found at {images_dir}")
        return

    print(f"Images folder has {len(os.listdir(images_dir))} files")

    processed = skipped = 0
    for _, row in df.iterrows():
        img_name = str(row[col_image]).strip()
        if not img_name.endswith(".png"):
            img_name += ".png"

        plane    = str(row[col_plane]).strip().replace(" ", "_").lower()
        is_train = int(row[col_train]) == 1
        split    = "train" if is_train else "test"

        src_path = os.path.join(images_dir, img_name)
        dst_dir  = os.path.join(OUTPUT_DIR, "classification", split, plane)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, img_name)

        if os.path.exists(dst_path):
            processed += 1
            continue
        if not os.path.exists(src_path):
            skipped += 1
            continue

        try:
            save_preprocessed(preprocess_ultrasound(src_path), dst_path)
            processed += 1
            if processed % 500 == 0:
                print(f"  Processed {processed}...")
        except Exception as e:
            skipped += 1

    print(f"\n  Processed: {processed} | Skipped: {skipped}")

    for split in ["train","test"]:
        split_dir = os.path.join(OUTPUT_DIR, "classification", split)
        if os.path.exists(split_dir):
            print(f"\n  {split} classes:")
            for cls in sorted(os.listdir(split_dir)):
                cp = os.path.join(split_dir, cls)
                if os.path.isdir(cp):
                    print(f"    {cls}: {len(os.listdir(cp))} images")

    print(f"\n✅ Done! Output: {OUTPUT_DIR}/classification/")
    print("Next: python train_fetal_model.py")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prepare_zenodo()
