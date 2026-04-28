import cv2
import numpy as np

def preprocess_ultrasound(img_input, target_size=(224, 224)):
    """
    2025 best-practice ultrasound preprocessing pipeline.
    Steps: Median Filter → CLAHE → Bilateral Filter → Resize → RGB → Normalize
    Accepts: file path (str), numpy array, or PIL Image
    Returns: numpy array shape (224, 224, 3) float32
    """
    if isinstance(img_input, str):
        img = cv2.imread(img_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot load: {img_input}")
    elif isinstance(img_input, np.ndarray):
        img = cv2.cvtColor(img_input, cv2.COLOR_RGB2GRAY) if len(img_input.shape) == 3 else img_input.copy()
    else:
        img = np.array(img_input.convert("L"))

    # Step 1: Median Filter - removes speckle noise
    img = cv2.medianBlur(img, 5)

    # Step 2: CLAHE - local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)

    # Step 3: Bilateral Filter - edge-preserving smoothing
    img = cv2.bilateralFilter(img, 9, 75, 75)

    # Step 4: Resize
    img = cv2.resize(img, target_size)

    # Step 5: Grayscale to RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Step 6: Normalize to [0,1]
    return img.astype(np.float32) / 255.0


def preprocess_batch(img_input, target_size=(224, 224)):
    """Returns shape (1, 224, 224, 3) for model.predict()"""
    return np.expand_dims(preprocess_ultrasound(img_input, target_size), axis=0)
