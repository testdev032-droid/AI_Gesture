"""Utility helpers for loading a Teachable Machine model and predicting gestures."""

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image
import streamlit as st

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - handled safely in app
    tf = None


# Teachable Machine image projects usually use 224x224 images.
TM_IMAGE_SIZE = (224, 224)


@st.cache_resource(show_spinner=False)
def load_local_teachable_machine_model(model_path: str, labels_path: str):
    """
    Load the local Keras model and labels one time.

    We cache the model so Streamlit does not reload it on every rerun.
    """
    if tf is None:
        raise ImportError(
            "TensorFlow is not installed. Use Python 3.10 or 3.11 and install requirements.txt."
        )

    model_file = Path(model_path)
    label_file = Path(labels_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not label_file.exists():
        raise FileNotFoundError(f"Labels file not found: {label_file}")

    # compile=False keeps loading lighter and avoids training-only settings.
    model = tf.keras.models.load_model(model_file, compile=False)
    labels = load_labels(label_file)
    return model, labels


def load_labels(labels_path: Path) -> List[str]:
    """Read class labels from labels.txt."""
    labels: List[str] = []
    with open(labels_path, "r", encoding="utf-8") as file:
        for line in file:
            cleaned = line.strip()
            if not cleaned:
                continue

            # Teachable Machine often writes labels like: "0 Open Palm"
            parts = cleaned.split(" ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1].strip())
            else:
                labels.append(cleaned)
    return labels


def preprocess_image_for_model(image: Image.Image, target_size=TM_IMAGE_SIZE) -> np.ndarray:
    """
    Prepare a PIL image for a Teachable Machine Keras image model.

    Many Teachable Machine Keras exports expect pixels normalized to [-1, 1].
    """
    rgb_image = image.convert("RGB")
    resized = rgb_image.resize(target_size)
    image_array = np.asarray(resized).astype(np.float32)
    normalized = (image_array / 127.5) - 1.0
    return np.expand_dims(normalized, axis=0)


def get_top_predictions(probabilities: np.ndarray, labels: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
    """Return the best predictions for display."""
    scored = list(enumerate(probabilities.tolist()))
    scored.sort(key=lambda item: item[1], reverse=True)

    top_predictions: List[Dict[str, Any]] = []
    for class_index, confidence in scored[:top_k]:
        label = labels[class_index] if class_index < len(labels) else f"Class {class_index}"
        top_predictions.append({
            "label": label,
            "confidence": float(confidence),
        })
    return top_predictions


def predict_gesture_from_image(model, labels: List[str], image: Image.Image) -> Dict[str, Any]:
    """Predict the gesture class from a captured webcam image."""
    input_tensor = preprocess_image_for_model(image)
    raw_prediction = model.predict(input_tensor, verbose=0)[0]

    best_index = int(np.argmax(raw_prediction))
    best_label = labels[best_index] if best_index < len(labels) else f"Class {best_index}"
    best_confidence = float(raw_prediction[best_index])

    return {
        "label": best_label,
        "confidence": best_confidence,
        "top_predictions": get_top_predictions(raw_prediction, labels, top_k=3),
    }
