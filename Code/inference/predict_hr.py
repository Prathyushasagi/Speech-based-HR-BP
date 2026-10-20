import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import joblib
import numpy as np

from features.audio_features import extract_features
from preprocessing.audio_loader import load_audio



MODEL_PATH = "results/best_hr_classifier.pkl"


def prepare_features(y, sr):

    features = extract_features(y, sr)

    selected_features = [

        "rms_mean",
        "rms_std",

        "zcr_mean",
        "zcr_std",

        "spectral_centroid_mean",
        "spectral_centroid_std",

        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",

        "spectral_rolloff_mean",
        "spectral_rolloff_std",

        "pitch_mean",
        "pitch_std"
    ]

    feature_vector = []

    for feature in selected_features:

        feature_vector.append(
            features[feature]
        )

    return np.array(feature_vector).reshape(1, -1)


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

def predict_hr_class(model, X):

    prediction = model.predict(X)[0]

    class_map = {
        0: "LOW HR",
        1: "MEDIUM HR",
        2: "HIGH HR"
    }

    return class_map[prediction]


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

if __name__ == "__main__":

    print("\n===================================")
    print("VOICE-BASED HR STATE PREDICTION")
    print("===================================")

    audio_path = input(
        "\nEnter WAV file path: "
    )

    print("\nLoading trained model...")

    model = joblib.load(MODEL_PATH)

    print("Processing audio...")

    y, sr = load_audio(audio_path)

    X = prepare_features(y, sr)

    prediction = predict_hr_class(
        model,
        X
    )

    print("\n===================================")

    print(f"PREDICTED HR STATE: {prediction}")

    print("===================================\n")