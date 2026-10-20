import pandas as pd
import os

from preprocessing.audio_loader import load_audio
from preprocessing.segmenter import segment_audio
from preprocessing.align_bpm import (
    load_bpm,
    align_bpm_to_segments
)

from features.audio_features import extract_features

from utils.file_utils import get_subject_ids


# ----------------------------------------
# Paths
# ----------------------------------------

audio_folder = r"E:\Speech for HR & BP\Voice_HR_Project\data\raw_audio"

bpm_folder = r"E:\Speech for HR & BP\Voice_HR_Project\data\bpm"


# ----------------------------------------
# Get Subjects
# ----------------------------------------

subject_ids = get_subject_ids(audio_folder)

print(f"Found {len(subject_ids)} subjects")


# ----------------------------------------
# Master Dataset
# ----------------------------------------

master_dataset = []


# ----------------------------------------
# Process Each Subject
# ----------------------------------------

for subject_id in subject_ids:

    print(f"\nProcessing Subject: {subject_id}")

    audio_path = os.path.join(
        audio_folder,
        f"{subject_id}.wav"
    )

    bpm_path = os.path.join(
        bpm_folder,
        f"{subject_id}.csv"
    )

    # -----------------------------
    # Load Audio
    # -----------------------------

    y, sr = load_audio(audio_path)

    # -----------------------------
    # Segment Audio
    # -----------------------------

    segments = segment_audio(
        y,
        sr,
        window_size=5
    )

    # -----------------------------
    # Load BPM
    # -----------------------------

    bpm_df = load_bpm(bpm_path)

    bpm_targets = align_bpm_to_segments(
        bpm_df,
        len(segments),
        window_size=5
    )

    # -----------------------------
    # Feature Extraction
    # -----------------------------

    for i, segment in enumerate(segments):

        features = extract_features(segment, sr)

        features["target_bpm"] = bpm_targets[i]

        features["subject_id"] = subject_id

        master_dataset.append(features)

    print(f"Finished Subject {subject_id}")


# ----------------------------------------
# Save Dataset
# ----------------------------------------

df = pd.DataFrame(master_dataset)

df.to_csv(
    "results/master_features_dataset.csv",
    index=False
)

print("\nMASTER DATASET CREATED!")

print(df.head())

print(f"\nTotal Samples: {len(df)}")