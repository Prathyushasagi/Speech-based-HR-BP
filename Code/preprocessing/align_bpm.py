import pandas as pd
import numpy as np

from preprocessing.audio_loader import load_audio
from preprocessing.segmenter import segment_audio


def load_bpm(bpm_path):

    df = pd.read_csv(bpm_path)

    return df


def align_bpm_to_segments(bpm_df, num_segments, window_size=5):

    bpm_targets = []

    for i in range(num_segments):

        start_ms = i * window_size * 1000
        end_ms = (i + 1) * window_size * 1000

        segment_bpm = bpm_df[
            (bpm_df["timestamp"] >= start_ms) &
            (bpm_df["timestamp"] < end_ms)
        ]

        avg_bpm = segment_bpm["BPM"].mean()

        bpm_targets.append(avg_bpm)

    return bpm_targets


if __name__ == "__main__":

    audio_path = r"E:\Speech for HR & BP\Voice_HR_Project\data\raw_audio\1.wav"

    bpm_path = r"E:\Speech for HR & BP\Voice_HR_Project\data\bpm\1.csv"

    # Load audio
    y, sr = load_audio(audio_path)

    # Segment audio
    segments = segment_audio(y, sr, window_size=5)

    # Load BPM
    bpm_df = load_bpm(bpm_path)

    # Align BPM
    bpm_targets = align_bpm_to_segments(
        bpm_df,
        len(segments),
        window_size=5
    )

    print(f"Total Segments: {len(segments)}")
    print(f"Total BPM Labels: {len(bpm_targets)}")

    print("\nFirst 5 BPM Targets:")
    print(bpm_targets[:5])