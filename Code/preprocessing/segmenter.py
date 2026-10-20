import numpy as np


def segment_audio(y, sr, window_size=5):

    """
    Segment audio into fixed windows.

    Args:
        y: audio signal
        sr: sample rate
        window_size: seconds

    Returns:
        segments: list of audio segments
    """

    segment_length = window_size * sr

    segments = []

    for start in range(0, len(y), segment_length):

        end = start + segment_length

        segment = y[start:end]

        # Skip incomplete segments
        if len(segment) == segment_length:
            segments.append(segment)

    return segments


if __name__ == "__main__":

    from audio_loader import load_audio

    audio_path = r"E:\Speech for HR & BP\Voice_HR_Project\data\raw_audio\1.wav"

    y, sr = load_audio(audio_path)

    segments = segment_audio(y, sr, window_size=5)

    print(f"Total Segments: {len(segments)}")

    print(f"Each Segment Shape: {segments[0].shape}")

    print(f"Segment Duration: {len(segments[0])/sr:.2f} sec")