import subprocess
import numpy as np
import soundfile as sf
import tempfile
import os
import imageio_ffmpeg


def convert_to_pcm(input_path):

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav_path = temp_wav.name
    temp_wav.close()

    command = [
        ffmpeg_path,
        "-i", input_path,
        "-ac", "1",          # mono
        "-ar", "16000",      # sample rate
        "-y",
        temp_wav_path
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return temp_wav_path


def load_audio(audio_path):

    converted_path = convert_to_pcm(audio_path)

    y, sr = sf.read(converted_path)

    os.remove(converted_path)

    return y, sr


if __name__ == "__main__":

    audio_path = r"E:\Speech for HR & BP\Voice_HR_Project\data\raw_audio\1.wav"

    y, sr = load_audio(audio_path)

    print(f"Audio Shape: {y.shape}")
    print(f"Sample Rate: {sr}")
    print(f"Duration: {len(y)/sr:.2f} seconds")