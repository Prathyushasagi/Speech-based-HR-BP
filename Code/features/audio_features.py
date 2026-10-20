import numpy as np
import librosa


def extract_features(y, sr):

    features = {}

   
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    for i in range(13):

        features[f"mfcc_{i+1}_mean"] = np.mean(mfccs[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfccs[i])

    
    delta_mfcc = librosa.feature.delta(mfccs)

    for i in range(13):

        features[f"delta_mfcc_{i+1}_mean"] = np.mean(delta_mfcc[i])
        features[f"delta_mfcc_{i+1}_std"] = np.std(delta_mfcc[i])

   
    pitches, magnitudes = librosa.piptrack(
        y=y,
        sr=sr
    )

    pitch_values = pitches[pitches > 0]

    if len(pitch_values) > 0:
        features["pitch_mean"] = np.mean(pitch_values)
        features["pitch_std"] = np.std(pitch_values)
    else:
        features["pitch_mean"] = 0
        features["pitch_std"] = 0

  
    rms = librosa.feature.rms(y=y)

    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)

  
    zcr = librosa.feature.zero_crossing_rate(y)

    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)

   
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    features["spectral_centroid_mean"] = np.mean(centroid)
    features["spectral_centroid_std"] = np.std(centroid)

   
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    features["spectral_bandwidth_mean"] = np.mean(bandwidth)
    features["spectral_bandwidth_std"] = np.std(bandwidth)

   
    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    features["spectral_rolloff_mean"] = np.mean(rolloff)
    features["spectral_rolloff_std"] = np.std(rolloff)

    return features