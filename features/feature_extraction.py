import librosa
import numpy as np


def extract_features(audio_path):

    y, sr = librosa.load(audio_path, sr=None)

    features = {}

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    features["mfcc_mean"] = np.mean(mfcc)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)

    features["zcr"] = np.mean(zcr)

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    features["centroid"] = np.mean(centroid)

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    features["bandwidth"] = np.mean(bandwidth)

    # RMS Energy
    rms = librosa.feature.rms(y=y)

    features["rms"] = np.mean(rms)

    return features
