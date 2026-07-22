import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os


def generate_waveform(audio_path, output_path):

    audio, sr = librosa.load(audio_path, sr=None)

    plt.figure(figsize=(10,3))

    librosa.display.waveshow(audio, sr=sr)

    plt.title("Waveform")

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()


def generate_spectrogram(audio_path, output_path):

    audio, sr = librosa.load(audio_path, sr=None)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(10,4))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(format="%+2.0f dB")

    plt.title("Mel Spectrogram")

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()