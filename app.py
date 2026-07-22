from flask import Flask, render_template, request
import os
import joblib
import pandas as pd
import soundfile as sf

from features.feature_extraction import extract_features
from preprocessing.preprocessing import (
    generate_waveform,
    generate_spectrogram
)

app = Flask(__name__)

# -----------------------------
# Load Trained AI Model
# -----------------------------
model = joblib.load("bird_model.pkl")

# -----------------------------
# Configuration
# -----------------------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"wav"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create required folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static/generated/waveforms", exist_ok=True)
os.makedirs("static/generated/spectrograms", exist_ok=True)


# -----------------------------
# Helper Function
# -----------------------------
def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Upload Route
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload():

    # Check if file exists
    if "audio" not in request.files:
        return "No file selected."

    file = request.files["audio"]

    # Empty filename
    if file.filename == "":
        return "No file selected."

    # Check extension
    if not allowed_file(file.filename):
        return "Only .wav files are allowed."

    # Save uploaded file
    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    # -----------------------------
    # Generate Waveform
    # -----------------------------
    waveform_path = os.path.join(
        "static",
        "generated",
        "waveforms",
        "waveform.png"
    )

    generate_waveform(
        filepath,
        waveform_path
    )

    # -----------------------------
    # Generate Spectrogram
    # -----------------------------
    spectrogram_path = os.path.join(
        "static",
        "generated",
        "spectrograms",
        "spectrogram.png"
    )

    generate_spectrogram(
        filepath,
        spectrogram_path
    )

    # -----------------------------
    # Read Audio Metadata
    # -----------------------------
    audio, sample_rate = sf.read(filepath)

    duration = len(audio) / sample_rate

    if len(audio.shape) == 1:
        channels = 1
    else:
        channels = audio.shape[1]

    # -----------------------------
    # Extract Audio Features
    # -----------------------------
    features = extract_features(filepath)

    # -----------------------------
    # Prepare Data for Prediction
    # -----------------------------
    input_data = pd.DataFrame([{
        "mfcc_mean": features["mfcc_mean"],
        "zcr": features["zcr"],
        "centroid": features["centroid"],
        "bandwidth": features["bandwidth"],
        "rms": features["rms"]
    }])

    # -----------------------------
    # Predict Bird Count
    # -----------------------------
    predicted_count = model.predict(input_data)[0]

    # -----------------------------
    # Show Result Page
    # -----------------------------
    return render_template(
    "result.html",
    filename=file.filename,
    duration=round(duration, 2),
    sample_rate=sample_rate,
    channels=channels,
    features=features,
    predicted_count=round(predicted_count, 2)
    )


# -----------------------------
# Run Flask App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)