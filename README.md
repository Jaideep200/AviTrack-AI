# 🐦 AviTrack AI
### AI-Powered Bird Audio Analysis & Population Estimation

AviTrack AI is a Flask-based web application that analyzes bird audio recordings using Digital Signal Processing (DSP) and Machine Learning techniques. The system extracts meaningful acoustic features from uploaded `.wav` files, generates waveform and Mel spectrogram visualizations, and predicts the estimated bird population from the recording.

The project is inspired by the **BioDCASE 2026 Bird Counting Challenge**, which focuses on passive acoustic monitoring for automated bird population estimation.

---

# 📖 Table of Contents

- Project Overview
- Features
- Technologies Used
- Project Architecture
- Folder Structure
- Installation
- Usage
- Machine Learning Pipeline
- Dataset
- Project Workflow
- Screenshots
- Future Improvements
- Author
- License

---

# 🚀 Project Overview

Traditional bird population monitoring requires experts to manually analyze audio recordings collected from forests, wetlands, and aviaries. This process is time-consuming, labor-intensive, and difficult to scale.

AviTrack AI automates this process by:

- Uploading bird audio recordings
- Extracting acoustic features
- Visualizing audio signals
- Predicting bird population using Machine Learning

The application provides a simple and interactive web interface for researchers, students, and wildlife enthusiasts.

---

# ✨ Features

## Audio Upload

- Upload `.wav` audio recordings
- File type validation
- Secure file handling

---

## Audio Analysis

Automatically extracts:

- Duration
- Sample Rate
- Audio Channels

---

## Audio Visualization

Generate:

- Waveform
- Mel Spectrogram

---

## Feature Extraction

Extracts acoustic features from uploaded recordings using Librosa.

Examples include:

- MFCC
- Spectral Centroid
- Spectral Bandwidth
- Zero Crossing Rate
- RMS Energy
- Spectral Roll-off

---

## Machine Learning Prediction

Uses a trained Random Forest model to estimate:

- Bird Population Count

---

## Interactive Web Interface

- Responsive design
- Modern landing page
- Upload dashboard
- Analysis result page

---

# 🛠 Technologies Used

## Backend

- Python 3
- Flask

## Machine Learning

- Scikit-learn
- Joblib

## Audio Processing

- Librosa
- SoundFile
- NumPy

## Visualization

- Matplotlib

## Frontend

- HTML5
- CSS3
- Jinja2 Templates

---

# 🏗 Project Architecture

```
                User
                  │
                  ▼
          Upload WAV File
                  │
                  ▼
             Flask Server
                  │
                  ▼
        Audio Preprocessing
                  │
                  ▼
        Feature Extraction
                  │
         ┌────────┴────────┐
         ▼                 ▼
    Waveform         Mel Spectrogram
         │                 │
         └────────┬────────┘
                  ▼
       Machine Learning Model
                  │
                  ▼
     Estimated Bird Population
                  │
                  ▼
          Display Results
```

---

# 📂 Project Structure

```
AviTrack-AI/

│
├── app.py
├── bird_model.pkl
├── requirements.txt
│
├── dataset_processing/
│   └── process_dataset.py
│
├── preprocessing/
│   └── preprocessing.py
│
├── features/
│   └── feature_extraction.py
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── style.css
│   ├── images/
│   └── generated/
│
├── uploads/
│
└── README.md
```

---

# ⚙ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/AviTrack-AI.git

cd AviTrack-AI
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv
```

Activate

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

# 🧠 Machine Learning Pipeline

The current implementation follows this workflow:

```
Bird Audio
      │
      ▼
Feature Extraction
      │
      ▼
Training Dataset
      │
      ▼
Random Forest Regressor
      │
      ▼
bird_model.pkl
      │
      ▼
Prediction
```

The trained model is loaded automatically when the Flask application starts.

---

# 📊 Dataset

This project uses the **BioDCASE 2026 Bird Counting Dataset**.

The dataset contains:

- Bird audio recordings
- Population counts
- Species metadata
- Target bird labels

The project currently uses a processed subset of the dataset for model training.

---

# 🔄 Project Workflow

```
Upload WAV File

↓

Read Audio

↓

Generate Waveform

↓

Generate Mel Spectrogram

↓

Extract Audio Features

↓

Load Trained Model

↓

Predict Bird Count

↓

Display Analysis Results
```

---



# 📄 License

This project is developed for educational and research purposes.

Inspired by the **BioDCASE 2026 Bird Counting Challenge**.

---
