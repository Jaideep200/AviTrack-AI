import os
import pandas as pd

from features.feature_extraction import extract_features

# -------------------------------------------------
# Paths
# -------------------------------------------------

AUDIO_FOLDER = r"BioDCASE_Subset\dev_aviary_1\chunk_000"

OUTPUT_CSV = "dataset_features.csv"

# -------------------------------------------------
# Store extracted features
# -------------------------------------------------

data = []

# -------------------------------------------------
# Read every WAV file
# -------------------------------------------------

for filename in os.listdir(AUDIO_FOLDER):

    if filename.endswith(".wav"):

        filepath = os.path.join(
            AUDIO_FOLDER,
            filename
        )

        print(f"Processing: {filename}")

        features = extract_features(filepath)

        row = {
    "filename": filename,
    "aviary_id": "dev_aviary_1",
    "mfcc_mean": features["mfcc_mean"],
    "zcr": features["zcr"],
    "centroid": features["centroid"],
    "bandwidth": features["bandwidth"],
    "rms": features["rms"]
    }
        data.append(row)

# -------------------------------------------------
# Save CSV
# -------------------------------------------------

df = pd.DataFrame(data)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("--------------------------------")
print("Dataset processing completed!")
print(f"Total audio files: {len(df)}")
print(f"CSV saved as: {OUTPUT_CSV}")