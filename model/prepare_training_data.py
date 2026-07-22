import pandas as pd

# Load extracted features
features_df = pd.read_csv("dataset_features.csv")

# Load ground truth labels
ground_truth_df = pd.read_csv(
    "BioDCASE_Subset/metadata/ground_truth.csv"
)

# Keep only target bird species
ground_truth_df = ground_truth_df[
    ground_truth_df["is_target"] == 1
]

# Merge on aviary_id
training_df = pd.merge(
    features_df,
    ground_truth_df[["aviary_id", "count"]],
    on="aviary_id",
    how="left"
)

# Rename label column
training_df.rename(
    columns={"count": "bird_count"},
    inplace=True
)

# Save training dataset
training_df.to_csv(
    "training_dataset.csv",
    index=False
)

print("Training dataset created successfully!")
print(training_df.head())