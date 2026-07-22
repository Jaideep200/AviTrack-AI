from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Emreargin/BioDCASE2026_Bird_Counting",
    repo_type="dataset",
    local_dir="dataset"
)

print("Download Complete!")