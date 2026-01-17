from huggingface_hub import snapshot_download

output_cc12m_images = "data/cc12m"
output_cc12m_annotations = "data/cc12m_annotations"

print("Download pixparse/cc12m-wds huggingface dataset ...")
snapshot_download(
    repo_id="pixparse/cc12m-wds",
    repo_type="dataset",
    max_workers=4,
    local_dir=output_cc12m_images,
)

print("Download spravil/cc12m_ccmatrix_captions_and_translations huggingface dataset ...")
snapshot_download(
    repo_id="spravil/cc12m_ccmatrix_captions_and_translations",
    repo_type="dataset",
    max_workers=4,
    local_dir=output_cc12m_annotations,
)
