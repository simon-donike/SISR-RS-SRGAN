from huggingface_hub import hf_hub_download
import zipfile
import os

def get_example_dataset(out_dir: str = "example_dataset/"):
    """Download and extract the example dataset for SRGAN training."""
    # make sure the target dir exists
    os.makedirs(out_dir, exist_ok=True)

    # download the file from your repo
    repo_id = "simon-donike/SR-GAN"
    filename = "example_dataset.zip"

    print("Downloading from Hugging Face Hub...")
    zip_path = hf_hub_download(repo_id=repo_id, filename=filename)

    # unzip after download
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

    print(f"✅ Extracted dataset to: {os.path.abspath(out_dir)}")
    
    # delete the zip file to save space
    os.remove(zip_path)