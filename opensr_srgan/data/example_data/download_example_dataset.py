from huggingface_hub import hf_hub_download
import zipfile, os

def get_example_dataset(out_dir: str = "example_dataset/"):
    """Download and extract the example dataset for SRGAN training."""
    os.makedirs(out_dir, exist_ok=True)

    repo_id = "simon-donike/SR-GAN"
    filename = "example_dataset.zip"

    print("📦 Downloading from Hugging Face Hub...")
    zip_path = hf_hub_download(repo_id=repo_id, filename=filename)

    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()

        # detect common top-level folder (e.g. "example_data/")
        prefix = os.path.commonprefix(members)
        if prefix and prefix.endswith("/"):
            for member in members:
                # strip the prefix
                target = member[len(prefix):]
                if not target:  # skip folder itself
                    continue
                target_path = os.path.join(out_dir, target)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with z.open(member) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())
        else:
            z.extractall(out_dir)

    os.remove(zip_path)
    print(f"✅ Extracted dataset to: {os.path.abspath(out_dir)}")
