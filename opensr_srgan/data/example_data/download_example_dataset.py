from huggingface_hub import hf_hub_download
import zipfile, os


def get_example_dataset(out_dir: str = "example_dataset/"):
    """Download and extract the bundled example dataset from Hugging Face Hub.

    Retrieves a small prepackaged example dataset used for SRGAN demonstrations
    and tests. The function ensures deterministic extraction by stripping any
    top-level folder prefixes (e.g., ``example_data/``) from the archive so that
    the files always end up directly under the specified output directory.

    Args:
        out_dir (str, optional): Target directory for extraction.
            Defaults to ``"example_dataset/"``.

    Behaviour:
        1. Creates the output folder if it does not exist.
        2. Downloads ``example_dataset.zip`` from the repository
           ``simon-donike/SR-GAN`` on Hugging Face Hub.
        3. Extracts the archive contents into ``out_dir``, removing any redundant
           root folder structure for cleaner layout.
        4. Deletes the downloaded zip file after extraction.

    Returns:
        None

    Example:
        >>> get_example_dataset()
        📦 Downloading from Hugging Face Hub...
        ✅ Extracted dataset to: /path/to/example_dataset
    """
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
                target = member[len(prefix) :]
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
