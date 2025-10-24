import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from pathlib import Path
import numpy as np


class ExampleDataset(Dataset):
    """Simple example dataset for SRGAN training and validation.

    Loads preprocessed high-resolution (HR) image patches stored as ``.npz`` files
    and generates corresponding low-resolution (LR) inputs via on-the-fly bicubic
    downsampling. Used primarily for demonstration, testing, and quick-start
    examples.

    Each ``.npz`` file typically contains a 3D array of shape ``(H, W, C)``
    representing an image in reflectance units. The dataset automatically splits
    files into training and validation sets based on their order.

    Args:
        folder (str | Path): Path to the dataset folder containing ``.npz`` files.
        phase (str, optional): Dataset split, one of {"train", "val"}.
            Defaults to "train". The last 20 samples are reserved for validation.

    Attributes:
        scale (int): Downsampling factor between HR and LR (default: 4).
        files (list[Path]): Sorted list of dataset files.
        key (str): Preferred key for array extraction within the .npz archives.

    Raises:
        FileNotFoundError: If the specified folder does not exist.
        ValueError: If an invalid ``phase`` argument is provided.

    Example:
        >>> dataset = ExampleDataset("example_dataset/", phase="train")
        >>> lr, hr = dataset[0]
        >>> lr.shape, hr.shape
        (torch.Size([C, H/4, W/4]), torch.Size([C, H, W]))
    """

    def __init__(self, folder, phase="train"):
        # check if input folder exists
        if not Path(folder).is_dir():
            raise FileNotFoundError(
                f"Dataset folder '{folder}' does not exist. "
                f"Please download the example dataset first."
                f" See 'opensr_srgan/data/example_data/download_example_dataset.py'."
            )
        self.scale = 4
        self.files = sorted(Path(folder).glob("hr_*.npz")) or sorted(
            Path(folder).glob("*.npz")
        )
        self.key = "hr"  # try this key first; fallback to first array in file

        if phase == "train":
            self.files = self.files[:-20]
        elif phase == "val":
            self.files = self.files[-20:]
        else:
            raise ValueError(f"Unknown phase '{phase}'")

    def __len__(self):
        """Return the total number of image samples available.

        Returns:
            int: The number of .npz files in the dataset split.
        """
        return len(self.files)

    def _load_npz(self, path):
        """Load a single .npz file and extract the stored image array.

        If the expected key ("hr") is not found, falls back to the first
        array in the archive.

        Args:
            path (str | Path): Path to the .npz file.

        Returns:
            np.ndarray: Image array of shape (H, W, C), typically dtype uint16 or float32.
        """
        with np.load(path) as z:
            if self.key in z:
                arr = z[self.key]
            else:
                # fallback: first array in the archive
                arr = z[list(z.files)[0]]
        return arr  # (H, W, C), typically uint16

    def __getitem__(self, idx):
        """Retrieve one (LR, HR) image pair from the dataset.

        Loads the high-resolution image from an ``.npz`` file, converts it to a
        normalized PyTorch tensor, and generates a corresponding low-resolution
        version using bicubic interpolation.

        Steps:
            1. Load HR image and convert to float32 tensor in CHW format.
            2. Normalize to [0, 1] if reflectance values exceed 1.5 (Sentinel-2 style).
            3. Downsample by ``self.scale`` using bicubic interpolation to create LR.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - ``lr``: Low-resolution image tensor, shape (C, H/scale, W/scale)
                - ``hr``: High-resolution image tensor, shape (C, H, W)
        """

        # load HR image
        hr_np = self._load_npz(self.files[idx])  # (H, W, C)

        # convert to float32 before torch.from_numpy
        hr_np = hr_np.astype(np.float32)

        # to torch CHW in [0,1]
        hr = torch.from_numpy(hr_np).permute(2, 0, 1)
        if hr.max() > 1.5:
            hr = hr / 10000.0  # Sentinel-2 normalization

        # make LR by bicubic downsample
        H, W = hr.shape[1], hr.shape[2]
        lr = F.interpolate(
            hr.unsqueeze(0),
            size=(H // self.scale, W // self.scale),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
        return lr, hr


if __name__ == "__main__":
    # simple test
    dataset = ExampleDataset("example_dataset/", phase="val")
