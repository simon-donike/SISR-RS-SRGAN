import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from pathlib import Path
import numpy as np

class ExampleDataset(Dataset):
    def __init__(self, folder, phase = "train"):
        # check if input folder exists
        if not Path(folder).is_dir():
            raise FileNotFoundError(f"Dataset folder '{folder}' does not exist. "
                                    f"Please download the example dataset first."
                                    f" See 'opensr_srgan/data/example_data/download_example_dataset.py'.")
        self.scale = 4
        self.files = sorted(Path(folder).glob("hr_*.npz")) or sorted(Path(folder).glob("*.npz"))
        self.key = "hr"  # try this key first; fallback to first array in file
        
        if phase == "train":
            self.files = self.files[:-20]
        elif phase == "val":
            self.files = self.files[-20:]
        else:
            raise ValueError(f"Unknown phase '{phase}'")
                
        
    def __len__(self):
        return len(self.files)

    def _load_npz(self, path):
        with np.load(path) as z:
            if self.key in z:
                arr = z[self.key]
            else:
                # fallback: first array in the archive
                arr = z[list(z.files)[0]]
        return arr  # (H, W, C), typically uint16

    def __getitem__(self, idx):
        # load HR image
        hr_np = self._load_npz(self.files[idx])              # (H, W, C)

        # convert to float32 before torch.from_numpy
        hr_np = hr_np.astype(np.float32)

        # to torch CHW in [0,1]
        hr = torch.from_numpy(hr_np).permute(2, 0, 1)
        if hr.max() > 1.5:
            hr = hr / 10000.0  # Sentinel-2 normalization

        # make LR by bicubic downsample
        H, W = hr.shape[1], hr.shape[2]
        lr = F.interpolate(hr.unsqueeze(0),
                        size=(H // self.scale, W // self.scale),
                        mode="bicubic",
                        align_corners=False).squeeze(0)
        return lr, hr


if __name__ == "__main__":
    # simple test
    dataset = ExampleDataset("example_dataset/", phase="val")
