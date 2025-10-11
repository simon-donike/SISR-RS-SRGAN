import torch
import numpy as np
import rasterio as rio
from tacoreader import TortillaDataFrame
import tacoreader

class SEN2NAIP(torch.utils.data.Dataset):
    def __init__(self, taco_file):
        # Load the dataset once in memory
        self.dataset: TortillaDataFrame = tacoreader.load(taco_file)
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Cache the file paths to avoid redundant Parquet reads
        if idx not in self.cache:
            sample: TortillaDataFrame = self.dataset.read(idx)
            lr: str = sample.read(0)
            hr: str = sample.read(1)
            self.cache[idx] = (lr, hr)
        else:
            lr, hr = self.cache[idx]

        # Open the files and load data
        with rio.open(lr) as src, rio.open(hr) as dst:
            lr_data: np.ndarray = src.read()
            hr_data: np.ndarray = dst.read()

        return lr_data, hr_data