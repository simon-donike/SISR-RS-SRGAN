import torch
import opensr_utils
from omegaconf import OmegaConf
import os

# set visible GPUs and device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model and weights
from model.SRGAN import SRGAN_model
model = SRGAN_model(config_file_path="configs/config_20m.yaml")
model = model.eval()
model.load_from_checkpoint("checkpoints/srgan-20m-6band/last.ckpt", strict=False)

# Set up Sen2 Inference Pipeline
sen2_path = "data/S2A_MSIL2A_20230901T104031_N0509_R137_T31TFJ_20230901T130204.SAFE" # Set Path to file or folder
sr_object = opensr_utils.large_file_processing(
			root=sen2_path,            # File or Folder path
			model=model,               # SR model
			window_size=(128, 128),    # LR window size for model input
			factor=4,                  # SR factor (10m → 2.5m)
			overlap=12,                # overlapping pixels for mosaic stitching
			eliminate_border_px=2,     # No of discarded border pixels per prediction
			device=device,             # "cuda" for GPU-accelerated inference
			gpus=[0],                   # pass GPU ID (int) or list of GPUs
			save_preview=False,        # save a low-res preview of the output, and a tif georef
			debug=False,
		)
sr_object.start_super_resolution()
