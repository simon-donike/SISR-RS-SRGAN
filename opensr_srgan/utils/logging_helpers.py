import io

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .spectral_helpers import minmax_percentile

def _to_numpy_img(t: torch.Tensor):
    """
    Expects t shape (C,H,W) on CPU; returns (H,W,3) or (H,W) numpy.
    """
    if t.dim() != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(t.shape)}")
    C, H, W = t.shape
    t = t.clamp(0, 1)
    if C == 1:
        out = np.array(t[0].contiguous())
        return out               # grayscale

    if C in (3, 4):
        rgb = t[:3]            
        out = np.array(rgb.permute(1, 2, 0).contiguous())
        return out

    return np.array(t.permute(1, 2, 0).contiguous())
    # Multichannel (first 3 shown upstream)

def plot_tensors(lr, sr, hr, title="Train"):
    # --- denorm + stretch on whatever device you're using ---
    lr = minmax_percentile(lr)
    sr = minmax_percentile(sr)
    hr = minmax_percentile(hr)

    # clamp in-place-friendly
    lr, sr, hr = lr.clamp(0, 1), sr.clamp(0, 1), hr.clamp(0, 1)

    # shapes
    B, C, H, W = lr.shape  # (B,C,H,W)
    # limit to max_n
    max_n = 2
    if B > max_n:
        lr = lr[:max_n]
        sr = sr[:max_n]
        hr = hr[:max_n]
        B = max_n

    # figure/axes: always 2D array even for B==1
    fixed_width = 15
    variable_height = (15 / 3) * B
    fig, axes = plt.subplots(B, 3, figsize=(fixed_width, variable_height), squeeze=False)

    # loop over batch
    with torch.no_grad():
        for i in range(B):
            img_lr = _to_numpy_img(lr[i].detach().cpu())
            img_sr = _to_numpy_img(sr[i].detach().cpu())
            img_hr = _to_numpy_img(hr[i].detach().cpu())

            axes[i, 0].imshow(img_lr)
            axes[i, 0].set_title('LR'); axes[i, 0].axis('off')

            axes[i, 1].imshow(img_sr)
            axes[i, 1].set_title('SR'); axes[i, 1].axis('off')

            axes[i, 2].imshow(img_hr)
            axes[i, 2].set_title('HR'); axes[i, 2].axis('off')

    fig.suptitle(title)
    fig.tight_layout()

    # render to PIL
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    pil_image = Image.open(buf).convert("RGB").copy()
    buf.close()
    plt.close(fig)
    return pil_image


