import torch
import matplotlib.pyplot as plt
from PIL import Image
import io
import torchvision.transforms as transforms
from utils.sen2_stretch import sen2_stretch,minmax_percentile
from utils.normalise_s2 import normalise_s2


def _to_numpy_img(t: torch.Tensor):
    """
    Expects t shape (C,H,W) on CPU; returns (H,W,3) or (H,W) numpy.
    """
    if t.dim() != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(t.shape)}")
    C, H, W = t.shape
    t = t.clamp(0, 1)
    if C == 1:
        return t[0].contiguous().numpy()               # grayscale
    else:
        return t.permute(1, 2, 0).contiguous().numpy() # RGB/Multichannel (first 3 shown upstream)

def plot_tensors(lr, sr, hr, title="Train"):
    # --- denorm + stretch on whatever device you're using ---
    #sr = normalise_s2(sr, stage="denorm")
    #lr = normalise_s2(lr, stage="denorm")
    #hr = normalise_s2(hr, stage="denorm")
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

def plot_fusion(lr,enc,hr):

    # prepare tensors
    enc = normalise_s2(enc,stage="denorm")
    lr = normalise_s2(lr,stage="denorm")
    hr = normalise_s2(hr,stage="denorm")
    lr = sen2_stretch(lr)
    enc = sen2_stretch(enc)
    hr = sen2_stretch(hr)
    lr, enc, hr = torch.clamp(lr,0,1), torch.clamp(enc,0,1), torch.clamp(hr,0,1)

    # create fig and prepare tensors
    lr,enc,hr = lr[0],enc[0],hr[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    lr = lr.detach().cpu().numpy().transpose(1, 2, 0)
    enc = enc.detach().cpu().numpy().transpose(1, 2, 0)
    hr = hr.detach().cpu().numpy().transpose(1, 2, 0)

    # plot images
    axes[0].imshow(lr)
    axes[0].axis('off')
    axes[0].set_title('LR')
    axes[1].imshow(enc)
    axes[1].axis('off')
    axes[1].set_title('Fused Image')
    axes[2].imshow(hr)
    axes[2].axis('off')
    axes[2].set_title('HR')

    # save and return image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    pil_image = Image.open(buf)
    plt.close()
    return pil_image

def misr_plot(lr,sr,hr):
    import numpy as np
    import matplotlib.pyplot as plt
    import torchvision

    # denorm etc
    hr = normalise_s2(hr,stage="denorm")
    hr = sen2_stretch(hr)
    sr = normalise_s2(sr,stage="denorm")
    sr = sen2_stretch(sr)
    lr = (((lr+1)/2)*10)/3
    lr = lr.clamp(0,1)
    sr = sr.clamp(0,1)

    # create LR grid
    lr = lr[0] # get first batch
    dim = lr.shape[-3] # get bands*views
    bands = 3 # get bands, hardcoded as 3
    images = int(dim/3) # get number of images
    #lr = lr.view(images,bands,lr.shape[-2], lr.shape[-1]) # reshape to go from V*B,W,H to V,B,W,H
    lr = torchvision.utils.make_grid(lr,nrow=2).cpu().numpy().transpose((1,2,0)) # make grid of images
    sr = sr[0].cpu().numpy().transpose((1,2,0))
    hr = hr[0].cpu().numpy().transpose((1,2,0))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(lr)
    axes[0].axis('off')
    axes[0].set_title('LR')
    axes[1].imshow(sr)
    axes[1].axis('off')
    axes[1].set_title('SR')
    axes[2].imshow(hr)
    axes[2].axis('off')
    axes[2].set_title('HR')

    # save and return image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    pil_image = Image.open(buf)
    plt.close()
    return pil_image

#lr = torch.rand(1,12,300,300)
#sr = torch.rand(1,3,300,300)
#hr = torch.rand(1,3,300,300)
#misr_plot(lr,sr,hr)


