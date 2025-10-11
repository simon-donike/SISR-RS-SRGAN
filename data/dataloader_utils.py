import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision
from skimage import exposure


"""
GENERAL
"""
def plot_tensor(im):
    pass

def minmax(img):
    min_val = torch.min(img)
    max_val = torch.max(img)
    normalized_img = (img - min_val) / (max_val - min_val)
    return normalized_img

"""
SPECTRAL MATCHING METHODS
"""
# Normal Standardization over whole dataset
def normalize(sen2,spot6,sen2_amount=1):
    transform_spot = transforms.Compose([transforms.Normalize(mean=[479.0, 537.0, 344.0], std=[430.0, 290.0, 229.0]) ])
    # dynamically define transform to reflect shape of tensor
    trans_mean,trans_std = [78.0, 91.0, 62.0]*sen2_amount,[36.0, 28.0, 30.0]*sen2_amount
    transform_sen = transforms.Compose([transforms.Normalize(mean=trans_mean, std= trans_std)])
    # perform transform
    sen2  = transform_sen(sen2)
    spot6 = transform_spot(spot6)
    return sen2,spot6

# HISTOGRAM MATCHING
def histogram(reference: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Per-channel histogram match of `target` -> `reference`.

    Supports (C,H,W) or (B,C,H,W).  
    If batch size differs, reference with B=1 is broadcast; else paired by batch.  
    Number of channels must be the same for reference and target.

    Returns tensor on target's device/dtype.
    """
    assert target.ndim in (3,4) and reference.ndim in (3,4), "Expected (C,H,W) or (B,C,H,W)"
    device, dtype = target.device, target.dtype

    # normalize to BCHW
    ref = reference.unsqueeze(0) if reference.ndim == 3 else reference
    tgt = target.unsqueeze(0) if target.ndim == 3 else target

    B_ref, C_ref, H_ref, W_ref = ref.shape
    B_tgt, C_tgt, H_tgt, W_tgt = tgt.shape
    assert C_ref == C_tgt, f"channel mismatch: reference={C_ref}, target={C_tgt}"

    # resize reference to target spatial size (bilinear, no align_corners)
    if (H_ref, W_ref) != (H_tgt, W_tgt):
        ref = F.interpolate(ref.to(dtype=torch.float32), size=(H_tgt, W_tgt), mode="bilinear", align_corners=False)

    # numpy buffers
    ref_np = ref.detach().cpu().numpy()
    tgt_np = tgt.detach().cpu().numpy()
    out_np = np.empty_like(tgt_np)

    for b in range(B_tgt):
        rb = b % B_ref  # broadcast if reference has B=1
        for c in range(C_tgt):
            ref_ch = ref_np[rb, c]
            tgt_ch = tgt_np[b, c]

            mask = np.isfinite(tgt_ch) & np.isfinite(ref_ch)
            if mask.any():
                matched = exposure.match_histograms(tgt_ch[mask], ref_ch[mask])
                out = tgt_ch.copy()
                out[mask] = matched
                out_np[b, c] = out
            else:
                out_np[b, c] = tgt_ch

    out = torch.from_numpy(out_np).to(device=device, dtype=dtype)
    return out[0] if target.ndim == 3 else out



# MOMENT MATCHING
def moment(sen2,spot6,sen2_amount=None):   
    sen2,spot6 = sen2.numpy(),spot6.numpy()
    c = 0
    for channel_sen,channel_spot in zip(sen2,spot6):
        c +=1
        #calculate stats
        sen2_mean   = np.mean(channel_sen)
        spot6_mean  = np.mean(channel_spot)
        sen2_stdev  = np.std(channel_sen)
        spot6_stdev = np.std(channel_spot)

        # calculate moment per channel
        channel_result = (((channel_spot - spot6_mean) / spot6_stdev) * sen2_stdev) + sen2_mean

        # stack channels to single array
        if c==1:
            spot6 = channel_result
        else:
            spot6 = np.dstack((spot6,channel_result))
        # transpose back to Cx..

    spot6 = torch.Tensor(spot6.transpose((2,0,1)))   
    return spot6 


"""
SPATIAL MATCHING METHODS
"""
def grid(lr=None,hr=None):
    return(hr)
def shiftnet(lr=None,hr=None):
    return(hr)

"""
Define Identity
"""
def identity(lr,hr):
    return(lr)