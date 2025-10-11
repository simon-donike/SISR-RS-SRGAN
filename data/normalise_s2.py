import torch

def normalise_s2(im,stage="norm"):
    assert stage in ["norm","denorm"]
    value = 3.
    if stage == "norm":
        im = im*(10./value)
        im = (im*2)-1
        im = torch.clamp(im,-1,1)
    if stage=="denorm":
        im = (im+1)/2
        im = im*(value/10.)
        im = torch.clamp(im,0,1)
    return(im)

def normalise_10k(im,stage="norm"):
    assert stage in ["norm","denorm"]
    if stage == "norm":
        im = (im/10000.)
        im = torch.clamp(im,0,1)
    if stage=="denorm":
        im = im*10000.
        im = torch.clamp(im,0,10000)
    return(im)
