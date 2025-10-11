import torch

def sen2_stretch(im):
    im = im*(10/3.)
    return(im)

def minmax_percentile(tensor, pmin=2, pmax=98):
    """ Min-max normalisation based on percentiles to reduce outlier effects """
    min_val = torch.quantile(tensor, pmin/100.)
    max_val = torch.quantile(tensor, pmax/100.)
    tensor = (tensor - min_val) / (max_val - min_val)
    return tensor