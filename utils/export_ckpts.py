import torch
from pathlib import Path

def filter_generator_ckpt(ckpt_path, output_path=None, keep_ema=True):
    """
    Filter a PyTorch Lightning checkpoint to keep only the generator weights
    (and optionally the EMA state), discarding optimizer, scheduler, and other metadata.

    This is useful for exporting a trained SRGAN generator for inference or fine-tuning
    without carrying unnecessary training state.

    Parameters
    ----------
    ckpt_path : str or Path
        Path to the input .ckpt file (Lightning checkpoint).
    output_path : str or Path, optional
        Destination path for the filtered checkpoint. If not provided, a new file
        with the suffix "_genonly.ckpt" will be created next to the original.
    keep_ema : bool, default=True
        Whether to retain the Exponential Moving Average (EMA) state if it exists.

    Behavior
    --------
    - Filters `ckpt["state_dict"]` to keep only entries containing "generator"
      (and "ema" if `keep_ema=True`).
    - Removes training artifacts such as optimizers, schedulers, and callbacks.
    - Saves the cleaned checkpoint to disk.

    Example
    -------
    >>> filter_generator_ckpt("logs/aa_best_models/RGB-NIR_4band.ckpt")
    Saved filtered checkpoint → logs/aa_best_models/RGB-NIR_4band_genonly.ckpt

    >>> filter_generator_ckpt("SWIR_6band.ckpt", keep_ema=False)
    Saved filtered checkpoint → SWIR_6band_genonly.ckpt
    """

    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # --- filter inside the state_dict ---
    if "state_dict" in ckpt:
        ckpt["state_dict"] = {
            k: v for k, v in ckpt["state_dict"].items()
            if "generator" in k or (keep_ema and "ema" in k)
        }

    # --- optionally keep EMA state if present ---
    if not keep_ema and "ema_state" in ckpt:
        ckpt.pop("ema_state")

    out = output_path or ckpt_path.with_name(ckpt_path.stem + "_genonly.ckpt")
    torch.save(ckpt, out)
    print(f"Saved filtered checkpoint → {out}")

# Example usage:
filter_generator_ckpt("logs/aa_best_models/RGB-NIR_4band.ckpt", "logs/aa_best_models/RGB-NIR_4band_inference.ckpt", keep_ema=False)
filter_generator_ckpt("logs/aa_best_models/SWIR_6band.ckpt","logs/aa_best_models/SWIR_6band_inference.ckpt", keep_ema=True)

