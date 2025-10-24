# inference.py

import os
from pathlib import Path

import torch

from .model.SRGAN import SRGAN_model


def load_model(config_path=None, ckpt_path=None, device=None):
    """Build SRGAN model and (optionally) load weights. Safe to call from tests."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SRGAN_model(config_file_path=config_path).eval().to(device)

    if ckpt_path:
        # Try Lightning API first (without 'strict'); fall back to raw state_dict
        try:
            model = (
                SRGAN_model.load_from_checkpoint(ckpt_path, map_location=device)
                .eval()
                .to(device)
            )
        except TypeError:
            state = torch.load(ckpt_path, map_location=device)
            state = state.get("state_dict", state)
            model.load_state_dict(state, strict=False)

    return model, device


def run_sen2_inference(
    sen2_path=None,
    config_path=None,
    ckpt_path=None,
    gpus=None,
    window_size=(128, 128),
    factor=4,
    overlap=12,
    eliminate_border_px=2,
    save_preview=False,
    debug=False,
):
    """Run Sentinel-2 SR inference. Kept out of import-time for CI."""
    if gpus is not None and len(gpus) > 0:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(map(str, gpus)))

    model, device = load_model(config_path=config_path, ckpt_path=ckpt_path)

    import opensr_utils

    sr_object = opensr_utils.large_file_processing(
        root=sen2_path,
        model=model,
        window_size=window_size,
        factor=factor,
        overlap=overlap,
        eliminate_border_px=eliminate_border_px,
        device=device,
        gpus=gpus if gpus is not None else ([0] if device == "cuda" else []),
        save_preview=save_preview,
        debug=debug,
    )
    sr_object.start_super_resolution()
    return sr_object


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    # ---- Define placeholders ----
    sen2_path = Path(__file__).resolve().parent / "data" / "S2A_MSIL2A_EXAMPLE.SAFE"
    config_path = Path(__file__).resolve().parent / "configs" / "config_20m.yaml"
    ckpt_path = "checkpoints/srgan-20m-6band/last.ckpt"
    gpus = [0]

    run_sen2_inference(
        sen2_path=str(sen2_path),
        config_path=str(config_path),
        ckpt_path=ckpt_path,
        gpus=gpus,
    )


if __name__ == "__main__":
    main()
