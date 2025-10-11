import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.metrics as km

def _cfg_get(cfg, keys, default=None):
    cur = cfg
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, None)
        else:
            cur = getattr(cur, k, None)
    return default if cur is None else cur

class GeneratorContentLoss(nn.Module):
    """
    Composite generator content loss that self-instantiates TruncatedVGG19 from cfg.TruncatedVGG.
    total = l1_w*L1 + sam_w*SAM + perc_w*MSE(VGG3) + tv_w*TV + psnr_w*(-PSNR) + ssim_w*(1-SSIM)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # --- weights & settings from config ---
        # (fallback to deprecated Training.perceptual_loss_weight if needed)
        self.l1_w   = float(_cfg_get(cfg, ["Training","Losses","l1_weight"], 1.0))
        self.sam_w  = float(_cfg_get(cfg, ["Training","Losses","sam_weight"], 0.05))
        perc_w_cfg  = _cfg_get(cfg, ["Training","Losses","perceptual_weight"],
                               _cfg_get(cfg, ["Training","perceptual_loss_weight"], 0.1))
        self.perc_w = float(perc_w_cfg)
        self.tv_w   = float(_cfg_get(cfg, ["Training","Losses","tv_weight"], 0.0))
        self.psnr_w = float(_cfg_get(cfg, ["Training","Losses","psnr_weight"], 0.0))
        self.ssim_w = float(_cfg_get(cfg, ["Training","Losses","ssim_weight"], 0.0))

        self.max_val  = float(_cfg_get(cfg, ["Training","Losses","max_val"], 1.0))
        self.ssim_win = int(_cfg_get(cfg, ["Training","Losses","ssim_win"], 11))

        self.randomize_bands = bool(_cfg_get(cfg, ["Training","Losses","randomize_bands"], True))
        fixed_idx = _cfg_get(cfg, ["Training","Losses","fixed_idx"], None)
        if fixed_idx is not None:
            fixed_idx = torch.as_tensor(fixed_idx, dtype=torch.long)
            assert fixed_idx.numel() == 3, "fixed_idx must have length 3"
        self.register_buffer("fixed_idx", fixed_idx if fixed_idx is not None else None, persistent=False)

        # --- instantiate & freeze VGG encoder from config ---
        from model.model_blocks import TruncatedVGG19
        i = int(_cfg_get(cfg, ["TruncatedVGG","i"], 5))
        j = int(_cfg_get(cfg, ["TruncatedVGG","j"], 4))
        self.truncated_vgg19 = TruncatedVGG19(i=i, j=j)
        for p in self.truncated_vgg19.parameters():
            p.requires_grad = False
        self.truncated_vgg19.eval()  # acts as fixed feature extractor

    # ---------- public API ----------
    def return_loss(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        comps = self._compute_components(sr, hr)
        return (
            self.l1_w   * comps["l1"] +
            self.sam_w  * comps["sam"] +
            self.perc_w * comps["perceptual"] +
            self.tv_w   * comps["tv"] +
            self.psnr_w * comps["psnr_loss"] +
            self.ssim_w * comps["ssim_loss"]
        )

    @torch.no_grad()
    def return_metrics(self, sr: torch.Tensor, hr: torch.Tensor, prefix: str = "") -> dict[str, torch.Tensor]:
        """
        Compute all unweighted metric components and (optionally) prefix their keys.

        Args:
            sr, hr: tensors in the same range as configured for PSNR/SSIM.
            prefix: key prefix like 'train/' or 'val'. If non-empty and doesn't end
                    with '/', a '/' is added automatically.

        Returns:
            dict mapping metric names -> tensors (detached), e.g. {'train/l1': ...}
        """
        comps = self._compute_components(sr, hr)
        p = (prefix + "/") if prefix and not prefix.endswith("/") else prefix
        return {f"{p}{k}": v.detach() for k, v in comps.items()}

    # ---------- internals ----------
    @staticmethod
    def _tv_loss(x: torch.Tensor) -> torch.Tensor:
        dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return dh + dw

    @staticmethod
    def _sam_loss(sr: torch.Tensor, hr: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        B, C, H, W = sr.shape
        sr_f = sr.view(B, C, -1)
        hr_f = hr.view(B, C, -1)
        dot  = (sr_f * hr_f).sum(dim=1)
        sr_n = sr_f.norm(dim=1).clamp_min(eps)
        hr_n = hr_f.norm(dim=1).clamp_min(eps)
        cos  = (dot / (sr_n * hr_n)).clamp(-1 + 1e-7, 1 - 1e-7)
        ang  = torch.acos(cos)
        return ang.mean()

    def _pick_rgb(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if C == 3:
            return x
        if self.randomize_bands:
            idx = torch.randperm(C, device=x.device)[:3]
        else:
            assert self.fixed_idx is not None, "Provide fixed_idx or enable randomize_bands"
            idx = self.fixed_idx.to(device=x.device)
        return x[:, idx, :, :]

    def _compute_components(self, sr: torch.Tensor, hr: torch.Tensor) -> dict:
        device = sr.device
        comps = {
            "l1": torch.tensor(0.0, device=device),
            "sam": torch.tensor(0.0, device=device),
            "perceptual": torch.tensor(0.0, device=device),
            "tv": torch.tensor(0.0, device=device),
            "psnr_loss": torch.tensor(0.0, device=device),  # -PSNR
            "ssim_loss": torch.tensor(0.0, device=device),  # 1 - SSIM
        }

        # L1
        if self.l1_w != 0.0:
            comps["l1"] = F.l1_loss(sr, hr)

        # SAM
        if self.sam_w != 0.0:
            comps["sam"] = self._sam_loss(sr, hr)

        # Perceptual (VGG on 3 bands)
        if self.perc_w != 0.0:
            sr_3 = self._pick_rgb(sr); hr_3 = self._pick_rgb(hr)
            sr_v = self.truncated_vgg19(sr_3)
            with torch.no_grad():
                hr_v = self.truncated_vgg19(hr_3)
            comps["perceptual"] = F.mse_loss(sr_v, hr_v)

        # TV
        if self.tv_w != 0.0:
            comps["tv"] = self._tv_loss(sr)

        # PSNR as loss = -PSNR
        if self.psnr_w != 0.0:
            psnr = km.psnr(sr, hr, max_val=self.max_val)
            if psnr.dim() > 0:
                psnr = psnr.mean()
            comps["psnr_loss"] = -psnr

        # SSIM as loss = 1 - SSIM
        if self.ssim_w != 0.0:
            ssim = km.ssim(sr, hr, window_size=self.ssim_win, max_val=self.max_val, reduction="mean")
            comps["ssim_loss"] = 1.0 - ssim

        return comps
