from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.metrics as km

from ...data.utils import Normalizer

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
    Composite generator content loss that self-instantiates the configured perceptual metric.
    total = l1_w*L1 + sam_w*SAM + perc_w*Perceptual + tv_w*TV
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

        self.max_val  = float(_cfg_get(cfg, ["Training","Losses","max_val"], 1.0))
        self.ssim_win = int(_cfg_get(cfg, ["Training","Losses","ssim_win"], 11))

        self.randomize_bands = bool(_cfg_get(cfg, ["Training","Losses","randomize_bands"], True))
        fixed_idx = _cfg_get(cfg, ["Training","Losses","fixed_idx"], None)
        if fixed_idx is not None:
            fixed_idx = torch.as_tensor(fixed_idx, dtype=torch.long)
            assert fixed_idx.numel() == 3, "fixed_idx must have length 3"
        self.register_buffer("fixed_idx", fixed_idx if fixed_idx is not None else None, persistent=False)

        # --- configure perceptual metric ---
        self.perc_metric = str(_cfg_get(cfg, ["Training", "Losses", "perceptual_metric"], "vgg")).lower()

        if self.perc_metric == "vgg":
            from .vgg import TruncatedVGG19

            i = int(_cfg_get(cfg, ["TruncatedVGG", "i"], 5))
            j = int(_cfg_get(cfg, ["TruncatedVGG", "j"], 4))
            self.perceptual_model = TruncatedVGG19(i=i, j=j)
        elif self.perc_metric == "lpips":
            import lpips

            self.perceptual_model = lpips.LPIPS(net="alex")
        else:
            raise ValueError(f"Unsupported perceptual metric: {self.perc_metric}")

        for p in self.perceptual_model.parameters():
            p.requires_grad = False
        self.perceptual_model.eval()

        # Shared normalizer for computing evaluation metrics
        self.normalizer = Normalizer(cfg)

    # ---------- public API ----------
    def return_loss(
        self, sr: torch.Tensor, hr: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the weighted generator loss and return accompanying raw metrics."""
        comps = self._compute_components(sr, hr, build_graph=True)
        loss = (
            self.l1_w   * comps["l1"] +
            self.sam_w  * comps["sam"] +
            self.perc_w * comps["perceptual"] +
            self.tv_w   * comps["tv"]
        )
        metrics = {k: v.detach() for k, v in comps.items()}
        return loss, metrics

    @torch.no_grad()
    def return_metrics(self, sr: torch.Tensor, hr: torch.Tensor, prefix: str = "") -> dict[str, torch.Tensor]:
        """
        Compute all unweighted metric components and (optionally) prefix their keys.

        Args:
            sr, hr: tensors in the same range as the generator output/HR targets.
            prefix: key prefix like 'train/' or 'val'. If non-empty and doesn't end
                    with '/', a '/' is added automatically.

        Returns:
            dict mapping metric names -> tensors (detached), e.g. {'train/l1': ...}.
            Includes raw PSNR/SSIM metrics computed on stretched/clipped inputs.
        """
        comps = self._compute_components(sr, hr, build_graph=False)
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

    def _perceptual_distance(self, sr_3: torch.Tensor, hr_3: torch.Tensor, *, build_graph: bool) -> torch.Tensor:
        requires_grad = build_graph and self.perc_w != 0.0

        if self.perc_metric == "vgg":
            if requires_grad:
                sr_features = self.perceptual_model(sr_3)
            else:
                with torch.no_grad():
                    sr_features = self.perceptual_model(sr_3)
            with torch.no_grad():
                hr_features = self.perceptual_model(hr_3)
            distance = F.mse_loss(sr_features, hr_features)
        elif self.perc_metric == "lpips":
            sr_norm = sr_3.mul(2.0).sub(1.0)
            hr_norm = hr_3.mul(2.0).sub(1.0).detach()
            if requires_grad:
                distance = self.perceptual_model(sr_norm, hr_norm)
            else:
                with torch.no_grad():
                    distance = self.perceptual_model(sr_norm, hr_norm)
            distance = distance.mean()
        else:
            raise RuntimeError(f"Unhandled perceptual metric: {self.perc_metric}")

        if not requires_grad:
            distance = distance.detach()
        return distance

    def _compute_components(
        self, sr: torch.Tensor, hr: torch.Tensor, *, build_graph: bool
    ) -> dict[str, torch.Tensor]:
        comps: dict[str, torch.Tensor] = {}

        def _compute(weight: float, fn) -> torch.Tensor:
            requires_grad = build_graph and weight != 0.0
            if requires_grad:
                return fn()
            with torch.no_grad():
                return fn().detach()

        # Core reconstruction metrics (always unweighted)
        comps["l1"] = _compute(self.l1_w, lambda: F.l1_loss(sr, hr))
        comps["sam"] = _compute(self.sam_w, lambda: self._sam_loss(sr, hr))

        # Perceptual distance on 3 selected bands
        sr_3 = self._pick_rgb(sr)
        hr_3 = self._pick_rgb(hr)
        comps["perceptual"] = self._perceptual_distance(sr_3, hr_3, build_graph=build_graph)

        # Total variation
        comps["tv"] = _compute(self.tv_w, lambda: self._tv_loss(sr))

        # --- Quality metrics ---
        with torch.no_grad():
            sr_metric = self.normalizer.normalize(sr)
            hr_metric = self.normalizer.normalize(hr)
            psnr = km.psnr(sr_metric, hr_metric, max_val=self.max_val)
            ssim = km.ssim(
                sr_metric,
                hr_metric,
                window_size=self.ssim_win,
                max_val=self.max_val,
            )

            if psnr.dim() > 0:
                psnr = psnr.mean()
            if ssim.dim() > 0:
                ssim = ssim.mean()

            comps["psnr"] = psnr.detach()
            comps["ssim"] = ssim.detach()

        return comps


if __name__ == "__main__":
    # simple test
    from omegaconf import OmegaConf

    config_path = Path(__file__).resolve().parents[2] / "configs" / "config_20m.yaml"
    with open(config_path, "r") as f:
        cfg = OmegaConf.load(f)

    loss_fn = GeneratorContentLoss(cfg)

    B, C, H, W = 2, 13, 64, 64
    sr = torch.rand(B, C, H, W)
    hr = torch.rand(B, C, H, W)

    loss, metrics = loss_fn.return_loss(sr, hr)
    print("Loss:", loss.item())
    print("Metrics:", {k: v.item() for k, v in metrics.items()})

    eval_metrics = loss_fn.return_metrics(sr, hr, prefix="val")
    print("Eval metrics:", {k: v.item() for k, v in eval_metrics.items()})