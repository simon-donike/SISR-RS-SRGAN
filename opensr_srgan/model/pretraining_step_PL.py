import torch

def pretraining_training_step(self, *, lr_imgs, hr_imgs, sr_imgs, optimizer_idx):
    """
    Pretraining branch logic (no forward here):
    - G step (optimizer_idx==1): content-only loss
    - D step (optimizer_idx==0): dummy loss + zeroed logs so closures run
    """

    # ======================================================================
    # SECTION: Generator (G) pretraining step
    # Purpose: train only the generator using content loss, no adversarial part.
    # ======================================================================
    if optimizer_idx == 1:
        content_loss, metrics = self.content_loss_criterion.return_loss(sr_imgs, hr_imgs)  # compute perceptual/content loss (e.g., VGG or L1)
        self._log_generator_content_loss(content_loss)                             # log content loss for G (consistent args)
        for key, value in metrics.items():
            self.log(f"train_metrics/{key}", value,sync_dist=True)                               # reuse computed metrics for logging

        # Ensure adversarial weight is logged even when not used during pretraining
        adv_weight = self._compute_adv_loss_weight()
        self._log_adv_loss_weight(adv_weight)
        return content_loss                                                        # return loss for optimizer step (G only)

    # ======================================================================
    # SECTION: Discriminator (D) pretraining step
    # Purpose: no real training — just log zeros and return dummy loss to satisfy closure.
    # ======================================================================
    elif optimizer_idx == 0:
        device, dtype = hr_imgs.device, hr_imgs.dtype                              # get tensor device and dtype for consistency
        zero = torch.tensor(0.0, device=device, dtype=dtype)                       # define reusable zero tensor

        # --- Log dummy discriminator "opinions" (always zero during pretrain) ---
        self.log("discriminator/D(y)_prob",    zero, prog_bar=True,  sync_dist=True)  # fake real-prob (always 0)
        self.log("discriminator/D(G(x))_prob", zero, prog_bar=True,  sync_dist=True)  # fake fake-prob (always 0)

        # --- Create dummy scalar loss (ensures PL closure runs) ---
        dummy = torch.zeros((), device=device, dtype=dtype, requires_grad=True)    # dummy value with grad for optimizer compatibility
        self.log("discriminator/adversarial_loss", dummy, sync_dist=True)          # log dummy adversarial loss (always 0)
        return dummy                                                               # return dummy loss (keeps Lightning loop intact)
