import torch
from opensr_srgan.model.pretraining_step_PL import pretraining_training_step

def training_step_PL1x(self, batch, batch_idx, optimizer_idx):
    # -------- CREATE SR DATA --------
    lr_imgs, hr_imgs = batch                                  # unpack LR/HR tensors from dataloader batch
    sr_imgs = self.forward(lr_imgs)                          # forward pass of the generator to produce SR from LR

    # ======================================================================
    # SECTION: Pretraining phase gate
    # Purpose: decide if we are in the content-only pretrain stage.
    # ======================================================================

    # -------- DETERMINE PRETRAINING --------
    pretrain_phase = self._pretrain_check()                  # check schedule: True => content-only pretraining
    if optimizer_idx == 1:  # log whether pretraining is active or not
        self.log("training/pretrain_phase", float(pretrain_phase), prog_bar=False,sync_dist=True)  # log once per G step to track phase state

    # ======================================================================
    # SECTION: Pretraining branch (delegated)
    # Purpose: during pretrain, only content loss for G and dummy logging for D.
    # ======================================================================

    # -------- IF PRETRAIN: delegate --------
    if pretrain_phase:
        # run pretrain step separately and return loss here
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
            return dummy           
    # -------- END PRETRAIN --------

    # ======================================================================
    # SECTION: Adversarial training — Discriminator step
    # Purpose: update D to distinguish HR (real) vs SR (fake).
    # ======================================================================

    # -------- Normal Train: Discriminator Step  --------
    if optimizer_idx==0:
        # run discriminator and get loss between pred labels and true labels
        hr_discriminated = self.discriminator(hr_imgs)       # D(real): logits for HR images
        sr_discriminated = self.discriminator(sr_imgs.detach()) # detach so G doesn’t get gradients from D’s step

        # targets
        real_target = torch.full_like(hr_discriminated, self.adv_target) # get labels/fuzzy labels
        fake_target = torch.zeros_like(sr_discriminated) # zeros, since generative prediction

        # Binary Cross-Entropy loss
        loss_real = self.adversarial_loss_criterion(hr_discriminated, real_target)   # BCEWithLogitsLoss for D(G(x))
        loss_fake = self.adversarial_loss_criterion(sr_discriminated, fake_target)  # BCEWithLogitsLoss for D(y)
        adversarial_loss = loss_real + loss_fake # Sum up losses
        self.log("discriminator/adversarial_loss",adversarial_loss,sync_dist=True) # log weighted loss

        # [LOG-B] Always log D opinions: real probs in normal training
        with torch.no_grad():
            d_real_prob = torch.sigmoid(hr_discriminated).mean()   # estimate mean real probability
            d_fake_prob = torch.sigmoid(sr_discriminated).mean()   # estimate mean fake probability
        self.log("discriminator/D(y)_prob", d_real_prob, prog_bar=True,sync_dist=True)      # log D(real) confidence
        self.log("discriminator/D(G(x))_prob", d_fake_prob, prog_bar=True,sync_dist=True)   # log D(fake) confidence

        # return weighted discriminator loss
        return adversarial_loss                                # PL will use this to step the D optimizer

    # ======================================================================
    # SECTION: Adversarial training — Generator step
    # Purpose: update G to minimize content loss + (weighted) adversarial loss.
    # ======================================================================

    # -------- Normal Train: Generator Step  --------
    if optimizer_idx==1:

        """ 1. Get VGG space loss """
        # encode images
        content_loss, metrics = self.content_loss_criterion.return_loss(sr_imgs, hr_imgs)   # perceptual/content criterion (e.g., VGG)
        self._log_generator_content_loss(content_loss)                             # log content loss for G (consistent args)
        for key, value in metrics.items():
            self.log(f"train_metrics/{key}", value,sync_dist=True)                             # log detailed metrics without extra forward passes


        """ 2. Get Discriminator Opinion and loss """
        # run discriminator and get loss between pred labels and true labels
        sr_discriminated = self.discriminator(sr_imgs)                             # D(SR): logits for generator outputs
        adversarial_loss = self.adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated)) # keep taargets 1.0 for G loss
        self.log("generator/adversarial_loss",adversarial_loss,sync_dist=True)     # log unweighted adversarial loss

        """ 3. Weight the losses"""
        adv_weight = self._adv_loss_weight() # get adversarial weight based on current step
        adversarial_loss_weighted = (adversarial_loss * adv_weight) # weight adversarial loss
        total_loss = content_loss + adversarial_loss_weighted # total content loss
        self.log("generator/total_loss",total_loss,sync_dist=True)                # log combined objective (content + λ_adv * adv)

        # return Generator loss
        return total_loss         
    
    
def training_step_PL2x(self, *_, **__):
    """Placeholder for PyTorch Lightning >= 2.0."""

    raise NotImplementedError(
        "PyTorch Lightning >= 2.0 support for training_step is not implemented yet."
    )