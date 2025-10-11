def print_model_summary(self):
    """
    Prints a detailed and visually structured summary of the SRGAN configuration.
    Includes architecture info, resolution scale, training parameters, loss weights, and model sizes.
    """

    # --- helper to count parameters (in millions) ---
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    g_params = count_params(self.generator)
    d_params = count_params(self.discriminator)
    total_params = g_params + d_params

    # ------------------------------------------------------------------
    # Derive human-readable generator description
    # ------------------------------------------------------------------
    g_type = self.config.Generator.model_type
    if g_type == "SRResNet":
        g_desc = "SRResNet (Residual Blocks with BatchNorm)"
    elif g_type == "res":
        g_desc = "flexible_generator (Residual Blocks without BatchNorm)"
    elif g_type == "rcab":
        g_desc = "flexible_generator (RCAB Blocks with Channel Attention)"
    elif g_type == "rrdb":
        g_desc = "flexible_generator (RRDB Dense Residual Blocks)"
    elif g_type == "lka":
        g_desc = "flexible_generator (LKA Large-Kernel Attention Blocks)"
    else:
        g_desc = f"Custom Generator Type: {g_type}"

    # ------------------------------------------------------------------
    # Resolution info (input → output)
    # ------------------------------------------------------------------
    scale_factor = getattr(self.config.Generator, "scaling_factor", getattr(self.generator, "scale", None))
    if scale_factor is not None:
        input_res = 128
        output_res = input_res * scale_factor
        res_str = f"{input_res}×{input_res} → {output_res}×{output_res}  (×{scale_factor})"
    else:
        res_str = "Unknown (scale not specified)"

    # ------------------------------------------------------------------
    # Retrieve loss weights if available
    # ------------------------------------------------------------------
    loss_cfg = getattr(self.config.Training, "Losses", {})
    content_w = getattr(loss_cfg, "content_loss_weight", 1.0)
    adv_w = getattr(loss_cfg, "adv_loss_beta", 1.0)
    perceptual_w = getattr(loss_cfg, "perceptual_loss_weight", None)
    total_w_str = (
        f"   • Content: {content_w} | Adversarial: {adv_w}"
        + (f" | Perceptual: {perceptual_w}" if perceptual_w is not None else "")
    )

    print("\n" + "=" * 90)
    print("🚀  SRGAN Model Summary")
    print("=" * 90)

    # ------------------------------------------------------------------
    # Generator Info
    # ------------------------------------------------------------------
    print(f"🧩 Generator")
    print(f"   • Architecture:      {g_desc}")
    print(f"   • Resolution:        {res_str}")
    print(f"   • Input Channels:    {self.config.Model.in_bands}")
    print(f"   • Feature Channels:  {self.config.Generator.n_channels}")
    print(f"   • Residual Blocks:   {self.config.Generator.n_blocks}")
    print(f"   • Kernel Sizes:      small={self.config.Generator.small_kernel_size}, large={self.config.Generator.large_kernel_size}")
    print(f"   • Params:            {g_params:.2f} M\n")

    # ------------------------------------------------------------------
    # Discriminator Info
    # ------------------------------------------------------------------
    d_type = getattr(self.config.Discriminator, "model_type", "standard")
    d_blocks = getattr(self.config.Discriminator, "n_blocks", None)
    effective_blocks = getattr(self.discriminator, "n_blocks", getattr(self.discriminator, "n_layers", d_blocks))
    base_channels = getattr(self.discriminator, "base_channels", "N/A")
    kernel_size = getattr(self.discriminator, "kernel_size", "N/A")
    fc_size = getattr(self.discriminator, "fc_size", None)

    if d_type == "patchgan":
        d_desc = "PatchGAN"
    else:
        d_desc = "SRGAN"

    print(f"🧠 Discriminator")
    print(f"   • Architecture:     {d_desc}")
    if effective_blocks is not None:
        print(f"   • Blocks/Layers:    {effective_blocks}")
    print(f"   • Base Channels:    {base_channels}")
    print(f"   • Kernel Size:      {kernel_size}")
    if fc_size is not None:
        print(f"   • FC Layer Size:    {fc_size}")
    print(f"   • Params:            {d_params:.2f} M\n")

    # ------------------------------------------------------------------
    # Training Setup
    # ------------------------------------------------------------------
    print(f"⚙️  Training Configuration")
    print(f"   • Pretrain Generator: {self.pretrain_g_only}")
    print(f"   • Pretrain Steps:     {self.g_pretrain_steps}")
    print(f"   • Adv. Ramp Steps:    {self.adv_loss_ramp_steps}")
    print(f"   • Label Smoothing:    {self.adv_target < 1.0}")
    print(f"   • Adv. Target Label:  {self.adv_target}\n")

    # ------------------------------------------------------------------
    # Loss Functions
    # ------------------------------------------------------------------
    print(f"📉 Loss Functions")
    print(f"   • Content Loss:       {type(self.content_loss_criterion).__name__}")
    print(f"   • Adversarial Loss:   {type(self.adversarial_loss_criterion).__name__}")
    print(total_w_str + "\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"📊 Model Summary")
    print(f"   • Total Trainable Params: {total_params:.2f} M")
    print(f"   • Device:                 {self.device if hasattr(self, 'device') else 'Not set'}")
    print(f"   • Config File:            {getattr(self.config, '_metadata', 'YAML file')}")
    print("=" * 90 + "\n")
