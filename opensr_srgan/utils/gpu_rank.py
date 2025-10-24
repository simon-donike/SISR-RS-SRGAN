import os


def _is_global_zero() -> bool:
    """Return True if this process is the global rank zero (primary) worker.

    Used to gate singleton side effects (e.g., logging, file writes, directory
    creation) in both single- and multi-process environments. Supports plain CPU,
    single-GPU, torch.distributed (``torchrun``), and SLURM-style launches.

    Detection order:
        1. Use ``torch.distributed`` if available and initialized.
        2. Fall back to environment variables ``RANK`` and ``WORLD_SIZE``.
        3. Default to True for non-distributed (single-process) runs.

    Returns:
        bool: True if this process is rank zero or if distributed training
        is not active (single process). False otherwise.
    """
    # Prefer torch.distributed if available
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass

    # Fallback to env vars commonly set by torchrun/SLURM
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank == 0 or world_size == 1
