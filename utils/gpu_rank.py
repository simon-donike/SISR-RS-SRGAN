import os

def _is_global_zero() -> bool:
    """
    True if this process should perform singleton side effects (create dirs, write files).
    Works with plain CPU, single-GPU, and torch.distributed (torchrun/SLURM).
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