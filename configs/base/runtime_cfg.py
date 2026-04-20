# configs/runtime_cfg.py

import torch
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class RuntimeCFG:
    """
    Runtime / training-time configuration.
    """

    # =================================================
    # Data collection / batching
    # =================================================
    N_ds:int = 20_000
    num_total_updates:int = 2
    collect_data_times: int = 3
    replay_reset_each_t: bool = True

    # =================================================
    # Logging / debugging
    # =================================================
    s_over_k_steps: int = 50
    a_grid_size: int = 31

    # =================================================
    # Logging / debugging
    # =================================================
    print_debug: bool = True
    log_every: int = 20          # iterations
    profile: bool = False

    # =================================================
    # Optimizer: actor
    # =================================================
    lr_actor: float = 1e-3
    max_grad_norm_actor: float = 1.0

    # =================================================
    # Optimizer: critic0
    # =================================================
    lr_critic0: float = 1e-2
    max_grad_norm_critic0: float = 10.0

    # =================================================
    # Optimizer: critic (lambda > 0)
    # =================================================
    lr_critic: float = 1e-2
    max_grad_norm_critic: float = 10.0

    # =================================================
    # Scheduler (ReduceLROnPlateau)
    # =================================================
    sched_factor: float = 0.5
    sched_patience: int = 5
    sched_threshold: float = 1e-5
    sched_threshold_mode: str = "abs"
    sched_cooldown: int = 2
    sched_min_lr: float = 5e-4

    # =================================================
    # Training loop control
    # =================================================
    max_iters_critic0: int = 512
    min_iters_critic0: int = 100

    max_iters_critic: int = 512
    min_iters_critic: int = 100

    max_iters_actor: int = 512
    min_iters_actor: int = 100

    sched_every:int = 1

    # =================================================
    # Early-stopping / stability (TrainState)
    # =================================================
    ema_beta: float = 0.9
    stable_abs: float = 5e-3
    stable_rel: float = 0.02
    stable_patience: int = 6
    warmup_ignore: int = 0
    bias_correction: bool = False

    max_grad_norm = 10.0

    # =================================================
    # Evaluation
    # =================================================
    eval_n_paths: int = 500


    # =================================================
    # serialization
    # =================================================
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize RuntimeCFG to JSON-compatible dict.

        - device / torch_dtype are intentionally excluded
        """
        d = asdict(self)
        d.pop("device", None)
        d.pop("torch_dtype", None)
        return d

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        # *,
        # device: torch.device,
        # torch_dtype: torch.dtype,
    ) -> "RuntimeCFG":
        """
        Rebuild RuntimeCFG from serialized dict,
        injecting runtime-only device / dtype.
        """
        return cls(
            # device=device,
            # torch_dtype=torch_dtype,
            **d,
        )
