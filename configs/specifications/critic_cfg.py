from dataclasses import dataclass
from typing import Dict, Any, Tuple
import torch

@dataclass(frozen=True)
class CriticCFG:
    # Identity
    level: str
    form: str

    # Device & dtype
    # device: torch.device  # critic
    # torch_dtype: torch.dtype  # linear, quadratic

    risk_lambda: float

    # State
    state_dim: int = 2

    # Network
    hidden: Tuple[int, ...] = (128, 128)
    head_dim: int = 64

    # 一般不需要更改
    eps_curv = 1e-8

    def __post_init__(self):
        if self.level != "critic":
            raise ValueError("CriticCFG.level must be 'critic'")

    # ===============================
    # serialization
    # ===============================
    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "form": self.form,
            # "device": self.device,
            # "torch_dtype": self.torch_dtype,
            "risk_lambda": self.risk_lambda,
            "state_dim": self.state_dim,
            "hidden": self.hidden,
            "head_dim": self.head_dim,
        }


    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CriticCFG":
        return cls(**d)