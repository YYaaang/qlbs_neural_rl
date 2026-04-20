from typing import Dict, Any, Tuple
import torch

class MarketCFG:
    def __init__(
            self, device, torch_dtype,
            # ---------- identity ----------
            level: str,  # must be "market"

            # ---------- market ----------
            S0: float,
            K: float,
            mu: float,
            r: float,
            sigma: float,
            T: float,
            T_steps: int,
            #
            transaction_cost_rate: float,
            #
            option_type = 'p',
    ):
        self.model = "bs"

        # 校验 level 必须为 "market"
        if level != "market":
            raise ValueError("MarketCFG.level must be 'market'")

        self.device = device
        self.torch_dtype = torch_dtype
        # 初始化所有属性
        self.level = level
        #
        self.S0 = S0
        self.K = K
        self.mu = mu
        self.r = r
        self.sigma = sigma
        self.T = T
        self.T_steps = T_steps

        #
        self.transaction_cost_rate = transaction_cost_rate
        self.option_type = option_type
        #
        self.dt = torch.tensor(T / T_steps, dtype=torch_dtype, device=device)  # 时间步长
        self.gamma_const = torch.exp(torch.tensor(-r * (T / T_steps), dtype=torch_dtype, device=device))  # 常数γ

        # 使实例不可变（模拟 dataclass(frozen=True)）
        self._frozen = True


    def __setattr__(self, name, value):
        # 检查是否已冻结，如果是则不允许修改属性
        if hasattr(self, '_frozen') and self._frozen and name != '_frozen':
            raise AttributeError(f"cannot assign to field '{name}' (MarketCFG is immutable)")
        super().__setattr__(name, value)

    # ===============================
    # serialization
    # ===============================
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "level": self.level,
            "S0": self.S0,
            "K": self.K,
            "mu": self.mu,
            "r": self.r,
            "sigma": self.sigma,
            "T": self.T,
            "T_steps": self.T_steps,
            "transaction_cost_rate": self.transaction_cost_rate,
            "option_type": self.option_type,
        }

    @classmethod
    def from_dict(
            cls,
            d: Dict[str, Any],
            *,
            device: torch.device,
            torch_dtype: torch.dtype,
    ) -> "MarketCFG":
        if d.get("level") != "market":
            raise ValueError("MarketCFG.level must be 'market'")

        return cls(
            device=device,
            torch_dtype=torch_dtype,
            level=d["level"],
            S0=d["S0"],
            K=d["K"],
            mu=d["mu"],
            r=d["r"],
            sigma=d["sigma"],
            T=d["T"],
            T_steps=d["T_steps"],
            transaction_cost_rate=d["transaction_cost_rate"],
            option_type=d["option_type"],
        )
