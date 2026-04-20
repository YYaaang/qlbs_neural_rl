from typing import Dict, Any, Tuple
import torch

class MarketCFG:
    def __init__(
            self, device, torch_dtype,
            # ---------- identity ----------
            level: str,  # must be "market"

            # ---------- market ----------
            S0: float,
            V0: float,
            mu: float,
            kappa:float,
            theta:float,
            sigma:float,
            rho:float,


            r: float,
            T: float,
            K: float,
            T_steps: int,
            #
            transaction_cost_rate: float,
            #
            option_type = 'p',
            #
            frozen = True,
    ):
        self.model =  "heston"
        # 校验 level 必须为 "market"
        if level != "market":
            raise ValueError("MarketCFG.level must be 'market'")

        self.device = device
        self.torch_dtype = torch_dtype
        # 初始化所有属性
        self.level = level
        #

        self.S0 = S0       # Initial stock price
        self.V0 = V0        # Initial volatility
        self.kappa = kappa      # Mean reversion speed of volatility
        self.theta = theta     # Long-term mean of volatility
        self.sigma = sigma      # Volatility of volatility
        self.rho = rho       # Correlation between price and volatility
        self.mu = mu
        self.T = T          # Option maturity (years)

        self.r = r         # Risk-free rate
        self.K = K        # Option strike price

        self.T_steps = T_steps

        #
        self.transaction_cost_rate = transaction_cost_rate
        self.option_type = option_type
        #
        self.dt = torch.tensor(T / T_steps, dtype=torch_dtype, device=device)  # 时间步长
        self.gamma_const = torch.exp(torch.tensor(-r * (T / T_steps), dtype=torch_dtype, device=device))  # 常数γ

        # 使实例不可变（模拟 dataclass(frozen=True)）
        self._frozen = frozen


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
            # ---------- identity ----------
            "level": self.level,
            "model": self.model,
            # ---------- heston dynamics ----------
            "S0": self.S0,
            "V0": self.V0,
            "mu": self.mu,
            "kappa": self.kappa,
            "theta": self.theta,
            "sigma": self.sigma,
            "rho": self.rho,
            "r": self.r,
            "T": self.T,
            "T_steps": self.T_steps,

            # ---------- option / trading ----------
            "K": self.K,
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
            # runtime context
            device=device,
            torch_dtype=torch_dtype,

            # identity
            level=d["level"],
            # heston dynamics
            S0=d["S0"],
            V0=d["V0"],
            mu=d["mu"],
            kappa=d["kappa"],
            theta=d["theta"],
            sigma=d["sigma"],
            rho=d["rho"],
            r=d["r"],
            T=d["T"],
            T_steps=d["T_steps"],

            # option / trading
            K=d["K"],
            transaction_cost_rate=d["transaction_cost_rate"],
            option_type=d.get("option_type", "p"),
        )

