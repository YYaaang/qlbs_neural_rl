from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any


@dataclass(frozen=True)
class ActorCFG:
    """
    Model architecture / function-class configuration.

    定义：
    - actor / critic0 / critic 使用什么函数族来近似
    - 不改变 MarketCFG（问题本身）
    - 但决定是否可以复用已有模型权重
    """

    # =================================================
    # identity
    # =================================================
    level: str = 'actor'  # must be "actor"

    risk_lambda: float = 10000.0
    # =================================================
    # Actor architecture
    # =================================================
    state_dim: int = 2
    hidden: Tuple[int, ...] =(128, 128)
    a_min: float = -2.0
    a_max: float = 2.0

    # 一般不需要更改
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    min_sigma: float = 1e-4

    # =================================================
    # serialization
    # =================================================
    def to_dict(self) -> Dict[str, Any]:
        """
        严格序列化，用于写入 cfg.json
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ActorCFG":
        """
        严格反序列化 + 结构校验
        """
        if d.get("level") != "actor":
            raise ValueError("ModelCFG.level must be 'actor'")
        return cls(**d)