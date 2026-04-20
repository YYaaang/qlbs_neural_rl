# configs/specs/__init__.py
# 导出所有配置结构类
from .market_bs_cfg import MarketCFG
from .actor_cfg import ActorCFG
from .critic_cfg import CriticCFG

# 可选：定义__all__，规范导出范围
__all__ = ["MarketCFG", "ActorCFG", "CriticCFG"]