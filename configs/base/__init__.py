# configs/base/__init__.py
# 导出环境相关核心变量（来自env_cfg.py）
from .env_cfg import device, torch_dtype

# 导出运行时配置类（来自runtime_cfg.py）
from .runtime_cfg import RuntimeCFG

from .policy_cfg import PolicyCFG

# 可选：定义__all__，规范「from configs.base import *」的导出范围（工程化最佳实践）
__all__ = ["device", "torch_dtype", "RuntimeCFG", "PolicyCFG"]