import torch
import numpy as np
import pandas as pd
from torch.distributions import MultivariateNormal
from configs.base import device
import time
from typing import Union, Dict, Optional, Tuple

from src.heston_model import sim_heston_paths

def mc_option_prices(
        S0: torch.Tensor,
        V0: torch.Tensor,
        r: torch.Tensor,
        kappa: torch.Tensor,
        theta: torch.Tensor,
        sigma_v: torch.Tensor,
        rho: torch.Tensor,
        T: torch.Tensor,
        strike: torch.Tensor,
        option_types: torch.Tensor,  # 1=call, 0=put [batch_size]
        num_paths: int = 10000,
        num_steps: int = 252,
        antithetic: bool = False
) -> (torch.Tensor, torch.Tensor):
    """
    蒙特卡洛模拟计算Heston模型下的期权价格（复用generate_stock_paths函数）
    输入：Heston模型参数 + 期权参数
    内部逻辑：调用路径生成函数 → 计算期权到期收益 → 折现得到期权价格
    返回：
        option_price: [batch_size] 期权价格（现值，市场公允价）
        stock_paths: [batch_size, num_paths, num_steps+1] 生成的股价路径
    """
    # ===================== 1. 输入预处理（统一设备 + 维度校验） =====================
    # 确保期权参数移到目标设备并转为1D Tensor
    # strike = strike.to(device).squeeze()  # [batch_size]
    # option_types = option_types.to(device).squeeze().to(torch.float32)  # 转为float保证广播兼容


    # 校验输入维度一致性
    batch_size = S0.shape[0]
    assert all([
        V0.shape[0] == batch_size,
        r.shape[0] == batch_size,
        kappa.shape[0] == batch_size,
        theta.shape[0] == batch_size,
        sigma_v.shape[0] == batch_size,
        rho.shape[0] == batch_size,
        T.shape[0] == batch_size,
        strike.shape[0] == batch_size,
        option_types.shape[0] == batch_size
    ]), "所有输入参数的batch_size必须一致"

    # ===================== 2. 复用路径生成函数生成股价路径 =====================
    stock_paths, log_returns, V = sim_heston_paths(
        S0=S0, V0=V0, mu=r, kappa=kappa, theta=theta,
        sigma_v=sigma_v, rho=rho, T=T,
        num_paths=num_paths, num_steps=num_steps,
        # antithetic=antithetic
    )

    # ===================== 3. 计算期权价格（蒙特卡洛核心） =====================
    # 提取到期日股价
    S_T = stock_paths[:, :, -1]  # [batch_size, num_paths]

    # 扩展维度用于广播（匹配路径维度）
    strike = strike.unsqueeze(1)  # [batch_size, 1]
    option_types = option_types.unsqueeze(1)  # [batch_size, 1]
    r = r.to(device).squeeze()  # 确保r是1D Tensor [batch_size]

    # 条件：option_types=1（看涨）→ 取max(S_T-strike, 0)；否则取max(strike-S_T, 0)
    payoff = torch.where(
        option_types == 1,  # 条件：是否为看涨期权 [batch_size, 1]
        torch.clamp(S_T - strike, min=0.0),  # 看涨收益
        torch.clamp(strike - S_T, min=0.0)   # 看跌收益
    )  #  [batch_size, num_paths]

    # 计算到期收益期望（NPV）并折现得到期权价格
    option_npv = payoff.mean(dim=1)  # [batch_size]（对路径求平均）
    # torch.exp(-r * T.to(device).squeeze())  # [batch_size] 折现因子
    option_price = option_npv * torch.exp(-r * T.to(device).squeeze())  # [batch_size] 最终期权价格

    return option_price, stock_paths

@torch.no_grad()
def mc_bump_delta(
    S0: torch.Tensor,
    V0: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    kappa: torch.Tensor,
    theta: torch.Tensor,
    sigma_v: torch.Tensor,
    rho: torch.Tensor,
    option_types: torch.Tensor,
    *,
    num_paths: int = 20000,
    num_steps: int = 252,
    rel_eps: float = 1e-3,
):
    """
    Central bump Delta using Monte Carlo pricing.
    Much noisier than FFT, but consistent with path simulation.
    """

    eps = rel_eps * S0

    price_up, _ = mc_option_prices(
        S0=S0 + eps,
        V0=V0,
        r=r,
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho,
        T=T,
        strike=K,
        option_types=option_types,
        num_paths=num_paths,
        num_steps=num_steps,
    )

    price_dn, _ = mc_option_prices(
        S0=S0 - eps,
        V0=V0,
        r=r,
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho,
        T=T,
        strike=K,
        option_types=option_types,
        num_paths=num_paths,
        num_steps=num_steps,
    )

    delta = (price_up - price_dn) / (2.0 * eps)
    return delta