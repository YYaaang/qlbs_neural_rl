import torch
import numpy as np
import pandas as pd
from torch.distributions import MultivariateNormal
from configs.base import device
import time
from typing import Union, Dict, Optional, Tuple

@torch.no_grad()
def fft_option_prices(
        S: torch.Tensor,               # (*state_shape)
        V: torch.Tensor,               # (*state_shape)
        K: torch.Tensor,               # (*state_shape)
        T: torch.Tensor,               # (*state_shape)
        kappa:  Union[float, torch.Tensor],   # (scalar or *state_shape)
        theta:  Union[float, torch.Tensor],   # (scalar or *state_shape)
        sigma:  Union[float, torch.Tensor],   # (scalar or *state_shape)
        rho:    Union[float, torch.Tensor],     # (scalar or *state_shape)
        r:      Union[float, torch.Tensor],       # (scalar or *state_shape)
        option_types: torch.Tensor,    # (*state_shape) 1=Call, 0=Put
        *,
        alpha: float = 1.0,
        eta: float = 0.01,
        n: int = 12,
        psi=None,                      # if provided, must be (N, *state_shape)
        device = device,
        dtype: torch.dtype = torch.float32,
        complex_dtype: torch.dtype = torch.complex64,

) -> torch.Tensor:
    """
    ND-safe Carr–Madan Heston FFT 定价。
    - 频域维固定为 dim=0
    - 状态维为任意多维 (*state_shape)
    - 返回的 price 形状为 (*state_shape)
    """

    # ---- 基础设置 ----
    S = S.to(device=device, dtype=dtype)
    V = V.to(device=device, dtype=dtype)
    K = K.to(device=device, dtype=dtype)
    T = T.to(device=device, dtype=dtype)
    option_types = option_types.to(device=device, dtype=dtype)

    # 形状检查：这些必须与 S 同形
    state_shape = S.shape
    for name, x in dict(V=V, K=K, T=T, option_types=option_types).items():
        if x.shape != state_shape:
            raise ValueError(f"{name}.shape {tuple(x.shape)} must equal S.shape {tuple(state_shape)}")

    # 其他参数可以是标量或与 state_shape 相同；无需强行扩展，PyTorch 会广播。
    # 但为了设备/类型一致性，转换到 device/dtype：
    # def to_real(x):
    #     if torch.is_tensor(x):
    #         return x.to(device=device, dtype=dtype)
    #     return torch.tensor(x, device=device, dtype=dtype)
    #
    # r = to_real(r)
    # kappa = to_real(kappa)
    # theta = to_real(theta)
    # sigma = to_real(sigma)
    # rho   = to_real(rho)

    # ---- FFT 网格 ----
    N = 2 ** n                        # 频域点数
    delta = (2 * np.pi / N) / eta     # log-strike 网格间距
    nuJ = (torch.arange(N, device=device, dtype=dtype) * eta).to(dtype=complex_dtype)  # (N,)

    # ---- 辅助 reshape：频域维 / 状态维 ----
    state_ndim = S.ndim

    def fft_expand(x_1d: torch.Tensor) -> torch.Tensor:
        """(N,) -> (N, 1, 1, ..., 1)"""
        return x_1d.view((x_1d.shape[0],) + (1,) * state_ndim)

    def state_expand(x_state: torch.Tensor) -> torch.Tensor:
        """(*state_shape) -> (1, *state_shape)"""
        return x_state.view((1,) + state_shape)

    # ---- 折现 ----
    # 折现与 T 同形，允许 r 是标量（广播）
    discount_real = torch.exp(-r * T)                           # (*state_shape)
    discount_e = state_expand(discount_real).to(complex_dtype)  # (1, *state_shape)

    # ---- log-strike beta ----
    beta = torch.log(K)                                         # (*state_shape)
    beta_e = state_expand(beta.to(dtype=complex_dtype))         # (1, *state_shape)

    # ---- Trapezoid 权重 ----
    w = eta * torch.ones(N, device=device, dtype=dtype)         # (N,)
    w[0] = eta / 2
    w_e = fft_expand(w).to(complex_dtype)                       # (N, 1, 1, ...)

    # ---- 计算 psi (若未提供) ----
    if psi is None:
        # ψ(u)
        psi = _fft_psi(S, V, T, kappa, theta, sigma, rho, r, nuJ, alpha)
    else:
        # 验证形状（可选）
        if psi.shape[0] != N or psi.shape[1:] != state_shape:
            raise ValueError(f"psi must have shape (N, *state_shape) = {(N,) + state_shape}, got {psi.shape}")
    psi = psi.to(device=device, dtype=complex_dtype)      # (N, *state_shape)

    # ---- 频域核 ----
    nuJ_e = fft_expand(nuJ)                               # (N, 1, 1, ...)
    kernel = torch.exp(-1j * nuJ_e * beta_e)              # (N, *state_shape)

    # ---- 组装被积函数并 FFT ----
    x = kernel * discount_e * psi * w_e                   # (N, *state_shape)
    y = torch.real(torch.fft.fft(x, dim=0))               # (N, *state_shape)

    # ---- 从频域回到价格域 ----
    m = torch.arange(N, device=device, dtype=dtype).view((N,) + (1,) * state_ndim)  # (N,1,1,...)
    km = beta_e.real + delta * m                          # (N, *state_shape)，注意 beta_e 是 complex

    cT_km = torch.exp(-alpha * km) / np.pi * y            # (N, *state_shape)
    call_price = cT_km[0, ...]                            # (*state_shape)

    # Put-Call parity
    put_price = call_price - S + K * discount_real        # (*state_shape)

    price = option_types * call_price + (1.0 - option_types) * put_price
    return price

@torch.no_grad()
def fft_bump_delta(
    S: torch.Tensor,
    V: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: torch.Tensor,
    kappa: torch.Tensor,
    theta: torch.Tensor,
    sigma_v: torch.Tensor,
    rho: torch.Tensor,
    option_types: torch.Tensor,
    *,
    rel_eps: float = 1e-4,
):
    """
    Central bump Delta using FFT pricing.

    All inputs must have the same shape (*state_shape).
    option_types: 1=Call, 0=Put
    """

    # ε = relative bump (scale with S!)
    eps = rel_eps * S

    price_up = fft_option_prices(
        S=S + eps,
        V=V,
        K=K,
        T=T,
        r=r,
        kappa=kappa,
        theta=theta,
        sigma=sigma_v,
        rho=rho,
        option_types=option_types,
    )

    price_dn = fft_option_prices(
        S=S - eps,
        V=V,
        K=K,
        T=T,
        r=r,
        kappa=kappa,
        theta=theta,
        sigma=sigma_v,
        rho=rho,
        option_types=option_types,
    )

    delta = (price_up - price_dn) / (2.0 * eps)
    return delta

#%%
@torch.no_grad()
def _fft_psi(
    S: torch.Tensor,
    V: torch.Tensor,
    T: torch.Tensor,
    kappa: Union[float, torch.Tensor],
    theta: Union[float, torch.Tensor],
    sigma: Union[float, torch.Tensor],
    rho:   Union[float, torch.Tensor],
    r:     Union[float, torch.Tensor],
    nuJ_complex: torch.Tensor,   # (N,)
    alpha: float,
    *,
    complex_dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """
    ND-safe ψ(u) for Carr–Madan transform
    返回形状：(N, *state_shape)
    """

    # Compute characteristic function
    cf = _fft_characteristic_function(
        phi=nuJ_complex - (alpha + 1) * 1j,
        S=S,
        V=V,
        T=T,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
        r=r,
        complex_dtype=complex_dtype,
    )
    state_ndim = V.ndim
    shape = (nuJ_complex.shape[0],) + (1,) * state_ndim
    denom1 = (alpha + 1j * nuJ_complex).reshape(shape)
    denom2 = (alpha + 1 + 1j * nuJ_complex).reshape(shape)

    return cf / (denom1 * denom2)  # (nuJ_complex.shape[0], *state_shape)

# Heston model characteristic function
@torch.no_grad()
def _fft_characteristic_function(
        phi: torch.Tensor,
        S: torch.Tensor,
        V: torch.Tensor,
        T: torch.Tensor,
        kappa: float | torch.Tensor,
        theta: float | torch.Tensor,
        sigma: float | torch.Tensor,
        rho: float | torch.Tensor,
        r: float | torch.Tensor,
        V0: float | torch.Tensor = None,
        *,
        complex_dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """
    ND-safe Heston 特征函数；返回形状 (N, *state_shape)
    频域维固定为 dim=0
    """
    if V0 is not None and V is None:
        V = V0

    state_ndim = S.ndim

    # 扩频域维：phi -> (N, 1, 1, ..., 1)
    phi_expanded = phi.view((phi.shape[0],) + (1,) * state_ndim)

    i = 1j

    temp1 = kappa - rho * sigma * phi_expanded * i
    d = torch.sqrt(temp1**2 + sigma**2 * (phi_expanded**2 + i * phi_expanded))

    power_term = 2 * kappa * theta / (sigma ** 2)

    numerator = (kappa * theta * T * temp1 / (sigma ** 2)
                 + i * phi_expanded * T * r
                 + i * phi_expanded * torch.log(S))

    cosh_term = torch.cosh(d * T / 2)
    sinh_term = torch.sinh(d * T / 2)
    log_denominator = power_term * torch.log(cosh_term + (temp1 / d) * sinh_term)

    temp2 = (phi_expanded**2 + i * phi_expanded) * V / (d / torch.tanh(d * T / 2) + temp1)

    log_phi = numerator - log_denominator - temp2

    return torch.exp(log_phi)