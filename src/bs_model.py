import torch
from configs.specifications.market_bs_cfg import MarketCFG

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

torch_dtype = torch.float32 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else torch.float

#%%
class BSConditionalSampler:
    """
    Helper class for QLBS (BS model):
    - Pre-simulate a reference BS path
    - Provide conditional one-step distributions S_{t+1} | S_t
    """

    def __init__(
        self,
        market_cfg: MarketCFG,
        ref_paths: int,
        device=device,
        torch_dtype=torch_dtype,
    ):
        self.model = "bs"
        self.cfg = market_cfg
        self.device = device
        self.torch_dtype = torch_dtype

        self.S0 = market_cfg.S0
        self.T_steps = market_cfg.T_steps

        # cache params
        self.mu = market_cfg.mu
        self.sigma = market_cfg.sigma
        self.dt = market_cfg.dt

        # ------------------------------------------------------------
        # 1. generate reference path(s)
        # ------------------------------------------------------------
        S, log_returns, _ = self.sim_paths(
            N_paths=ref_paths,
            T_steps=market_cfg.T_steps,
        )
        self.S_path = S
        self.log_returns = log_returns
        self.V_path = _

    # ---------------------------------------------------------------------
    # Conditional one-step simulation
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def get_existing_next_paths(
            self,
            t: int,
            N_paths: int = 20_000,
    ):
        return self.S_path[:, t: t + 2], self.log_returns[:, t: t + 1]


    @torch.no_grad()
    def sim_next_path(
            self,
            t: int = None,
            N_paths: int = 20_000,
    ):
        S_full_scaled, log_returns, _ = sim_bs_paths(
            S0=self.S0, mu=self.mu, sigma=self.sigma,
            dt=self.dt, N_paths=N_paths,
            T_steps=1,  # simulate ONE step ahead
            anchor_days=0,
            device=device,
            torch_dtype=torch_dtype
        )
        return S_full_scaled, log_returns

    @torch.no_grad()
    def sim_paths(
            self,
            N_paths: int,
            T_steps: int,
            V_new = None,
            existing_path=None,
    ):
        S_full_scaled, log_returns, _ = sim_bs_paths(
            S0=self.S0, mu=self.mu, sigma=self.sigma,
            dt=self.dt, N_paths=N_paths,
            T_steps=T_steps,  # simulate ONE step ahead
            anchor_days=0,
            device=device,
            torch_dtype=torch_dtype
        )
        return S_full_scaled, log_returns, _
#%%
@torch.no_grad()
def sim_bs_paths_cfg(
        s_cfg: MarketCFG,
        N_paths: int = 10000,
        T_steps: int = 252,
        anchor_days: int = 0,  # “前 anchor_days 天”作为历史窗
        device=device,
        torch_dtype=torch_dtype
):
    #
    S0 = s_cfg.S0
    mu = s_cfg.mu
    sigma = s_cfg.sigma
    dt = s_cfg.dt

    return sim_bs_paths(
        S0=S0, mu=mu, sigma=sigma,
        dt=dt, N_paths=N_paths,
        T_steps=T_steps,
        anchor_days=anchor_days,
        device=device,
        torch_dtype=torch_dtype
    )

@torch.no_grad()
def sim_bs_paths(
        S0: float,
        mu: float,
        sigma: float,
        dt: torch.Tensor,  # 允许传入 tensor（与原调用一致）
        N_paths: int = 10000,
        T_steps: int = 252,
        anchor_days: int = 0,  # “前 anchor_days 天”作为历史窗
        device = device,
        torch_dtype = torch_dtype
) -> (torch.Tensor, torch.Tensor):
    """shape=[N_paths, T_steps+1]"""

    log_returns = d_log_S(mu, sigma, dt, N_paths, T_steps, anchor_days)
    cum_log_ret = torch.cat(
        [torch.zeros((N_paths, 1), device=device, dtype=torch_dtype),
         torch.cumsum(log_returns, dim=1)],
        dim=1)
    S_paths = S0 * torch.exp(cum_log_ret)

    S_anchor = S_paths[:, anchor_days]  # [N_paths]
    scale = (S0 / S_anchor).unsqueeze(1)  # [N_paths, 1]
    S_full_scaled = S_paths * scale

    return S_full_scaled, log_returns, None

#%%
@torch.no_grad()
def d_log_S(
        mu: float,
        sigma: float,
        dt: torch.Tensor,  # 允许传入 tensor（与原调用一致）
        N_paths: int = 10000,
        T_steps: int = 252,
        anchor_days: int = 0,  # “前 anchor_days 天”作为历史窗
):
    # z = torch.normal(0.0, 1.0, size=(N_paths, T_steps + anchor_days), device=device)
    return ((mu - 0.5 * sigma ** 2) * dt + sigma * torch.sqrt(
        dt
    ) *
            torch.normal(
                0.0, 1.0, size=(N_paths, T_steps + anchor_days),
                device=device,
                dtype=torch_dtype)
            )

import numpy as np
import py_vollib_vectorized as pvv

def cal_bs_prices(option_type, S_mat, K, tau_vec, r_f, sigma):
    """
    S_mat: shape [N_paths, T_steps]
    tau_vec: shape [T_steps]
    返回 shape [N_paths, T_steps]
    """

    N, T = S_mat.shape

    # 展平
    S_flat = S_mat.reshape(-1)
    tau_flat = np.tile(tau_vec, reps=N)

    bs_price_flat = pvv.vectorized_black_scholes(
        option_type,
        S_flat,
        K,
        tau_flat,
        r_f,
        sigma,
        return_as='numpy'
    )

    return bs_price_flat.reshape(N, T)


def cal_bs_delta(option_type, S_mat, K, tau_vec, r_f, sigma):
    """
    同 cal_bs_prices，一维展开调用 Vollib 再 reshape 回来
    """

    N, T = S_mat.shape

    S_flat = S_mat.reshape(-1)
    tau_flat = np.tile(tau_vec, reps=N)

    bs_delta_flat = pvv.vectorized_delta(
        option_type,
        S_flat,
        K,
        tau_flat,
        r_f,
        sigma,
        return_as='numpy'
    )

    return bs_delta_flat.reshape(N, T)
