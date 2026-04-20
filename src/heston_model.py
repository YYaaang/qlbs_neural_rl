import torch

from configs.specifications.market_heston_cfg import MarketCFG

from configs.base.env_cfg import device, torch_dtype

#%%

class HestonConditionalSampler:
    """
    Helper class for QLBS:
    - Pre-simulate a reference Heston path
    - Provide conditional one-step distributions S_{t+1} | S_t, V_t
    """

    def __init__(
        self,
        market_cfg: MarketCFG,
        ref_paths: int = 1,
        device=device,
    ):
        """
        Parameters
        ----------
        market_cfg : MarketCFG
            Physical-measure Heston config
        ref_paths : int
            Number of reference paths (usually 1 is enough)
        """
        self.model = "heston"

        self.cfg = market_cfg
        self.device = device

        # ------------------------------------------------------------
        # 1. generate reference path(s)
        # ------------------------------------------------------------
        S, log_returns, V = sim_heston_paths_cfg(
            s_cfg=market_cfg,
            N_paths=ref_paths,
            T_steps=market_cfg.T_steps,
            device=device,
        )

        # keep only trajectories (t dimension)
        # S, V : [1, P, T+1] or [B, P, T+1]
        self.S0 = market_cfg.S0
        self.V0 = market_cfg.V0
        self.S_path = S.squeeze(0) if S.dim() == 3 else S
        self.V_path = V.squeeze(0) if V.dim() == 3 else V
        self.log_returns = log_returns.squeeze(0) if V.dim() == 3 else V


        # cache model params (for speed / clarity)
        self.mu = market_cfg.mu
        self.kappa = market_cfg.kappa
        self.theta = market_cfg.theta
        self.sigma_v = market_cfg.sigma
        self.rho = market_cfg.rho

        ################
        # import matplotlib.pyplot as plt
        # S_path, _, _ = self.sim_paths(10, 50, self.V0)
        # plt.figure(figsize=(10, 6))
        # paths = S_path.cpu().numpy()
        # plt.plot(paths[:100].T, linewidth=0.7, alpha=0.6)  # 自动画100条
        # plt.show()

    # ---------------------------------------------------------------------
    # Conditional one-step simulation
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def get_existing_next_paths(
            self,
            t: int,
            N_paths: int = 20_000,
    ):
        return self.S_path[:,t : t+2], self.log_returns[:,t : t+1]

    @torch.no_grad()
    def sim_next_path(
            self,
            t: int,
            N_paths: int = 20_000,
    ):
        S_t = torch.tensor([self.S0], device=self.device)
        V_t = self.V_path[:, t]

        S, log_returns, _ = sim_heston_paths(
            S0=S_t,
            V0=V_t,
            mu=torch.tensor([self.mu], device=self.device),
            kappa=torch.tensor([self.kappa], device=self.device),
            theta=torch.tensor([self.theta], device=self.device),
            sigma_v=torch.tensor([self.sigma_v], device=self.device),
            rho=torch.tensor([self.rho], device=self.device),
            T=torch.tensor([1.0], device=self.device),   # one-step horizon
            num_paths=N_paths,
            num_steps=1,  # simulate ONE step ahead
            device=self.device,
        )

        return S[0], log_returns[0]

    @torch.no_grad()
    def sim_paths(
            self,
            N_paths: int,
            T_steps: int,
            V_new: float,
            existing_path=False,
    ):
        if existing_path:
            indices = torch.randperm(self.S_path.shape[0])[:N_paths]
            return self.S_path[indices], self.log_returns[indices], self.V_path[indices]

        S, log_returns, V = sim_heston_paths(
            S0=torch.tensor([self.S0], device=self.device),
            V0=torch.tensor([V_new], device=self.device),
            mu=torch.tensor([self.mu], device=self.device),
            kappa=torch.tensor([self.kappa], device=self.device),
            theta=torch.tensor([self.theta], device=self.device),
            sigma_v=torch.tensor([self.sigma_v], device=self.device),
            rho=torch.tensor([self.rho], device=self.device),
            T=torch.tensor([1.0], device=self.device),   # one-step horizon
            num_paths=N_paths,
            num_steps=T_steps,  # simulate steps
            device=self.device,
        )

        return S[0], log_returns[0], V[0]

#%%
@torch.no_grad()
def sim_heston_paths_cfg(
        s_cfg: MarketCFG,
        N_paths: int = 10000,
        T_steps: int = 252,
        device=device,
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    physical-measure Heston path simulation via MarketCFG
    """

    S0 = s_cfg.S0
    V0 = s_cfg.V0
    mu = s_cfg.mu                  # physical drift
    kappa = s_cfg.kappa
    theta = s_cfg.theta
    sigma_v = s_cfg.sigma          # vol-of-vol
    rho = s_cfg.rho
    T = s_cfg.T

    def _to_1d_tensor(x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        if x.dim() == 0:
            x = x.unsqueeze(0)
        return x

    S0 = _to_1d_tensor(S0)
    V0 = _to_1d_tensor(V0)
    mu = _to_1d_tensor(mu)
    kappa = _to_1d_tensor(kappa)
    theta = _to_1d_tensor(theta)
    sigma_v = _to_1d_tensor(sigma_v)
    rho = _to_1d_tensor(rho)
    T = _to_1d_tensor(T)

    return sim_heston_paths(
        S0=S0,
        V0=V0,
        mu=mu,
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho,
        T=T,
        num_paths=N_paths,
        num_steps=T_steps,
        device=device,
    )

@torch.no_grad()
def sim_heston_paths(
        S0: torch.Tensor,
        V0: torch.Tensor,
        mu: torch.Tensor,
        kappa: torch.Tensor,
        theta: torch.Tensor,
        sigma_v: torch.Tensor,
        rho: torch.Tensor,
        T: torch.Tensor,
        num_paths: int = 10000,
        num_steps: int = 252,
        device: torch.device = torch.device('cpu')
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    """
    # 确保输入是1D Tensor并移到目标设备

    batch_size = S0.shape[0]

    # 计算时间步长（扩展为[batch_size, 1, 1]，匹配后续维度）
    dt = (T / num_steps).unsqueeze(1)  # [batch_size, 1]
    sqrt_dt = torch.sqrt(dt)  # [batch_size, 1]

    # 扩展所有参数到[batch_size, 1]，确保广播匹配
    mu = mu.unsqueeze(1)  # [batch_size, 1]
    kappa = kappa.unsqueeze(1)  # [batch_size, 1]
    theta = theta.unsqueeze(1)  # [batch_size, 1]
    sigma_v = sigma_v.unsqueeze(1)  # [batch_size, 1]
    rho = rho.unsqueeze(1)  # [batch_size, 1]

    # 1. 生成所有随机数（[batch_size, sim_paths, num_steps]）
    Z1 = torch.randn(batch_size, num_paths, num_steps, device=device)  # [B, P, T]
    Z2 = rho.unsqueeze(2) * Z1 + torch.sqrt(
        1 - rho.unsqueeze(2) ** 2) * torch.randn(batch_size, num_paths, num_steps,
                                                 device=device)  # [B, P, T]

    # 3. 初始化路径张量（[batch_size, sim_paths, num_steps+1]）
    S = torch.zeros(batch_size, num_paths, num_steps + 1, device=device)
    V = torch.zeros(batch_size, num_paths, num_steps + 1, device=device)
    log_returns = torch.zeros(
        batch_size, num_paths, num_steps,
        device=device
    )
    # integrated_var = torch.zeros(batch_size, sim_paths, num_steps + 1, device=device)

    # 设置初始值（广播到所有路径）
    S[:, :, 0] = S0.unsqueeze(1)  # [batch_size, sim_paths]
    V[:, :, 0] = V0.unsqueeze(1)  # [batch_size, sim_paths]

    # 4. 向量化时间步循环（内部全批量计算）
    for t in range(1, num_steps + 1):
        # 提取前一步的方差（[batch_size, sim_paths]）
        V_prev = V[:, :, t - 1]  # [B, P]
        # 方差保护（避免负数/零）
        V_prev_clamped = torch.clamp(V_prev, min=1e-8)  # [B, P]
        sqrt_V_prev = torch.sqrt(V_prev_clamped)  # [B, P]

        # --------------------------
        # 向量化更新股票价格（维度匹配）
        # --------------------------
        log_return = (mu - 0.5 * V_prev_clamped) * dt + \
                     sqrt_V_prev * sqrt_dt * Z1[:, :, t - 1]  # [B, P]

        log_returns[:, :, t - 1] = log_return

        # 批量更新股价
        S[:, :, t] = S[:, :, t - 1] * torch.exp(log_return)  # [B, P]

        # --------------------------
        # 向量化更新方差（仅1处改动：合并clamp）
        # --------------------------
        V_update = kappa * (theta - V_prev_clamped) * dt + \
                   sigma_v * sqrt_V_prev * sqrt_dt * Z2[:, :, t - 1]  # [B, P]
        # 原始逻辑：先赋值再clamp → 优化后：合并为一次操作
        V[:, :, t] = torch.clamp(V_prev + V_update, min=1e-8)

    return S, log_returns, V


