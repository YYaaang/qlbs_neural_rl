import math
import torch
import torch.distributions as D

@torch.no_grad()
def make_S_over_K(
        S: torch.Tensor,
        S0: float,
        device, dtype,
        s_over_k_steps=40,
        ranges=((0.9, 1.1), (0.6, 1.4), (0.3, 1.7)),
        base_m=(0.2, 0.25, 1.75, 1.8),
):
    # S: [ds, T+1]

    m_base = _make_S_over_K_with_jitter(
        device=device, dtype=dtype,
        ranges=ranges,
        q_steps=s_over_k_steps - len(base_m),
    )
    base_m = torch.tensor(base_m, device=device, dtype=dtype)
    base_m += torch.rand_like(base_m) * 0.1 - 0.05

    m_base = torch.cat([base_m, m_base], dim=0)

    data =  (S / S0).unsqueeze(2) * m_base  # [ds, T+1, m]

    data.clamp(0.1, 3)

    return data  # data[0,0].cpu().numpy()

@torch.no_grad()
def _make_S_over_K_with_jitter(
        device, dtype,
        ranges,
        q_steps=7,
):
    """
    """

    total_random = q_steps
    R = len(ranges)

    # 每个 range 分配 roughly 相同数量
    count = total_random // R
    extra = total_random % R  # 如果无法整除，给前 extra 个区间多1个样本

    samples = []

    for i, (lo, hi) in enumerate(ranges):
        lo = max(0.01, lo)
        hi = min(2.01, hi)
        n_i = count + (1 if i < extra else 0)
        if i == len(ranges) - 1:
            # uniform_random_with_noise
            x = _uniform_random(
                ranges[-1][0], ranges[-1][1], n_i, device=device, dtype=dtype)
        else:
            # Normal random in [lo, hi]
            x = torch.randn(n_i, device=device, dtype=dtype)
            x = lo + (hi - lo) * torch.sigmoid(x)  # sigmoid 压缩到 (0,1)
        samples.append(x)
    # 合并随机点
    m_all = torch.cat(samples, dim=0)

    # 排序（不改变数量）
    # m_all = torch.sort(m_all).values

    return m_all

#%%
@torch.no_grad()
def build_action_grid_random(
        device,
        torch_dtype,
        batch_size: int,
        a_grid_size: int = 31,
        base_actions=(-1, -0.5, 0, 0.5, 1),
        a_min: float = -2.0,
        a_max: float = 2.0,
):
    """
    构造动作网格：
    - base_actions 固定保留
    - 剩余动作在 [a_min, a_max] 内完全随机
    - 随机后排序
    """

    # base actions
    base_actions = torch.tensor(base_actions, device=device, dtype=torch_dtype)
    B = base_actions.numel()

    A_reg = max(a_grid_size - B, 0)

    # 随机部分
    if A_reg > 0:
        a_reg = torch.empty(
            batch_size,
            A_reg,
            device=device,
            dtype=torch_dtype
        ).uniform_(a_min, a_max)

        # 顺序随机点
        a_reg, _ = torch.sort(a_reg, dim=1)

        a_base = base_actions.view(1, -1).expand(batch_size, -1)
        a_all = torch.cat([a_base, a_reg], dim=1)
    else:
        a_all = base_actions.view(1, -1).expand(batch_size, -1)

    return a_all

@torch.no_grad()
def build_action_grid_from_actor(
        device,
        torch_dtype,
        actor,
        s_t_base: torch.Tensor,
        a_grid_size: int = 31,
        base_actions=(-1, -0.5, 0, 0.5, 1),
        std_noise_factor: float = 0.05,
        min_action_span_a: float = 0.4,   # a-space 最小跨度
        min_sigma: float = 1e-3,
):
    """
    构造动作网格（总量 a_grid_size）
    """

    s_over_k_steps, state_dim = s_t_base.shape

    # ----------------------------------------------------------------------
    # 0) base actions（tuple -> tensor），无须任何 None 判断
    # ----------------------------------------------------------------------
    base_actions = torch.tensor(base_actions, device=device, dtype=torch_dtype)
    B = base_actions.numel()
    A_reg = max(a_grid_size - B, 0)        # regular grid 可用数量

    # ----------------------------------------------------------------------
    # 1) actor 分布：μ、σ
    # ----------------------------------------------------------------------
    mu, log_std = actor(s_t_base)
    std = log_std.exp().clamp_min(min_sigma)

    # ----------------------------------------------------------------------
    # 2) regular Gaussian quantile grid（若 A_reg > 0）
    # ----------------------------------------------------------------------
    if A_reg > 0:
        # 等概率分位点
        p = torch.linspace(1e-3, 1 - 1e-3, A_reg, device=device, dtype=torch_dtype)
        z = torch.erfinv(2 * p - 1) * math.sqrt(2)

        # 添加小扰动
        z = z + torch.randn_like(z) * (std_noise_factor * float(std.median()))
        # u, a
        u= mu.unsqueeze(1) + std.unsqueeze(1) * z
        a_reg = actor._squash_to_bounds(torch.tanh(u))

        # ------------------------------------------------------------------
        # 3) enforce a-space minimal action span
        # ------------------------------------------------------------------
        a_range = float(a_reg.max() - a_reg.min())
        if a_range < min_action_span_a:
            scale = min_action_span_a / (a_range + 1e-12)
            z = z * scale
            u = mu.unsqueeze(1) + std.unsqueeze(1) * z
            a_reg = actor._squash_to_bounds(torch.tanh(u))

    else:
        a_reg = None

    # ----------------------------------------------------------------------
    # 4) base actions expand 到完整维度 [ds, B, m, L]
    # ----------------------------------------------------------------------
    if B > 0:
        a_base = base_actions.view(1, -1).expand(mu.numel(), -1)
    else:
        a_base = None

    # ----------------------------------------------------------------------
    # 5) 合并 & 裁剪到 a_grid_size
    # ----------------------------------------------------------------------
    if (a_reg is not None) and (a_base is not None):
        a_all = torch.cat([a_base, a_reg], dim=1)
    else:
        a_all = a_reg if a_reg is not None else a_base

    return a_all

#%%
# ##########
@torch.no_grad()
def make_state_t(
        S_over_K_t: torch.Tensor,
        tau_scalar: torch.Tensor,
) -> torch.Tensor:

    # 1) log(S/K)
    log_m = torch.log(S_over_K_t.clamp_min(1e-12))

    x_tanh = torch.tanh(log_m / 0.35)

    # 2) tau : 标量 -> [...]
    tau = tau_scalar.expand(S_over_K_t.shape)

    # 堆叠到最后一维（特征维）: [..., 2]
    s_t = torch.stack([log_m, x_tanh, tau], dim=-1)  # [..., 2]
    return s_t

@torch.no_grad()
def make_state_t_past(
        S_over_K_t: torch.Tensor,
        tau_scalar: torch.Tensor,
        lambda_perK: torch.Tensor,
) -> torch.Tensor:

    ds, n_t, m = S_over_K_t.shape
    L = lambda_perK.shape[0]

    # 1) log(S/K) : [ds, m] -> [ds, n_t, m, L]
    log_m = torch.log(S_over_K_t.clamp_min(1e-12)).unsqueeze(-1).expand(-1, -1, -1, L)

    # 2) tau : 标量 -> [ds, n_t, m, L]
    tau = tau_scalar.view(1,-1,1,1).expand(ds, n_t, m, L)

    # 3) lambda_perK : [L] -> [ds, n_t, m, L]
    lam = lambda_perK.expand(ds, n_t, m, L)

    # 堆叠到最后一维（特征维）: [..., 3]
    s_t = torch.stack([log_m, tau, lam], dim=-1)  # [ds, n_t, m, L, 3]
    return s_t
#%%
@torch.no_grad()
def pack_state_action(s_t_base: torch.Tensor, a_grid_t: torch.Tensor):
    # s_t_base: [ds, m, 2]; a_grid_t: [ds, A, m]
    ds, m, L, sd = s_t_base.shape
    A = a_grid_t.shape[1]
    S_input = s_t_base.unsqueeze(1).expand(ds, A, m, L, sd)      # [ds, A, m, L, 3]
    A_input = a_grid_t                                        # [ds, A, m]（你的 critic 支持网格前向）
    return S_input, A_input

#%%
@torch.no_grad()
def _uniform_random(min_val, max_val, length, device, dtype):

    step = (max_val - min_val) / (length - 1)

    x = torch.linspace(
        max(0, min_val - step / 2),
        max_val + step / 2,
        length,
        device = device, dtype = dtype
    )

    perturb = (torch.rand_like(x, device=device, dtype=dtype) - 0.5) * step  # 等价于 ±noise_half*2
    x = x + perturb

    # 4. 防止首尾超出范围
    x = torch.clamp(x, min_val, max_val)

    return x
