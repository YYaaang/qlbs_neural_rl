import torch
from configs.specifications.market_bs_cfg import MarketCFG
from src.data_processing import make_state_t
from src.options import payoff_vanilla
from src.rl_models import ModelsTargGroup

#%%
@torch.no_grad()
def generate_mean_q_function(
        cfg: MarketCFG,
        t_step: int,
        S:torch.Tensor,
        S_over_K_tensor: torch.Tensor,  # [ds, m_dim]
        tau_t_all_backward: torch.Tensor, # tau
        expm1_dlr:torch.Tensor,  # [ds]
        a_grid_t0: torch.Tensor,  # [m_dim, A_dim]
        #
        all_risk_lambda_k: torch.Tensor,
        #
        targ_group: ModelsTargGroup,
):
    q0_t1 = None
    q_t1 = None
    if t_step > 2:
        # cal Pi+1
        s_k = make_state_t(
            S_over_K_tensor.clamp_min(1e-12),
            tau_t_all_backward,
        )  # [ds, m, 2]
        a_tp1 = targ_group.actor_targ.mean(s_k)  # [ds, m]
        q0_t1 = targ_group.critic0_targ(
            s_k,
            a_tp1)  # [ds, m]
        # Pi = - Q
        pi_t1 = - q0_t1
        q_t1 = targ_group.critic_targ(
            s_k,
            a_tp1,
        )
    else:
        pi_t1 = payoff_vanilla(S_T=S_over_K_tensor, K=1, option_type=cfg.option_type)
        a_tp1 = torch.zeros_like(pi_t1)

    pi_hat_squared_expansion = reward_pi_hat_squared_expansion_term(
        a_grid_t0, pi_t1, expm1_dlr, cfg.gamma_const
    )  # 惩罚项 [m_dim, A_dim]
    # transaction_cost 费用
    tc = cal_transaction_cost(
        cfg, S_over_K_tensor / S[:,:1], a_tp1, a_grid_t0, expm1_dlr)
    # R mean
    R_grid_mean = build_reward_mean(
        a_grid_t0, expm1_dlr, pi_hat_squared_expansion, all_risk_lambda_k, tc
    )  # [L, m_dim, A_dim]

    # Q-function
    if t_step > 2:
        if all_risk_lambda_k.numel() != 1:
            q_tp1 = torch.cat([
                q0_t1.unsqueeze(1),
                q_t1.reshape(q0_t1.shape[0], -1, q0_t1.shape[1])
            ], dim=1)  # [ds, L, m_dim]
        else:
            q_tp1 = q_t1.reshape(q0_t1.shape[0], -1, q0_t1.shape[1])
    else:
        var_pi = (pi_t1 - pi_t1.mean(dim=0, keepdim=True)).pow(2).mean(dim=0)  # [m]
        Q_T = -pi_t1.unsqueeze(1) - var_pi.view(1, 1, -1) * all_risk_lambda_k.view(1, -1, 1)  # [ds, L, m_dim]
        q_tp1 = Q_T

    Y_grid = R_grid_mean + cfg.gamma_const * q_tp1.mean(dim=0).unsqueeze(-1)
    return Y_grid  # Y_grid.cpu().detach().numpy()

#%%
@torch.no_grad()

def vectorized_pi_tp1(device, torch_dtype,
                      pi_T: torch.Tensor,           # [ds, m]
                      a_seq: torch.Tensor,          # [ds, L, m]
                      y_seq: torch.Tensor,          # [ds, L]
                      gamma_const: torch.Tensor):   # 标量 γ
    ds, dt, m, L = a_seq.shape

    w = gamma_const ** torch.arange(dt, device=device, dtype=torch_dtype)    # [dt]，w[i]=gamma^i, i=0..dt-1
    sum_term = (a_seq * y_seq.unsqueeze(-1).unsqueeze(-1) * w.view(1, dt, 1, 1)).sum(dim=1)  # [ds, m, L]
    return gamma_const ** dt * pi_T.unsqueeze(-1) - sum_term    # [ds, m, L]

#%%
@torch.no_grad()

def reward_pi_hat_squared_expansion_term(
        a_grid_t0: torch.Tensor,  # [ds, A, m, L]
        pi_tp1: torch.Tensor,
        dlr_t0: torch.Tensor,
        gamma_const: torch.Tensor,
        eps_var: float = 1e-12,
):
    # pi_tp1: [ds, m] ; y_t0: [ds]
    pi_mean = pi_tp1.mean(dim=0)                         # [m]
    dlr_mean  = dlr_t0.mean()                                # 标量

    pi_hat = pi_tp1 - pi_mean.unsqueeze(0)               # [ds, m]
    dlr_hat  = dlr_t0 - dlr_mean                               # [ds]

    E_var_pi = (pi_hat.pow(2)).mean(dim=0)               # [m]
    E_cov_py = (pi_hat * dlr_hat.view(-1,1)).mean(dim=0)   # [m]
    E_var_dlr  = dlr_hat.pow(2).mean().clamp_min(eps_var)    # 标量（或保留为 float）

    # 惩罚项（B 形态）
    quad = (
            ((gamma_const ** 2) * E_var_pi).unsqueeze(1)
            - 2.0 * gamma_const * a_grid_t0 * E_cov_py.unsqueeze(1)
            + (a_grid_t0 ** 2) * E_var_dlr
    )  # [m_dim, A_dim]
    return quad

#%%
@torch.no_grad()
def build_reward_mean(a_grid_t0: torch.Tensor,  # [m_dim, A_dim]
                      y_t0: torch.Tensor,  # [ds]
                      pi_hat_squared_expansion: torch.Tensor,  # 惩罚项
                      risk_lambda: torch.Tensor,  # [L]
                      tc: torch.Tensor = None,
                      ):

    reward = (
            a_grid_t0 * y_t0.view(-1,1,1,1)                                 # 线性项
            - risk_lambda.view(1,-1,1,1) * pi_hat_squared_expansion   # 惩罚项
    )    # [ds, L, m, A]
    if tc is not None:
        reward -= tc
    return reward.mean(dim=0)

#%%
@torch.no_grad()
def cal_transaction_cost(
        cfg,
        one_over_K_grid: torch.Tensor,
        a_tp1,
        a_t,
        expm1_dlr,
):
    return cfg.transaction_cost_rate * (torch.abs(
        (a_tp1 * cfg.gamma_const).view(*a_tp1.shape,1)
        - a_t * (expm1_dlr + 1).view(-1,1,1)
    ) * one_over_K_grid.unsqueeze(-1)).unsqueeze(1)  # [ds, m_dim, A_dim]
    # return cfg.transaction_cost_rate * (torch.abs(
    #     (a_tp1 * cfg.gamma_const).view(*a_tp1.shape,1)
    #     - a_t * (expm1_dlr + 1).view(-1,1,1)
    # ) * 0.01).unsqueeze(1)  # [ds, m_dim, A_dim]

#%%
def build_target_terminal(
        R_grid: torch.Tensor,       # [ds, A, m, L]
        pi_T: torch.Tensor,       # [ds, m]
        gamma_const: torch.Tensor,  # 标量 γ
        risk_lambda: torch.Tensor   # [L]
):
    """
    """
    # ---- Var_ds[Π_{t+1}]：仅在 ds 上去中心求方差 → [m]
    var_pi = (pi_T - pi_T.mean(dim=0, keepdim=True)).pow(2).mean(dim=0)  # [m]

    # Q_T^(λ): [ds, m, L]
    Q_T = -pi_T.unsqueeze(-1) - var_pi.view(1, -1, 1) * risk_lambda.view(1, 1, -1)

    # ---- 返回 Y_grid: [ds, A, m, L]
    return R_grid + gamma_const * Q_T.unsqueeze(1)

#%%
