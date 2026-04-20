


import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_processing import make_state_t, make_S_over_K
from src.qlbs import vectorized_pi_tp1
from src.options import payoff_vanilla

from configs.specifications.market_heston_cfg import MarketCFG
from utils.pricing.quantlib_heston_model import QuantLibHestonModel

# ============================================================
# 完整 evaluate_full_paths（含还原）
# ============================================================
@torch.no_grad()
def evaluate_full_paths(
        market_cfg:MarketCFG,
        OPTION_MODEL_TYPE: str,
        stock_path_generater,
        actor,
        critic0,
        critic,
        n_paths=50,
        alpha=0.2,
        existing_path=False,
):
    if OPTION_MODEL_TYPE == 'heston':
        from utils.pricing.quantlib_heston_model import QuantLibHestonModel as EvaluateOptionPricer
        V0 = market_cfg.V0
    elif OPTION_MODEL_TYPE == 'bs':
        V0 = None
        from utils.pricing.pvv_bs_model import PVVBSModel as EvaluateOptionPricer
    else:
        raise NameError('OPTION_MODEL_TYPE is wrong')

    evaluate_option_pricer:QuantLibHestonModel = EvaluateOptionPricer(
        market_cfg=market_cfg,
    )
    device = market_cfg.device
    dt = market_cfg.dt
    T  = market_cfg.T_steps
    K  = market_cfg.K
    r  = market_cfg.r
    sigma = market_cfg.sigma
    option_type = market_cfg.option_type

    # ------------------------------------------------------------
    # 1) 生成 路径
    # ------------------------------------------------------------
    S, log_ret, V = stock_path_generater.sim_paths(
        N_paths=n_paths,
        T_steps=T,
        V_new=V0,
        existing_path=existing_path
    )  # S: [N, T+1]

    dS = S[:, 1:] - S[:, :-1]  # [N,T]

    # ------------------------------------------------------------
    # 2) time-to-maturity τ（倒序）
    # ------------------------------------------------------------
    tau = (market_cfg.T_steps - torch.arange(market_cfg.T_steps, device=device)) * market_cfg.dt
    # tau: [T], 如 [20*dt,19*dt,...,dt]

    # ------------------------------------------------------------
    # 3) 构造状态（注意：模型输入无量纲 S/K）
    # ------------------------------------------------------------
    S_over_K = (S[:, :-1] / K).clamp_min(1e-12)     # [N,T]
    state_actor = make_state_t(S_over_K, tau)       # [N,T,2]

    # ------------------------------------------------------------
    # 4) Actor / Critic 前向（无量纲）
    # ------------------------------------------------------------
    a_mean_hat = actor.mean(state_actor)                  # [N,T]
    a_star_hat = critic.a_star(
        state_actor,
        clamp_bounds=(float(actor.a_min_buf), float(actor.a_max_buf))
    )                                                     # [N,T]

    Q0_mean_hat   = critic0(state_actor, a_mean_hat)         # [N,T]
    Q_mean_hat = critic(state_actor, a_mean_hat)         # [N,T]
    Q_star_hat = critic(state_actor, a_star_hat)         # [N,T]

    # ------------------------------------------------------------
    # 5) 无量纲 → 有量纲还原
    # ------------------------------------------------------------
    # 动作： u_t = û_t * (K / S_t)
    S_t = S[:, :-1]          # [N,T]

    scale = (K / S_t)        # [N,T]

    a_mean_real = a_mean_hat * scale      # [N,T]
    a_star_real = a_star_hat * scale      # [N,T]

    # Q：Q_real = K * Q_hat
    Q0_mean   = K * Q0_mean_hat
    Q_mean = K * Q_mean_hat
    Q_star = K * Q_star_hat

    # ------------------------------------------------------------
    # 6) Black–Scholes baseline（2D）
    # ------------------------------------------------------------
    S_mid_np = S_t.detach().cpu().numpy()      # [N,T]
    tau_np   = tau.detach().cpu().numpy()      # [T]
    V_mid_np = V[:, :-1].detach().cpu().numpy() if isinstance(V, torch.Tensor) else V

    model_price_np, model_delta_np = evaluate_option_pricer.price_and_delta(
        S_mat=S_mid_np,
        K_mat=np.full_like(S_mid_np, K),
        tau_vec=tau_np,
        V_mat=V_mid_np
    )

    BS_price = torch.tensor(model_price_np, device=device, dtype=market_cfg.torch_dtype)
    BS_delta = torch.tensor(model_delta_np, device=device, dtype=market_cfg.torch_dtype)

    # ------------------------------------------------------------
    # 7) Π(t) 递推（使用真实有量纲动作）
    # ------------------------------------------------------------
    payoff_T = payoff_vanilla(S[:, -1], K, option_type)   # [N]

    Pi_mean = torch.zeros((n_paths, T+1), device=device)
    Pi_star = torch.zeros((n_paths, T+1), device=device)
    Pi_delta = torch.zeros((n_paths, T+1), device=device)

    Pi_mean[:, -1] = payoff_T
    Pi_star[:, -1] = payoff_T
    Pi_delta[:, -1] = payoff_T

    discount = torch.exp(-r * dt)

    for t in reversed(range(T)):
        Pi_mean[:, t] = discount * (Pi_mean[:, t+1] - a_mean_real[:, t] * dS[:, t])
        Pi_star[:, t] = discount * (Pi_star[:, t+1] - a_star_real[:, t] * dS[:, t])
        Pi_delta[:, t] = discount * (Pi_delta[:, t+1] - BS_delta[:, t] * dS[:, t])

    # ------------------------------------------------------------
    # 8) 绘图
    # ------------------------------------------------------------
    def plot_paths(data, title):
        plt.figure(figsize=(10,4))
        data_np = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data
        for i in range(min(n_paths,500 + 1)):
            plt.plot(data_np[i], alpha=alpha)
        plt.title(title)
        plt.xlabel("t")
        plt.grid(True)
        plt.tight_layout()

    # S
    plot_paths(S, "S Path")

    # # BS
    plot_paths(BS_delta, "BS Delta")
    plot_paths(BS_price, "BS Price")

    # 动作
    plot_paths(a_mean_real, "Actor Mean (real u_t)")
    plot_paths(a_star_real, "Critic a* (real u_t)")

    # Q
    plot_paths(- Q0_mean,   "Critic0 Q(s, u_mean) (real)")
    plot_paths(- Q_mean, "Critic Q(s, u_mean) (real)")
    plot_paths(- Q_star, "Critic Q(s, u_star) (real)")

    # Π
    plot_paths(Pi_mean, "Pi_mean(t)")
    plot_paths(Pi_star, "Pi_star(t)")
    plot_paths(Pi_delta, "Pi_delta(t)")

    print(f"t0, Pi_mean: {Pi_mean[:,0].mean()}, std: {Pi_mean[:,1].std()}")
    print(f"t0, Pi_star: {Pi_star[:,0].mean()}, std: {Pi_star[:,1].std()}")
    print(f"t0, Pi_delta: {Pi_delta[:,0].mean()}, std: {Pi_delta[:,1].std()}")
    plt.show()
    return

#%%
import torch

from configs.specifications.market_bs_cfg import MarketCFG
from configs.base.runtime_cfg import RuntimeCFG

class CrossSectionEvaluator:
    def __init__(
        self,
            OPTION_MODEL_TYPE: str,
            market_cfg:MarketCFG,
            runtime_cfg:RuntimeCFG,
            r_tensor,
            actor,
            critic0,
            critic,
            stock_path_generator,
    ):
        """
        初始化：只保存**不变**的配置和模型
        所有固定参数从 cfg 中提取，避免重复访问
        """
        # 核心模型与张量（不变）
        self.OPTION_MODEL_TYPE = OPTION_MODEL_TYPE
        self.actor = actor
        self.critic0 = critic0
        self.critic = critic
        self.r_tensor = r_tensor

        # 从 cfg 提取所有固定参数，单独保存，提升效率
        self.K = market_cfg.K
        self.r_f = market_cfg.r
        self.sigma = market_cfg.sigma
        self.option_type = market_cfg.option_type
        self.gamma_const = market_cfg.gamma_const
        self.device = market_cfg.device
        self.torch_dtype = market_cfg.torch_dtype
        self.lambda_list = torch.tensor(
            [critic.risk_lambda], device=r_tensor.device, dtype=r_tensor.dtype)  # 固定风险偏好列表
        self.L = self.lambda_list.numel()

        # 路径模拟固定参数
        self.S0 = market_cfg.S0
        self.mu = market_cfg.mu
        self.dt = market_cfg.dt
        self.N_ds = runtime_cfg.N_ds
        self.T_steps = market_cfg.T_steps
        self.s_over_k_steps = runtime_cfg.s_over_k_steps
        self.actor_risk_lambda_k = actor.risk_lambda

        # 动作裁剪边界（提前计算，不变）
        self.clamp_bounds = (
            float(actor.a_min_buf.item()),
            float(actor.a_max_buf.item())
        )

        # S/K 网格生成固定范围
        self.s_over_k_ranges = (
            (0.9, 1.1),
            (0.7, 1.3),
            (0.4, 1.6)
        )

        if OPTION_MODEL_TYPE == 'heston':
            from utils.pricing.quantlib_heston_model import QuantLibHestonModel as EvaluateOptionPricer
        elif OPTION_MODEL_TYPE == 'bs':
            from utils.pricing.pvv_bs_model import PVVBSModel as EvaluateOptionPricer
        else:
            raise NameError('OPTION_MODEL_TYPE is wrong')

        self.evaluate_option_pricer: QuantLibHestonModel = EvaluateOptionPricer(
            market_cfg=market_cfg,
        )
        self.stock_path_generator = stock_path_generator

    @torch.no_grad()
    def __call__(
            self, *,
            t,
            tau_all: torch.Tensor,
            tau_t: torch.Tensor,
    ):
        """
        每次循环调用：仅传入变化的参数 t, tau_all, tau_t
        完全复用初始化的固定参数，逻辑与原函数 1:1 一致
        """
        # ===================== 原函数逻辑开始 =====================
        V_new = None
        if self.OPTION_MODEL_TYPE == 'heston':
            V_new = float(self.stock_path_generator.V_path[0, t])
        S, log_returns, V = self.stock_path_generator.sim_paths(
            N_paths=self.N_ds,
            T_steps=self.T_steps - t,
            V_new = V_new,
        )  # S: [N, T+1]

        S_over_K_tensor = make_S_over_K(
            S, self.S0,
            device=self.device, dtype=self.torch_dtype,
            s_over_k_steps=self.s_over_k_steps,
            ranges=self.s_over_k_ranges,
        )
        # S_over_K_tensor = torch.sort(S_over_K_tensor, dim=-1).values
        ds, T1, m = S_over_K_tensor.shape

        S_over_K_t = S_over_K_tensor[0, 1]
        driftless_log_return = log_returns - self.r_tensor[t:] * self.dt
        expm1_dlr = torch.expm1(driftless_log_return)

        # 构造状态
        s_t0 = make_state_t(S_over_K_t, tau_t)

        # ===================== Critic 最优动作与价值 =====================
        a_star_flat = self.critic.a_star(s_t0, clamp_bounds=self.clamp_bounds)
        q_star_flat = self.critic(s_t0, a_star_flat)
        q0_star_flat = self.critic0(s_t0, a_star_flat)

        a_star = a_star_flat.reshape(1, m, self.L)
        q_star = q_star_flat.reshape(1, m, self.L)
        q0_star = q0_star_flat.reshape(1, m, self.L)

        # ===================== Actor 确定性动作 =====================
        a_mean_flat = self.actor.mean(s_t0)
        q_a_flat = self.critic(s_t0, a_mean_flat.repeat(self.L))
        q0_a_flat = self.critic0(s_t0, a_mean_flat)

        a_mean = a_mean_flat.reshape(1, m)
        q_a_mean = q_a_flat.reshape(1, m, self.L)
        q0_a_mean = q0_a_flat.reshape(1, m)

        # ===================== 未来收益计算 =====================
        if self.option_type == "c":
            payoff_T = torch.clamp(S_over_K_tensor[:, -1] - 1, min=0.0)
        else:
            payoff_T = torch.clamp(1 - S_over_K_tensor[:, -1], min=0.0)

        t_next = 1
        if t_next < T1 - 1:
            driftless = expm1_dlr[:, t_next:]
            a_seq = []

            for k in range(t_next, T1 - 1):
                s_k = make_state_t(
                    S_over_K_tensor[:, k:k+1],
                    tau_all[k],
                )[:, 0]

                # s_k_lambdaN = torch.cat(
                #     [s_k.repeat(1, self.L, 1),
                #      self.lambda_list.repeat_interleave(s_k.shape[1]).expand(s_k.shape[0], -1).unsqueeze(-1)],
                #     dim=-1
                # )
                #
                # s_kf = s_k_lambdaN.reshape(-1, 3)
                a_k = self.actor.mean(s_k).reshape(ds, m, self.L)
                a_seq.append(a_k[..., 0])

            if len(a_seq) == 0:
                pi_tp1 = payoff_T
            else:
                a_seq = torch.stack(a_seq, dim=1)
                y_seq = driftless[:, :a_seq.shape[1]]
                pi_tp1 = vectorized_pi_tp1(
                    self.device, self.torch_dtype,
                    payoff_T, a_seq.unsqueeze(-1), y_seq, self.gamma_const
                )[:, :, 0]
        else:
            pi_tp1 = payoff_T

        K_Pi_t = (self.K * pi_tp1.mean(dim=0)).cpu().numpy()

        # ===================== 价格与 Delta =====================
        S_vec = np.expand_dims((self.K * S_over_K_t).cpu().numpy(), axis=-1)
        tau_val = np.array([float(tau_t)])
        V_mid_np = np.full_like(S_vec, float(V[0,0])) if isinstance(V, torch.Tensor) else V

        model_price_np, model_delta_np = self.evaluate_option_pricer.price_and_delta(
            S_mat=S_vec,
            K_mat=np.full_like(S_vec, self.K),
            tau_vec=tau_val,
            V_mat=V_mid_np
        )

        # ===================== 组装 DataFrame =====================
        df = pd.DataFrame({
            "S_over_K": S_over_K_t.cpu().numpy(),
            "tau": np.full_like(S_over_K_t.cpu().numpy(), tau_val),
        })

        # 动作列
        df[f'{self.OPTION_MODEL_TYPE}_delta'] = model_delta_np
        a_star_data = (a_star[0] / S_over_K_t.unsqueeze(1)).cpu().numpy()
        actor_mean = (a_mean[0] / S_over_K_t).cpu().numpy()
        df[f'actor_mean_{self.actor_risk_lambda_k}'] = actor_mean
        df[[f"a_star_{i:.1f}" for i in self.lambda_list.cpu().numpy()]] = a_star_data

        # 价值列
        df[f'{self.OPTION_MODEL_TYPE}_price'] = model_price_np
        df['K_Pi_t'] = K_Pi_t
        q_star_data = (self.K * q_star[0]).cpu().numpy()
        q0_star_data = (self.K * q0_star[0]).cpu().numpy()
        q_a_mean = (self.K * q_a_mean[0]).cpu().numpy()
        q0_a_mean = (self.K * q0_a_mean[0]).cpu().numpy()

        df[f'K_Q0_a_mean_{self.actor_risk_lambda_k}'] = q0_a_mean
        df[[f"K_Q_a_mean_{i:.1f}" for i in self.lambda_list.cpu().numpy()]] = q_a_mean
        df[[f"K_Q0_star_{i:.1f}" for i in self.lambda_list.cpu().numpy()]] = q0_star_data
        df[[f"K_Q_star_{i:.1f}" for i in self.lambda_list.cpu().numpy()]] = q_star_data

        df = df.sort_values(by="S_over_K")

        return df