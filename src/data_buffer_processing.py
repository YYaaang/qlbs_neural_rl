import torch

from src.qlbs import generate_mean_q_function
from src.data_processing import make_S_over_K, build_action_grid_random, make_state_t

from configs.specifications.market_bs_cfg import MarketCFG
from configs.base.runtime_cfg import RuntimeCFG


@torch.no_grad()
def collect_data_at_t(
        device, torch_dtype,
        path_generator,
        marketcfg: MarketCFG,
        runtime_cfg: RuntimeCFG,
        buffer,
        t,
        t_step,
        total_updates,
        tau_t_local_backward,
        r_tensor,
        all_risk_lambda_k,
        targ_group,
        only_critic=False,
):
    if path_generator.model == 'bs':
        generate_and_collect_data_for_timestep(
            device=device, torch_dtype=torch_dtype,
            path_generator=path_generator,
            marketcfg=marketcfg,
            runtime_cfg=runtime_cfg,
            buffer=buffer,
            t=t,
            t_step=t_step,
            total_updates=total_updates,
            tau_t_local_backward=tau_t_local_backward,
            r_tensor=r_tensor,
            all_risk_lambda_k=all_risk_lambda_k,
            targ_group=targ_group,
            only_critic=only_critic
        )

    elif path_generator.model == 'heston':
        collect_existing_data_for_timestep(
            device=device, torch_dtype=torch_dtype,
            path_generator=path_generator,
            marketcfg=marketcfg,
            runtime_cfg=runtime_cfg,
            buffer=buffer,
            t=t,
            t_step=t_step,
            total_updates=total_updates,
            tau_t_local_backward=tau_t_local_backward,
            r_tensor=r_tensor,
            all_risk_lambda_k=all_risk_lambda_k,
            targ_group=targ_group,
            only_critic=only_critic,
        )

@torch.no_grad()
def collect_existing_data_for_timestep(
        device, torch_dtype,
        path_generator,
        marketcfg:MarketCFG,
        runtime_cfg:RuntimeCFG,
        buffer,
        t,
        t_step,
        total_updates,
        tau_t_local_backward,
        r_tensor,
        all_risk_lambda_k,
        targ_group,
        only_critic=False,
):
    """

    """
    critic_dim = targ_group.critic_targ.state_dim

    (total_updates, m_dim, A_dim, G0, GN,
     state_big, A_big, Y0_big, YN_big) = _generate_empty_data(
        device=device,
        torch_dtype=torch_dtype,
        total_updates = total_updates,
        runtime_cfg=runtime_cfg,
        critic_dim=critic_dim,
        only_critic = only_critic,
    )


    if G0 == 0:
        all_risk_lambda_k = all_risk_lambda_k[-1]

    # 使用已知数据
    S, log_returns = path_generator.get_existing_next_paths(
        t=t,
        N_paths=runtime_cfg.N_ds,
    )
    S = S / S[:,:1] * marketcfg.S0

    # =======================================================================
    #                            主循环 K 次生成数据
    # =======================================================================
    for k in range(total_updates):
        # -----------------------------------------------------------
        # (1) paths (每次取2/3)，增加随机性
        # -----------------------------------------------------------
        random_indices = torch.randperm(S.shape[0])[:int(S.shape[0] * 2/3)]

        driftless_log_return = log_returns[random_indices] - r_tensor[t] * marketcfg.dt
        expm1_dlr = torch.expm1(driftless_log_return)

        state_t0, a_grid_t0, Y0_group, YN_group = _generate_one_times_data(
            device, torch_dtype,
            S[random_indices], expm1_dlr, marketcfg, tau_t_local_backward, t_step,
            all_risk_lambda_k, targ_group,
            m_dim, A_dim, G0,
        )

        # -----------------------------------------------------------
        # (7) 写入 staging buffer
        # -----------------------------------------------------------
        state_big[k] = state_t0
        A_big[k] = a_grid_t0
        Y0_big[k] = Y0_group
        YN_big[k] = YN_group

    # ====================================================================
    # (8) flatten mega-batch
    # ====================================================================

    state_merge = state_big.reshape(total_updates * GN, critic_dim)
    A_merge = A_big.reshape(total_updates * GN, A_dim)
    Y0_merge = Y0_big.reshape(total_updates * G0, A_dim)
    YN_merge = YN_big.reshape(total_updates * GN, A_dim)


    buffer.add_from_tensor(
        #
        t_step=t_step,
        #
        S_input=state_merge,          # [total_updates * G0, 2]
        #
        A_input=A_merge,           # [total_updates * GN, A_dim]
        #
        Y0_input=Y0_merge,          # [total_updates * G0, A_dim]
        #
        Y_input=YN_merge,           # [total_updates * GN, A_dim]
    )
    return

@torch.no_grad()
def generate_and_collect_data_for_timestep(
        device, torch_dtype,
        path_generator,
        marketcfg:MarketCFG,
        runtime_cfg:RuntimeCFG,
        buffer,
        t,
        t_step,
        total_updates,
        tau_t_local_backward,
        r_tensor,
        all_risk_lambda_k,
        targ_group,
        only_critic=False,
):
    """

    """
    critic_dim = targ_group.critic_targ.state_dim

    (total_updates, m_dim, A_dim, G0, GN,
     state_big, A_big, Y0_big, YN_big) = _generate_empty_data(
        device=device,
        torch_dtype=torch_dtype,
        total_updates = total_updates,
        runtime_cfg=runtime_cfg,
        critic_dim=critic_dim,
        only_critic = only_critic,
    )

    if G0 == 0:
        all_risk_lambda_k = all_risk_lambda_k[-1]

    # =======================================================================
    #                            主循环 K 次生成数据
    # =======================================================================
    for k in range(total_updates):
        # -----------------------------------------------------------
        # (1) sim paths
        # -----------------------------------------------------------

        S, log_returns = path_generator.sim_next_path(
            t=19,
            N_paths=runtime_cfg.N_ds,
        )

        driftless_log_return = log_returns - r_tensor[t] * marketcfg.dt
        expm1_dlr = torch.expm1(driftless_log_return)

        state_t0, a_grid_t0, Y0_group, YN_group = _generate_one_times_data(
            device, torch_dtype,
            S, expm1_dlr, marketcfg, tau_t_local_backward, t_step,
            all_risk_lambda_k, targ_group,
            m_dim, A_dim, G0,
        )

        # -----------------------------------------------------------
        # (7) 写入 staging buffer
        # -----------------------------------------------------------
        state_big[k] = state_t0
        A_big[k] = a_grid_t0
        Y0_big[k] = Y0_group
        YN_big[k] = YN_group

    # ====================================================================
    # (8) flatten mega-batch
    # ====================================================================

    state_merge = state_big.reshape(total_updates * GN, critic_dim)
    A_merge = A_big.reshape(total_updates * GN, A_dim)
    Y0_merge = Y0_big.reshape(total_updates * G0, A_dim)
    YN_merge = YN_big.reshape(total_updates * GN, A_dim)


    buffer.add_from_tensor(
        t_step=t_step,
        #
        S_input=state_merge,          # [total_updates * G0, 2]
        #
        A_input=A_merge,           # [total_updates * GN, A_dim]
        #
        Y0_input=Y0_merge,          # [total_updates * G0, A_dim]
        #
        Y_input=YN_merge,           # [total_updates * GN, A_dim]
    )
    return


def _generate_empty_data(
        device: torch.device,
        torch_dtype: torch.dtype,
        total_updates: int,
        runtime_cfg:RuntimeCFG,
        critic_dim,
        only_critic=False,
):
    m_dim = runtime_cfg.s_over_k_steps
    A_dim = runtime_cfg.a_grid_size

    G0 = m_dim                # λ=0 的 group 数
    if only_critic:
        G0 = 0
    GN = m_dim                # λ!=0 的 group 数

    # λ=0 and λ!=0 bucket
    state_big = torch.empty((total_updates, GN, critic_dim), device=device, dtype=torch_dtype)
    A_big = torch.empty((total_updates, GN, A_dim), device=device, dtype=torch_dtype)
    # -------- λ=0 bucket --------
    Y0_big = torch.empty((total_updates, G0, A_dim), device=device, dtype=torch_dtype)
    # -------- λ!=0 bucket --------
    YN_big = torch.empty((total_updates, GN, A_dim), device=device, dtype=torch_dtype)
    return total_updates, m_dim, A_dim, G0, GN, state_big, A_big, Y0_big, YN_big

def _generate_one_times_data(
        device: torch.device, torch_dtype: torch.dtype,
        S, expm1_dlr, marketcfg, tau_t_local_backward, t_step,
        all_risk_lambda_k, targ_group,
        m_dim, A_dim, G0,
):
    # -----------------------------------------------------------
    # (2) S_over_K
    # -----------------------------------------------------------

    S_over_K_tensor = make_S_over_K(
        S, marketcfg.S0,
        device=device, dtype=torch_dtype,
        s_over_k_steps=m_dim,
        ranges=((0.9, 1.1), (0.75, 1.27), (0.3, 1.7)),
    )

    # -----------------------------------------------------------
    # (3) make state
    # -----------------------------------------------------------
    state_t0 = make_state_t(
        S_over_K_tensor[:, 0].mean(dim=0),
        tau_t_local_backward[0],
    )  # shape: [..., 2]

    # -----------------------------------------------------------
    # (4) action grid
    # -----------------------------------------------------------
    a_grid_t0 = build_action_grid_random(
        device=device,
        torch_dtype=torch_dtype,
        batch_size=m_dim,
        a_grid_size=A_dim
    )
    # a_grid_t0 = build_action_grid_from_actor(
    #     device=device,
    #     torch_dtype=torch_dtype,
    #     actor=actor,
    #     s_t_base=state_t0,
    #     a_grid_size=A_dim
    # )    # [m_dim, A_dim]

    # -----------------------------------------------------------
    # (5) Q-grid
    # -----------------------------------------------------------
    Y_grid = generate_mean_q_function(
        marketcfg,
        t_step,
        S,
        S_over_K_tensor[:, 1],  # [ds, m_dim]
        tau_t_local_backward[1],
        expm1_dlr,
        a_grid_t0,  # [m_dim, A_dim]
        all_risk_lambda_k,
        targ_group,
    )  # [L, m_dim, A_dim]

    # -----------------------------------------------------------
    # (6) flatten to group-level
    # -----------------------------------------------------------

    # ==== Y_group: [G, A_dim], λ block-friendly ====
    Y_group = Y_grid.reshape(-1, A_dim)  # [G, A_dim]  #

    Y0_group = Y_group[:G0]
    YN_group = Y_group[G0:]

    return state_t0, a_grid_t0, Y0_group, YN_group