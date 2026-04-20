import torch


def cal_transaction_cost(
        cfg,
        a_tp1,
        a_t,
        expm1_dlr,
):
    return cfg.transaction_cost_rate * torch.abs(
        (a_tp1 * cfg.gamma_const).view(*a_tp1.shape,1)
        - a_t * (expm1_dlr + 1).view(-1,1,1)
    ).unsqueeze(1)  # [ds, 1, m_dim, A_dim]
