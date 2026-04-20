import torch

@torch.no_grad()
def payoff_vanilla(S_T, K, option_type='p'):
    """期权到期收益H(S_T)（论文2.1节）"""
    if option_type == 'p':
        return torch.clamp(K - S_T, min=0.0)
    return torch.clamp(S_T - K, min=0.0)
