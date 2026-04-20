import numpy as np


def delta_hedge_full_analytics(
        S_paths: np.ndarray,
        delta_paths: np.ndarray,
        initial_option_price: float,
        K: float,
        option_type: str = "call",
        r: float = 0.0,
        transaction_cost_rate: float = 0.0,
        dt: float = 1 / 252
) -> dict:
    N, T_plus_1 = S_paths.shape
    T = T_plus_1 - 1
    dt = float(dt)
    # 1. 计算调仓产生的即时现金流 (未计息)
    # t=0 的初始现金流：卖期权收入 - 买初始头寸支出
    initial_cash = initial_option_price - delta_paths[:, 0] * S_paths[:, 0]

    # 随后的 Delta 变动导致的现金流 (t=1 到 T)
    delta_diff = np.diff(delta_paths, axis=1)
    rebalance_cash_flows = -(delta_diff * S_paths[:, 1:])

    # 交易成本 (t=1 到 T)
    tc_paths = np.abs(delta_diff) * S_paths[:, 1:] * transaction_cost_rate

    # 汇总所有时间点的原始现金流变动 (N, T+1)
    # index 0 是初始现金，index 1: 是后续变动
    raw_flows = np.zeros((N, T_plus_1))
    raw_flows[:, 0] = initial_cash
    raw_flows[:, 1:] = rebalance_cash_flows - tc_paths

    # 2. 计算计息后的现金账户路径 (关键逻辑)
    # 现金账户递推公式：Cash_t = Cash_{t-1} * exp(r*dt) + Flow_t
    # 这种形式可以用带权重的 cumsum 或者简单的循环。
    # 如果 r=0，直接 np.cumsum(raw_flows)；如果 r!=0，为了绝对严谨建议处理如下：

    cash_paths = np.zeros((N, T_plus_1))
    cash_paths[:, 0] = initial_cash
    for t in range(1, T_plus_1):
        cash_paths[:, t] = cash_paths[:, t - 1] * np.exp(r * dt) + raw_flows[:, t]

    # 3. 计算投资组合价值路径 (Portfolio Value = Cash + Delta * S)
    portfolio_paths = cash_paths + delta_paths * S_paths

    # 4. 计算期权 Payoff (仅终点)
    if option_type.lower() == "call":
        payoff = np.maximum(S_paths[:, -1] - K, 0.0)
    else:
        payoff = np.maximum(K - S_paths[:, -1], 0.0)

    terminal_pnl = portfolio_paths[:, -1] - payoff

    return {
        "terminal_pnl": terminal_pnl,
        "cash_paths": cash_paths,  # 现金账户完整路径
        "portfolio_paths": portfolio_paths,  # 组合价值完整路径
        "payoff": payoff,
        "total_tc": np.sum(tc_paths, axis=1)
    }