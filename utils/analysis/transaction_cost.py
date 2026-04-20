import numpy as np


def cal_transaction_cost_np(
    a_next: np.ndarray,
    a_curr: np.ndarray,
    dlr: np.ndarray,
    gamma: float,
    tc_rate: float,
) -> np.ndarray:
    """
    Vectorized transaction cost for path-wise hedging.

    Parameters
    ----------
    a_next : np.ndarray
        Hedge at time t+1, shape (N, T-1)
    a_curr : np.ndarray
        Hedge at time t, shape (N, T-1)
    dlr : np.ndarray
        Discounted log-return increment, shape (N, T-1)
    gamma : float
        Discount factor
    tc_rate : float
        Transaction cost rate

    Returns
    -------
    tc : np.ndarray
        Transaction costs, shape (N, T-1)
    """
    return tc_rate * np.abs(
        a_next * gamma - a_curr * (1.0 + dlr)
    )