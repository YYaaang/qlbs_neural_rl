# utils/analysis/pnl_analysis.py

import numpy as np
import pandas as pd


# ============================================================
# Core statistics
# ============================================================

def pnl_summary(
    pnl: np.ndarray,
    quantiles=(0.01, 0.05, 0.1),
) -> dict:
    """
    Basic PnL statistics.

    Returns
    -------
    stats : dict
        {
            mean,
            std,
            var,
            quantiles,
            min,
            max
        }
    """
    stats = {
        "mean": np.mean(pnl),
        "std": np.std(pnl),
        "var": np.var(pnl),
        "min": np.min(pnl),
        "max": np.max(pnl),
    }

    qs = np.quantile(pnl, quantiles)
    stats["quantiles"] = dict(zip(quantiles, qs))

    return stats


# ============================================================
# Tail risk
# ============================================================

def var_cvar(
    pnl: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Value-at-Risk and Conditional VaR.

    Parameters
    ----------
    alpha : float
        Tail probability (e.g., 0.05)

    Returns
    -------
    dict
        { "VaR", "CVaR" }
    """
    var = np.quantile(pnl, alpha)
    cvar = pnl[pnl <= var].mean()

    return {
        "VaR": var,
        "CVaR": cvar,
    }


# ============================================================
# Comparative analysis
# ============================================================

def compare_pnl_methods(
    pnl_dict: dict,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compare multiple PnL arrays.

    Parameters
    ----------
    pnl_dict : dict
        {
            "Actor": pnl_actor,
            "BS": pnl_bs,
            "Heston": pnl_heston,
        }

    Returns
    -------
    DataFrame
        Rows = methods
        Columns = summary metrics
    """
    rows = []

    for name, pnl in pnl_dict.items():
        base = pnl_summary(pnl)
        tail = var_cvar(pnl, alpha)

        row = {
            "method": name,
            "mean": base["mean"],
            "std": base["std"],
            "VaR": tail["VaR"],
            "CVaR": tail["CVaR"],
            "min": base["min"],
        }

        rows.append(row)

    return pd.DataFrame(rows).set_index("method")


# ============================================================
# Dominance / win rate (RL-specific, but optional)
# ============================================================

def win_rate(
    pnl_a: np.ndarray,
    pnl_b: np.ndarray,
) -> float:
    """
    Probability that method A outperforms method B path-wise.
    """
    assert pnl_a.shape == pnl_b.shape
    return np.mean(pnl_a > pnl_b)


def dominance_matrix(
    pnl_dict: dict,
) -> pd.DataFrame:
    """
    Pairwise win-rate matrix.

    Entry (i, j) = P(PnL_i > PnL_j)
    """
    methods = list(pnl_dict.keys())
    mat = np.zeros((len(methods), len(methods)))

    for i, mi in enumerate(methods):
        for j, mj in enumerate(methods):
            if i == j:
                mat[i, j] = np.nan
            else:
                mat[i, j] = win_rate(
                    pnl_dict[mi],
                    pnl_dict[mj],
                )

    return pd.DataFrame(mat, index=methods, columns=methods)
