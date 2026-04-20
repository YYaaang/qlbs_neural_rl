from typing import Optional

import numpy as np
import py_vollib_vectorized as pvv

from configs.specifications.market_bs_cfg import MarketCFG

class PVVBSModel:
    """
    Risk-neutral Black–Scholes price / delta benchmark using py_vollib_vectorized.

    This class mirrors QuantLibHestonPricer, but for BS model.
    """

    def __init__(
            self,
            market_cfg: MarketCFG,
    ):
        """
        Parameters
        ----------
        r_f : risk-free rate
        sigma : BS volatility
        option_type : "c" or "p"
        """
        self.r_f = market_cfg.r
        self.sigma = market_cfg.sigma
        self.option_type = market_cfg.option_type.lower()

    # ------------------------------------------------------------------
    # grid pricing
    # ------------------------------------------------------------------
    def price_and_delta(
            self,
            S_mat: np.ndarray,     # [N_paths, T_steps]
            K_mat: np.ndarray,     # [N_paths, T_steps]
            tau_vec: np.ndarray,  # [T_steps]
            V_mat: Optional[float] = None, # meaning less
    ):
        """
        Compute BS price and delta on a whole state grid.

        Returns
        -------
        price_mat : np.ndarray, shape [N_paths, T_steps]
        delta_mat : np.ndarray, shape [N_paths, T_steps]
        """

        N, T = S_mat.shape

        assert K_mat.shape == (N, T)
        assert tau_vec.shape[0] == T

        # ------------------------------------------------------------
        # vectorize inputs
        # ------------------------------------------------------------
        S_flat = S_mat.reshape(-1)
        K_flat = K_mat.reshape(-1)

        tau_flat = np.tile(tau_vec, reps=N)

        # ------------------------------------------------------------
        # BS price
        # ------------------------------------------------------------
        price_flat = pvv.vectorized_black_scholes(
            self.option_type,
            S_flat,
            K_flat,
            tau_flat,
            self.r_f,
            self.sigma,
            return_as="numpy",
        )

        # ------------------------------------------------------------
        # BS delta
        # ------------------------------------------------------------
        delta_flat = pvv.vectorized_delta(
            self.option_type,
            S_flat,
            K_flat,
            tau_flat,
            self.r_f,
            self.sigma,
            return_as="numpy",
        )

        price_mat = price_flat.reshape(N, T)
        delta_mat = delta_flat.reshape(N, T)

        return price_mat, delta_mat

    def price(
        self,
        S_mat: np.ndarray,
        K_mat: np.ndarray,
        tau_vec: np.ndarray,
    ):
        prices, _ = self.price_and_delta(S_mat, K_mat, tau_vec)
        return prices

    def delta(
        self,
        S_mat: np.ndarray,
        K_mat: np.ndarray,
        tau_vec: np.ndarray,
    ):
        _, deltas = self.price_and_delta(S_mat, K_mat, tau_vec)
        return deltas


def bs_delta_from_iv(
    S0: float,
    K_array: np.ndarray,
    T: float,
    r: float,
    iv_array: np.ndarray,
    option_flag: str,
):
    """
    Compute BS delta using strike-dependent implied volatility.

    Parameters
    ----------
    S0 : float
        Spot price.
    K_array : np.ndarray
        Strike grid.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    iv_array : np.ndarray
        Implied volatility for each strike.
    option_flag : str
        "c" or "p".

    Returns
    -------
    delta_bs : np.ndarray
        BS delta evaluated at (S0, K, T, sigma=iv_K).
    """

    S = np.full_like(K_array, S0, dtype=float)
    tau = np.full_like(K_array, T, dtype=float)

    delta_bs = pvv.vectorized_delta(
        option_flag,
        S,
        K_array,
        tau,
        r,
        iv_array,
        return_as="numpy",
    )

    return delta_bs
