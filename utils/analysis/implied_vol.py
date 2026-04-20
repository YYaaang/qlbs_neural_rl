import numpy as np
from tqdm import tqdm
import py_vollib_vectorized


def bootstrap_implied_vol(
    S_T_array,
    K_array,
    S0,
    T,
    r,
    q,
    option_flag="c",
    n_bootstrap=100,
    sample_size=30_000,
    seed=42,
):
    """
    Bootstrap BS implied volatility curve from terminal samples S_T.

    Parameters
    ----------
    S_T_array : np.ndarray, shape (N_paths,)
        Terminal asset prices.
    K_array : np.ndarray, shape (n_K,)
        Strike grid.
    option_flag : {'c','p'}
        Call or put.
    """

    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)

    N_paths = S_T_array.shape[0]
    n_K = K_array.size

    # bootstrap indices
    idxs = rng.choice(N_paths, size=(n_bootstrap, sample_size), replace=True)

    IV_B = np.empty((n_bootstrap, n_K))

    for i, idx in enumerate(tqdm(idxs, leave=False)):
        S_T = S_T_array[idx]                 # (sample_size,)
        S_T = S_T[:, None]                   # (sample_size, 1)

        if option_flag == "c":
            payoff = np.maximum(S_T - K_array[None, :], 0.0)
        else:
            payoff = np.maximum(K_array[None, :] - S_T, 0.0)

        price = disc * payoff.mean(axis=0)   # (n_K,)

        ivs = py_vollib_vectorized.vectorized_implied_volatility(
            price=price,
            S=S0,
            K=K_array,
            t=T,
            r=r,
            q=q,
            flag=option_flag,
            model="black_scholes_merton",
            on_error="ignore",
            return_as="numpy",
        )

        IV_B[i] = ivs

    # robust statistics (ignore NaNs)
    IV_mean = np.nanmean(IV_B, axis=0)
    IV_5, IV_95 = np.nanpercentile(IV_B, [5, 95], axis=0)
    IV_SE = 1.96 * np.nanstd(IV_B, axis=0)

    return IV_mean, IV_SE, IV_5, IV_95