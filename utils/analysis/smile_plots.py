import matplotlib.pyplot as plt
import numpy as np


def plot_iv_smile(
    K_array: np.ndarray,
    smiles: dict,
    *,
    title: str = None,
    xlabel: str = r"Strike Price",
    ylabel: str = r"BS Implied Volatility",
    figsize=(10, 6),
    show: bool = True,
):
    """
    Plot implied volatility smiles with optional bootstrap confidence bands.

    Parameters
    ----------
    K_array : np.ndarray, shape (n_K,)
        Strike grid.

    smiles : dict
        Dictionary defining smiles to plot.

        Expected format:
        {
            "QE": {
                "mean": IV_mean,
                "low":  IV_5,
                "high": IV_95,
                "color": "tab:blue",
            },
            "EM": {
                "mean": IV_mean,
                "low":  IV_5,
                "high": IV_95,
                "color": "tab:red",
            },
            "AN": {
                "mean": iv_array_FFT,
                "color": "tab:green",
            }
        }

        For smiles without uncertainty bands (FFT / QuantLib),
        simply omit "low" and "high".

    title : str, optional
        Figure title.

    show : bool
        Whether to call plt.show().
    """

    plt.figure(figsize=figsize)

    for name, spec in smiles.items():
        mean = spec["mean"]
        color = spec.get("color", None)

        plt.plot(
            K_array,
            mean,
            label=name,
            color=color,
            linewidth=2,
        )

        # confidence band (bootstrap)
        if "low" in spec and "high" in spec:
            plt.fill_between(
                K_array,
                spec["low"],
                spec["high"],
                color=color,
                alpha=0.2,
            )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


    if title is not None:
        plt.title(title)

    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    if show:
        plt.show()

    return None