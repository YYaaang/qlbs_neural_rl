import QuantLib as ql
import numpy as np
from configs.specifications.market_heston_cfg import MarketCFG


class QuantLibHestonModel:
    """
    Heston option pricer for validation / benchmark purposes.

    Supports:
      - Analytic Heston price + bump delta
      - FD (PDE) Heston price + PDE delta or bump delta

    This class is NOT intended for production pricing, only for
    cross-checking RL / QLBS hedging results.
    """

    # ------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------
    def __init__(
        self,
        market_cfg: MarketCFG,
        engine_type: str = "analytic",   # "analytic" | "fd"
        delta_method: str = "auto",      # "auto" | "fd" | "bump"
        fd_params: dict | None = None,
        bump_eps: float = 1e-4,
        day_count=None,
        calendar=None,
    ):
        """
        Parameters
        ----------
        engine_type:
            "analytic" : AnalyticHestonEngine (price only)
            "fd"       : FdHestonVanillaEngine (price + delta)

        delta_method:
            "auto" : analytic -> bump, fd -> engine delta
            "fd"   : force PDE delta (fd engine only)
            "bump" : force bump-and-reprice delta

        fd_params:
            parameters passed to FdHestonVanillaEngine

        bump_eps:
            relative bump size for spot when computing delta
        """

        assert engine_type in ["analytic", "fd"]
        assert delta_method in ["auto", "fd", "bump"]

        self.market_cfg = market_cfg

        self.engine_type = engine_type
        self.delta_method = delta_method
        self.bump_eps = bump_eps

        # Heston parameters (risk-neutral)
        self.kappa = market_cfg.kappa
        self.theta = market_cfg.theta
        self.sigma_v = market_cfg.sigma
        self.rho = market_cfg.rho
        self.r_f = market_cfg.r

        self.option_type = (
            ql.Option.Call
            if market_cfg.option_type.lower().startswith("c")
            else ql.Option.Put
        )

        # QuantLib global settings
        self.day_count = day_count or ql.Actual365Fixed()
        self.calendar = calendar or ql.NullCalendar()

        self.today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = self.today

        self.rf_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(self.today, self.r_f, self.day_count)
        )
        self.div_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(self.today, 0.0, self.day_count)
        )

        # FD parameters (fixed across all states!)
        # self.fd_params = fd_params or dict(
        #     tGrid=100,
        #     xGrid=200,
        #     vGrid=100,
        #     dampingSteps=0,
        #     schemeDesc=ql.FdmSchemeDesc.Douglas(),
        # )
        self.fd_params = fd_params or (100, 200, 100, 0)

    # ------------------------------------------------------------------
    # path simulation (risk-neutral, QuantLib Euler)
    # ------------------------------------------------------------------
    def simulate_paths(
            self,
            N_paths: int,
            seed: int = 42,
            T:float = None,
            T_steps:int = None,
            antithetic: bool = False,
    ):
        """
        Simulate Heston paths under the risk-neutral measure
        using QuantLib's Euler discretization.

        Parameters
        ----------
        N_paths : int
            Number of Monte Carlo paths.
        seed : int
            Random seed.
        antithetic : bool
            Whether to use antithetic variates.

        Returns
        -------
        S_paths : np.ndarray, shape (N_paths, T_steps + 1)
            Simulated asset price paths.
        V_paths : np.ndarray, shape (N_paths, T_steps + 1)
            Simulated variance paths.
        """

        # time grid
        if T is None:
            T = self.market_cfg.T
        if T_steps is None:
            T_steps = self.market_cfg.T_steps

        time_grid = ql.TimeGrid(T, T_steps)

        # initial state
        S0 = self.market_cfg.S0
        V0 = self.market_cfg.V0

        process = ql.HestonProcess(
            self.rf_ts,
            self.div_ts,
            ql.QuoteHandle(ql.SimpleQuote(S0)),
            V0,
            self.kappa,
            self.theta,
            self.sigma_v,
            self.rho,
        )

        # random number generator
        dim = process.factors() * (len(time_grid) - 1)

        ursg = ql.UniformRandomSequenceGenerator(
            dim,
            ql.UniformRandomGenerator(seed),
        )

        grsg = ql.GaussianRandomSequenceGenerator(ursg)

        path_generator = ql.GaussianMultiPathGenerator(
            process,
            time_grid,
            grsg,
            antithetic,
        )

        # storage
        S_paths = np.zeros((N_paths, len(time_grid)))
        V_paths = np.zeros_like(S_paths)

        for i in range(N_paths):
            sample = path_generator.next()
            values = sample.value()

            # values[0] -> S path, values[1] -> V path
            S_paths[i, :] = values[0]
            V_paths[i, :] = values[1]

        return S_paths, V_paths

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def NPV(self, S_t, V_t, K, tau):
        return self._analytic_price(S_t, V_t, K, tau)

    def price_and_delta(
            self,
            S_mat: np.ndarray,   # [N_paths, T_steps]
            K_mat: np.ndarray,   # [N_paths, T_steps]
            tau_vec: np.ndarray, # [T_steps]
            V_mat: np.ndarray,  # [N_paths, T_steps]
    ):
        N, T = S_mat.shape
        assert V_mat.shape == (N, T)
        assert K_mat.shape == (N, T)
        assert tau_vec.shape[0] == T

        prices = np.zeros((N, T))
        deltas = np.zeros((N, T))

        for t in range(T):
            tau = float(tau_vec[t])
            for i in range(N):
                p, d = self._price_and_delta_at_state(
                    S_t=float(S_mat[i, t]),
                    V_t=float(V_mat[i, t]),
                    K=float(K_mat[i, t]),
                    tau=tau,
                )
                prices[i, t] = p
                deltas[i, t] = d

        return prices, deltas

    def price(
        self,
        S_mat: np.ndarray,
        V_mat: np.ndarray,
        K_mat: np.ndarray,
        tau_vec: np.ndarray,
    ):
        prices, _ = self.price_and_delta(S_mat, V_mat, K_mat, tau_vec)
        return prices

    def delta(
        self,
        S_mat: np.ndarray,
        V_mat: np.ndarray,
        K_mat: np.ndarray,
        tau_vec: np.ndarray,
    ):
        _, deltas = self.price_and_delta(S_mat, V_mat, K_mat, tau_vec)
        return deltas

    # ------------------------------------------------------------------
    # core dispatcher
    # ------------------------------------------------------------------
    def _price_and_delta_at_state(
        self,
        S_t: float,
        V_t: float,
        K: float,
        tau: float,
    ):
        if self.engine_type == "fd":
            price, fd_delta = self._fd_price_and_delta(
                S_t=S_t,
                V_t=V_t,
                K=K,
                tau=tau,
            )

            if self.delta_method in ["auto", "fd"]:
                delta = fd_delta
            else:
                delta = self._bump_delta(S_t, V_t, K, tau)

            return price, delta

        # analytic engine
        price = self._analytic_price(
            S_t=S_t,
            V_t=V_t,
            K=K,
            tau=tau,
        )
        delta = self._bump_delta(S_t, V_t, K, tau)
        return price, delta

    # ------------------------------------------------------------------
    # pricing backends
    # ------------------------------------------------------------------
    def _analytic_price(self, S_t, V_t, K, tau):
        option = self._build_option(S_t, K, tau)

        model = ql.HestonModel(self._build_process(S_t, V_t))
        engine = ql.AnalyticHestonEngine(model)

        option.setPricingEngine(engine)
        return option.NPV()

    def _fd_price_and_delta(self, S_t, V_t, K, tau):
        option = self._build_option(S_t, K, tau)

        model = ql.HestonModel(self._build_process(S_t, V_t))

        tGrid, xGrid, vGrid, dampingSteps = self.fd_params

        engine = ql.FdHestonVanillaEngine(
            model,
            tGrid,
            xGrid,
            vGrid,
            dampingSteps,
        )

        option.setPricingEngine(engine)

        price = option.NPV()
        delta = option.delta()

        return price, delta

    # ------------------------------------------------------------------
    # bump delta (model-agnostic)
    # ------------------------------------------------------------------
    def _bump_delta(self, S_t, V_t, K, tau):
        bump = self.bump_eps * max(S_t, 1.0)

        p_up = self._price_at_state(S_t + bump, V_t, K, tau)
        p_dn = self._price_at_state(S_t - bump, V_t, K, tau)

        return (p_up - p_dn) / (2.0 * bump)

    def _price_at_state(self, S_t, V_t, K, tau):
        if self.engine_type == "fd":
            price, _ = self._fd_price_and_delta(S_t, V_t, K, tau)
            return price
        else:
            return self._analytic_price(S_t, V_t, K, tau)

    # ------------------------------------------------------------------
    # QuantLib helpers
    # ------------------------------------------------------------------
    def _build_option(self, S_t, K, tau):
        maturity = self.today + int(365 * tau)
        payoff = ql.PlainVanillaPayoff(self.option_type, K)
        exercise = ql.EuropeanExercise(maturity)
        return ql.VanillaOption(payoff, exercise)

    def _build_process(self, S_t, V_t):
        spot = ql.QuoteHandle(ql.SimpleQuote(S_t))

        return ql.HestonProcess(
            self.rf_ts,
            self.div_ts,
            spot,
            V_t,
            self.kappa,
            self.theta,
            self.sigma_v,
            self.rho,
        )


def heston_delta_slice(
    heston_model,
    S0: float,
    v0: float,
    K_array: np.ndarray,
    T: float,
):
    """
    Compute Heston delta (QuantLib) for a single maturity slice.

    Returns
    -------
    delta_heston : np.ndarray
        Heston delta for each strike.
    """

    delta_heston = np.array([
        heston_model._bump_delta(
            S_t=S0,
            V_t=v0,
            K=K,
            tau=T,
        )
        for K in K_array
    ])

    return delta_heston
