from dataclasses import dataclass

# Import your configs
from configs.base.env_cfg import device, torch_dtype
from configs.base.runtime_cfg import RuntimeCFG
from configs.specifications.market_heston_cfg import MarketCFG
from configs.base.policy_cfg import PolicyCFG
from configs.specifications.actor_cfg import ActorCFG
from configs.specifications.critic_cfg import CriticCFG


@dataclass(frozen=False)
class FullConfig:
    # -------------------------------------------------------------------------
    # ALL PARAMETERS AT TOP LEVEL
    # -------------------------------------------------------------------------
    seed = 42

    # Runtime
    N_ds = 50_000
    num_total_updates: int = 5
    collect_data_times: int = 3

    s_over_k_steps: int = 50
    a_grid_size: int = 31
    print_debug: bool = True

    actor_lambda: float = 100_000.0
    critic_lambda: float = actor_lambda

    # -------------------------------------------------------------------------
    # Market params (PHYSICAL HESTON)
    # -------------------------------------------------------------------------
    S0: float = 100.0
    K: float = 100.0

    mu: float = 0.08            # ✅ physical drift

    # ~
    V0: float = 0.04            # ✅ initial variance

    # ============================================================
    # 1 美股常态环境（Core US Equity Regime）
    # ------------------------------------------------------------
    # • 微笑：显著而稳定，OTM Put 明显更贵但不过度极端
    # • 波动率：活跃但可预测，高波动阶段可控
    # • 走势：正常上涨伴随回撤，risk-on / risk-off 切换频繁
    # • 对标：SPX 常年状态、NDX、AAPL / MSFT 等 mega-cap
    # ------------------------------------------------------------
    kappa: float = 2.0  # 回归速度中等，波动率不会滞留太久
    theta: float = 0.035  # 长期波动约 18–19%
    sigma: float = 0.22  # vol-of-vol 偏高，维持真实 smile
    rho: float = -0.45  # 典型美股 leverage effect



    r: float = 0.03
    T: float = 0.25 # 1/4 year, 3 months
    T_steps: int = 60 # ～60 days
    # T_steps: int =  10 # ～60 days

    transaction_cost_rate: float = 0.0005
    transaction_cost_rate: float = 0.000
    option_type: str = "p"

    # -------------------------------------------------------------------------
    # Critic0 network
    # -------------------------------------------------------------------------
    critic0_state_dim: int = 3
    critic0_hidden: tuple = (128, 128)
    critic0_head_dim: int = 64

    # Actor network
    actor_state_dim: int = 3
    actor_hidden: tuple = (128, 128)
    actor_a_min: float = -2.0
    actor_a_max: float = 2.0

    # Critic network
    critic_state_dim: int = 3
    critic_hidden: tuple = (128, 128)
    critic_head_dim: int = 64

    critic0_form: str = "linear"
    critic_form: str = "quadratic"

    # -------------------------------------------------------------------------
    # Auto-initialized sub-configs
    # -------------------------------------------------------------------------
    def __post_init__(self):
        # Runtime
        self.runtime = RuntimeCFG(
            N_ds=self.N_ds,
            num_total_updates=self.num_total_updates,
            collect_data_times=self.collect_data_times,
            s_over_k_steps=self.s_over_k_steps,
            a_grid_size=self.a_grid_size,
            print_debug=self.print_debug,
        )

        self.device = device
        self.torch_dtype = torch_dtype

        # ---------------------------------------------------------------------
        # Market (Physical Heston)
        # ---------------------------------------------------------------------
        self.market = MarketCFG(
            device=self.device,
            torch_dtype=self.torch_dtype,
            level="market",

            S0=self.S0,
            V0=self.V0,

            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma,     # vol-of-vol
            rho=self.rho,

            mu=self.mu,
            r=self.r,
            T=self.T,
            K=self.K,
            T_steps=self.T_steps,

            transaction_cost_rate=self.transaction_cost_rate,
            option_type=self.option_type,
        )

        # Actor
        self.actor = ActorCFG(
            level="actor",
            risk_lambda=self.actor_lambda,
            state_dim=self.actor_state_dim,
            hidden=self.actor_hidden,
            a_min=self.actor_a_min,
            a_max=self.actor_a_max,
        )

        # Critic0
        self.critic0 = CriticCFG(
            level="critic",
            form=self.critic0_form,
            risk_lambda=0,
            state_dim=self.critic0_state_dim,
            hidden=self.critic0_hidden,
            head_dim=self.critic0_head_dim,
        )

        # Critic
        self.critic = CriticCFG(
            level="critic",
            form=self.critic_form,
            risk_lambda=self.critic_lambda,
            state_dim=self.critic_state_dim,
            hidden=self.critic_hidden,
            head_dim=self.critic_head_dim,
        )

        # Policy
        self.policy = PolicyCFG(
            level="policy",
            actor_lambda=self.actor_lambda,
        )


full_cfg = FullConfig()

if __name__ == "__main__":
    cfg = full_cfg