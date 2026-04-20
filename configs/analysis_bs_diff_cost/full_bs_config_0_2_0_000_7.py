from dataclasses import dataclass

# Import your configs
from configs.base import device, torch_dtype, RuntimeCFG, PolicyCFG
from configs.specifications import MarketCFG, ActorCFG, CriticCFG

@dataclass(frozen=False)
class FullConfig:
    # -------------------------------------------------------------------------
    # ALL PARAMETERS AT TOP LEVEL (EASY TO MODIFY)
    # -------------------------------------------------------------------------
    seed = 42
    #
    N_ds = 50_000
    num_total_updates:int = 5
    collect_data_times: int = 3
    #
    s_over_k_steps: int = 50
    a_grid_size: int = 31
    print_debug: bool = True

    actor_lambda: float = 100_000.0
    critic_lambda:float = actor_lambda

    # Market params
    S0: float = 100.0
    K: float = 100.0
    mu: float = 0.08
    r: float = 0.03
    # sigma: float = 0.15
    sigma: float = 0.2

    T: float = 0.5
    T_steps: int = 50

    transaction_cost_rate: float = 0.0007
    option_type: str = "p"

    # Critic0 network
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

    # Critic forms
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
            print_debug=self.print_debug
        )
        self.device = device
        self.torch_dtype = torch_dtype

        # Market
        self.market = MarketCFG(
            device=self.device,
            torch_dtype=self.torch_dtype,
            level="market",
            S0=self.S0,
            K=self.K,
            mu=self.mu,
            r=self.r,
            sigma=self.sigma,
            T=self.T,
            T_steps=self.T_steps,
            transaction_cost_rate=self.transaction_cost_rate,
            option_type=self.option_type
        )

        # Actor
        self.actor = ActorCFG(
            level="actor",
            risk_lambda=self.actor_lambda,
            state_dim=self.actor_state_dim,
            hidden=self.actor_hidden,
            a_min=self.actor_a_min,
            a_max=self.actor_a_max
        )

        # Critic0
        self.critic0 = CriticCFG(
            # device=self.device,
            # torch_dtype=self.torch_dtype,
            level="critic",
            form=self.critic0_form,
            risk_lambda=0,
            state_dim=self.critic0_state_dim,
            hidden=self.critic0_hidden,
            head_dim=self.critic0_head_dim
        )

        # Critic
        self.critic = CriticCFG(
            # device=self.device,
            # torch_dtype=self.torch_dtype,
            level="critic",
            form=self.critic_form,
            risk_lambda=self.critic_lambda,
            state_dim=self.critic_state_dim,
            hidden=self.critic_hidden,
            head_dim=self.critic_head_dim
        )

        # Policy
        self.policy = PolicyCFG(
            level="policy",
            actor_lambda=self.actor_lambda
        )

full_cfg = FullConfig()

if __name__ == '__main__':
    cfg = full_cfg