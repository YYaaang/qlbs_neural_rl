# ============================================================
# init_train_models.py
# ============================================================
import time
import torch
import numpy as np
import os
import random

from src.rl_models import Actor, Critic0, Critic, ModelsTargGroup
from src.trainer import ACTrainer
from src.buffer import ReplayBuffer

from src.bs_model import BSConditionalSampler as StockPathGenerator

from utils.log_print import LogPrint
from utils.analysis.evaluation import CrossSectionEvaluator, evaluate_full_paths
from utils.train_one_step import train_one_step
from utils.model_saver import save_full_experiment
from configs.full_bs_config import FullConfig as FullConfig


def train_new_models(
        full_cfg: FullConfig,
        option_model_type:str,
        save_root: str = 'models',
):
    train_all = True
    # -------------------------------------------------
    if option_model_type == 'heston':
        from src.heston_model import HestonConditionalSampler as StockPathGenerator
    elif option_model_type == 'bs':
        from src.bs_model import BSConditionalSampler as StockPathGenerator
    else:
        raise NameError('OPTION_MODEL_TYPE is wrong')
    # -------------------------------------------------
    # Stage 1. configs
    # -------------------------------------------------
    runtime_cfg = full_cfg.runtime
    market_cfg = full_cfg.market
    actor_cfg = full_cfg.actor
    critic0_cfg = full_cfg.critic0
    critic_cfg = full_cfg.critic
    #
    seed = full_cfg.seed
    device = full_cfg.device
    torch_dtype = full_cfg.torch_dtype

    np.random.seed(seed)
    torch.manual_seed(seed)

    if option_model_type=='bs':
        save_dir = os.path.join(
            save_root,
            f"{option_model_type}_{str(market_cfg.sigma).replace('.', '_')}"
        )
    elif option_model_type=='heston':
        save_dir = os.path.join(
            save_root,
            f"{option_model_type}_V0{str(market_cfg.V0).replace('.', '_')}__"
            f"kappa{str(market_cfg.kappa).replace('.', '_')}__"
            f"theta{str(market_cfg.theta).replace('.', '_')}__"
            f"sigma{str(market_cfg.sigma).replace('.', '_')}__"
            f"rho{str(market_cfg.rho).replace('-', '_').replace('.', '_')}"
        )

    log_print = LogPrint(
        print_debug=runtime_cfg.print_debug,
        save_dir=os.path.join(save_dir, "logs"),
        prefix="init_train",
    )

    log_print.write(
        f"{'=' * 100}\n"
        f">>> TRAIN NEW {option_model_type} MODELS \n"
        f">>> {time.strftime("%Y%m%d_%H%M")}\n"
        f">>> Using device: {device}\n"
        f"{'=' * 100}"
    )

    # -------------------------------------------------
    # Stage 2. Initialize models (ONLY from ModelCFG)
    # -------------------------------------------------
    log_print.write(">>> Initializing models...")

    actor = Actor(
        cfgs= actor_cfg,
        device=device,
        torch_dtype=torch_dtype,
    )

    critic0 = Critic0(
        cfgs= critic0_cfg,
        device=device,
    )

    critic = Critic(
        cfgs=critic_cfg,
        device=device,
    )

    # target
    critic0_targ = Critic0(
        cfgs= critic0_cfg,
        device=device,
    )
    critic0_targ.load_state_dict(critic0.state_dict())
    critic_targ = Critic(
        cfgs=critic_cfg,
        device=device,
    )
    critic_targ.load_state_dict(critic.state_dict())
    actor_targ = Actor(
        cfgs= actor_cfg,
        device=device,
        torch_dtype=torch_dtype,
    )
    actor_targ.load_state_dict(actor.state_dict())
    targ_group = ModelsTargGroup(actor_targ, critic0_targ, critic_targ)

    # -------------------------------------------------
    # Stage 3. Initialize ReplayBuffer and ACTrainer
    # -------------------------------------------------

    T_steps = market_cfg.T_steps

    buffer = ReplayBuffer(
        critic0_state_dim=critic0.state_dim,
        collect_data_times=runtime_cfg.collect_data_times,
        a_grid_size=runtime_cfg.a_grid_size,
        s_over_k_steps=runtime_cfg.s_over_k_steps,
        T_steps=T_steps,
        device=device,
        dtype=torch_dtype,
    )

    trainer = ACTrainer(
        runtime_cfg,
        actor=actor,
        critic0=critic0,
        critic=critic,
        critic_targ_group=targ_group,
        buffer=buffer
    )
    # -------------------------------------------------
    # Stage 4. Stock Path Generator and Evaluator
    # -------------------------------------------------
    r_tensor = torch.full(
        fill_value=market_cfg.r, size=(T_steps,), device=device, dtype=torch_dtype)

    # Evaluator: StockPathGenerator, Path Generater
    stock_path_generator = StockPathGenerator(
        market_cfg = market_cfg,
        ref_paths = 50000,
        device = device,
    )
    # Evaluator: CrossSectionEvaluator
    evaluator = CrossSectionEvaluator(
        OPTION_MODEL_TYPE=option_model_type,
        market_cfg=market_cfg,
        runtime_cfg=runtime_cfg,
        r_tensor=r_tensor,
        actor=actor,
        critic0=critic0,
        critic=critic,
        stock_path_generator=stock_path_generator,
    )

    # -------------------------------------------------
    # Stage 5. Training
    # -------------------------------------------------

    all_risk_lambda_k = torch.tensor([critic0.risk_lambda, critic.risk_lambda], device=device, dtype=torch_dtype)

    tau_t_all_backward = (market_cfg.dt) * torch.arange(
        T_steps, 0 - 1, -1, device=device, dtype=torch_dtype)

    # -------------------------------------------------
    # Stage 5.1 backward time training...
    # -------------------------------------------------
    log_print.write(
        f"{'=' * 100}\n"
        f">>> Updating models via backward time training..."
        f"{'=' * 100}\n"
    )

    for t in range(T_steps - 1, -1, -1):
        #
        ratio_new = 0.5
        samples_N = 1024
        c_loops = None
        collect_data_times = runtime_cfg.collect_data_times
        #
        if (t == T_steps - 1) or (t % 10 == 0):
            use_evaluator = evaluator
        else:
            use_evaluator = None
        #
        train_one_step(
            device=device,
            torch_dtype=torch_dtype,
            t=t,
            T_steps=T_steps,
            runtime_cfg=runtime_cfg,
            market_cfg=market_cfg,
            log_print=log_print,
            buffer=buffer,
            r_tensor=r_tensor,
            tau_t_all_backward=tau_t_all_backward,
            all_risk_lambda_k=all_risk_lambda_k,
            targ_group=targ_group,
            stock_path_generator=stock_path_generator,
            trainer=trainer,
            evaluator=use_evaluator,
            #
            ratio_new=ratio_new,
            samples_N=samples_N,
            c_loops=c_loops,
            collect_data_times=collect_data_times,
            train_critic0=train_all,
            train_critic=train_all,
            train_actor=train_all
        )

    # -------------------------------------------------
    # Stage 5.2 random time training...
    # -------------------------------------------------
    log_print.write(
        f"{'=' * 100}\n"
        f">>> Updating models with random time training..."
        f"{'=' * 100}\n"
    )
    #
    remaining_train_times = (runtime_cfg.num_total_updates - 1) * T_steps
    #
    for it in range(remaining_train_times):
        #
        ratio_new = 0.1
        samples_N = 512
        c_loops = 0
        collect_data_times = 1
        #
        if (it == remaining_train_times - 1) or (it == 0) or (it % 50 == 0):
            use_evaluator = evaluator
        else:
            use_evaluator = None
        #
        t = random.randint(0, T_steps - 1)

        train_one_step(
            device=device,
            torch_dtype=torch_dtype,
            t=t,
            T_steps=T_steps,
            runtime_cfg=runtime_cfg,
            market_cfg=market_cfg,
            log_print=log_print,
            buffer=buffer,
            r_tensor=r_tensor,
            tau_t_all_backward=tau_t_all_backward,
            all_risk_lambda_k=all_risk_lambda_k,
            targ_group=targ_group,
            stock_path_generator=stock_path_generator,
            trainer=trainer,
            evaluator=use_evaluator,
            #
            ratio_new=ratio_new,
            samples_N=samples_N,
            c_loops=c_loops,
            collect_data_times=collect_data_times,
            train_critic0=train_all,
            train_critic=train_all,
            train_actor=train_all
        )

        # buffer.reset_all()
        # buffer.reset_size()
    # -------------------------------------------------
    # Stage 6. evaluator all
    # -------------------------------------------------

    evaluate_full_paths(
        market_cfg,
        option_model_type,
        stock_path_generator,
        actor, critic0, critic,
        n_paths=50,
        existing_path=True,

    )

    # -------------------------------------------------
    # Stage 7. Save
    # -------------------------------------------------
    save_full_experiment(
        log_print=log_print,
        save_dir=save_dir,
        market_cfg=market_cfg,
        critic0=critic0,
        critic0_cfg=critic0_cfg,
        runtime_cfg=runtime_cfg,
        actor_cfg=actor_cfg,
        actor=actor,
        critic_cfg=critic_cfg,
        critic=critic,
        stock_path_generator=stock_path_generator,
    )

    # Train Finish
    log_print.write("Training finished.")
    log_print.close()

def clear():
    import gc
    torch.mps.empty_cache()
    gc.collect()

if __name__ == "__main__":
    OPTION_MODEL_TYPE = 'bs'
    from configs.full_bs_config import full_cfg as full_cfg
    train_new_models(full_cfg, OPTION_MODEL_TYPE)


    # OPTION_MODEL_TYPE = 'heston'
    # from configs.full_heston_config_1 import full_cfg as full_cfg
    # train_new_models(full_cfg, OPTION_MODEL_TYPE)
