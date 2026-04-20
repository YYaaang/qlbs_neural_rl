# ============================================================
# init_train_models.py
# ============================================================
import time
import torch
import numpy as np
import os
import random

from configs.base.env_cfg import seed, device, torch_dtype
from src.rl_models import Critic, ModelsTargGroup
from src.trainer import ACTrainer
from src.buffer import ReplayBuffer
from src.data_buffer_processing import collect_data_at_t
from src.bs_model import BSConditionalSampler as StockPathGenerator

from utils.log_print import LogPrint
from utils.analysis.evaluation import CrossSectionEvaluator, evaluate_full_paths
from utils.train_one_step import train_one_step
from utils.model_saver import save_critic_lambda, load_full_experiment, generate_policy_cfg
from configs.full_bs_config import FullConfig as FullConfig
from configs.specifications.critic_cfg import CriticCFG



def train_new_critic_from_existing_actor_and_critic0(
        device, torch_dtype,
        option_model_type:str,
        load_dir: str,
        actor_risk_lambda: float,
        new_critic_lambda: float,
        print_debug = True,
        seed = seed,
):
    only_critic = True

    # -------------------------------------------------
    # Stage 1. Runtime & basic configs
    # -------------------------------------------------
    if option_model_type == 'heston':
        from src.heston_model import HestonConditionalSampler as StockPathGenerator
    elif option_model_type == 'bs':
        from src.bs_model import BSConditionalSampler as StockPathGenerator
    else:
        raise NameError('OPTION_MODEL_TYPE is wrong')

    log_print = LogPrint(
        print_debug=print_debug,
        save_dir=os.path.join(load_dir, "logs"),
        prefix=f"train_critic_lambda_{new_critic_lambda}",
    )

    np.random.seed(seed)
    torch.manual_seed(seed)

    log_print.write(
        f"{'=' * 100}\n"
        f">>> TRAIN NEW CRITIC FROM EXISTING EXPERIMENT\n"
        f">>> new critic lambda = {new_critic_lambda}\n"
        f">>> {time.strftime('%Y%m%d_%H%M')}\n"
        f">>> Using device: {device}\n"
        f"{'=' * 100}"
    )
    # -------------------------------------------------
    # Stage 2. Load full experiment (CRITICAL STEP)
    # -------------------------------------------------
    log_print.write(">>> Loading full experiment artifact")
    (
        critic0,
        market_cfg,
        critic0_cfg,
        runtime_cfg,   # NOTE: 当前未使用，仅保留
        actor,
        actor_cfg,
        critic,
        critic_cfg,
    ) = load_full_experiment(
        log_print=log_print,
        load_dir=load_dir,
        device=device,
        torch_dtype=torch_dtype,
        actor_lambda=actor_risk_lambda,
        critic_lambda=actor_risk_lambda
    )
    del critic
    # -------------------------------------------------
    # Stage 3. Freeze actor & critic0
    # -------------------------------------------------
    log_print.write(">>> Freezing actor and critic0")

    for p in actor.parameters():
        p.requires_grad_(False)
    actor.eval()

    for p in critic0.parameters():
        p.requires_grad_(False)
    critic0.eval()
    # -------------------------------------------------
    # Stage 4. Create new CriticCFG and Critic
    # -------------------------------------------------
    log_print.write(">>> Initializing new critic")
    critic_cfg = CriticCFG(  # new
            level="critic",
            form=critic_cfg.form,
            risk_lambda=new_critic_lambda,
            state_dim=critic_cfg.state_dim,
            hidden=critic_cfg.hidden,
            head_dim=critic_cfg.head_dim
        )
    critic = Critic(
        cfgs=critic_cfg,
        device=device,
    )
    critic_targ = Critic(
        cfgs=critic_cfg,
        device=device,
    )
    critic_targ.load_state_dict(critic.state_dict())
    targ_group = ModelsTargGroup(
        actor_targ=actor,        # frozen
        critic0_targ=critic0,    # frozen
        critic_targ=critic_targ,
    )
    # -------------------------------------------------
    # Stage 5. Initialize ReplayBuffer and ACTrainer
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
    log_print.write(">>> Start training new critic")
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
        ratio_new = 0.45
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
            train_critic0 = not only_critic,
            train_critic = only_critic,
            train_actor = not only_critic
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
            train_critic0 = not only_critic,
            train_critic = only_critic,
            train_actor = not only_critic
        )

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
    log_print.write(">>> Saving new critic")

    save_critic_lambda(
        log_print=log_print,
        save_dir=load_dir,
        actor_lambda=actor_cfg.risk_lambda,
        critic_cfg=critic_cfg,
        critic=critic,
    )
    log_print.write(f"Critic saved (λ={critic_cfg.risk_lambda})")

    generate_policy_cfg(
        log_print=log_print,
        dir=load_dir,
        actor_lambda=actor_cfg.risk_lambda,
    )
    log_print.write("PolicyCFG generated")

    log_print.write(">>> New critic training finished")
    log_print.close()

def clear():
    import gc
    torch.mps.empty_cache()
    gc.collect()

if __name__ == "__main__":
    all_new_L = [4000, 9000, 15000, 20000, 10000000.0]
    OPTION_MODEL_TYPE = 'bs'
    for L in all_new_L:
        train_new_critic_from_existing_actor_and_critic0(
            device=device,
            torch_dtype=torch_dtype,
            option_model_type=OPTION_MODEL_TYPE,
            #
            load_dir="models/4_2_bs_0_2",   #
            actor_risk_lambda=100_000,
            new_critic_lambda=L,
            #
            seed=42,
        )
        clear()
    #
    # OPTION_MODEL_TYPE = 'heston'
    # for L in all_new_L:
    #     train_new_critic_from_existing_actor_and_critic0(
    #         device=device,
    #         torch_dtype=torch_dtype,
    #         option_model_type=OPTION_MODEL_TYPE,
    #         #
    #         load_dir="models/heston_V00_04__kappa1_5__theta0_04__sigma0_25__rho_0_5__trans_cost0_000_1",   #
    #         actor_risk_lambda=100_000,
    #         new_critic_lambda=L,
    #         #
    #         seed=42,
    #     )
    #     clear()