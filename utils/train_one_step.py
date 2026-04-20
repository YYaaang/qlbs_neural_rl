import time
import numpy as np
from src.data_buffer_processing import collect_data_at_t

def train_one_step(
        device,
        torch_dtype,
        t,
        T_steps,
        runtime_cfg,
        market_cfg,
        log_print,
        buffer,
        r_tensor,
        tau_t_all_backward,
        all_risk_lambda_k,
        targ_group,
        stock_path_generator,
        trainer,
        *,
        evaluator = None,
        #
        ratio_new,
        samples_N,
        c_loops,
        collect_data_times,
        train_critic0=True,
        train_critic=True,
        train_actor=True,
):

    time_t0 = time.time()
    log_print.write(
        f"{'=' * 100} \n"
        f"{'=' * 22}           Training critic λ={trainer.critic.risk_lambda:.4f} | t:{t: 3d} ｜ pct:{((1 - t / T_steps) * 100): 6.2f} %       {'=' * 22} \n"
        f"{'=' * 100} \n"
    )

    t_step = T_steps - t + 1
    tau_t_local_backward = tau_t_all_backward[-t_step:]

    collect_data_at_t(
        device=device, torch_dtype=torch_dtype,
        path_generator=stock_path_generator,
        marketcfg=market_cfg,
        runtime_cfg=runtime_cfg,
        buffer=buffer,
        t=t,
        t_step=t_step,
        total_updates=collect_data_times,
        tau_t_local_backward=tau_t_local_backward,
        r_tensor=r_tensor,
        all_risk_lambda_k=all_risk_lambda_k,
        targ_group=targ_group,
        only_critic= not train_critic0
    )

    # === 单时间步完整训练 ===
    ret = trainer.train_one_timestep(
        t_step=t_step,
        max_iters_critic0=runtime_cfg.max_iters_critic0 if train_critic0 else 0,
        min_iters_critic0=runtime_cfg.min_iters_critic0 if train_critic0 else 0,
        max_iters_critic=runtime_cfg.max_iters_critic if train_critic else 0,
        min_iters_critic=runtime_cfg.min_iters_critic if train_critic else 0,
        max_iters_actor=runtime_cfg.max_iters_actor if train_actor else 0,
        min_iters_actor=runtime_cfg.min_iters_actor if train_actor else 0,
        sched_every=runtime_cfg.sched_every,
        log_print=log_print,
        #
        samples_N=samples_N,
        ratio_new=ratio_new,
        c_loops=c_loops,
    )

    # -------------------------------------------------
    # Stage 5.1 - evaluator
    # -------------------------------------------------
    if evaluator:
        df = evaluator(t=t, tau_all=tau_t_local_backward, tau_t=tau_t_local_backward[0])

        print_i = np.linspace(1, len(df) - 2, int(len(df) / 2), dtype=int)
        log_print.write(
            df.iloc[print_i].map(lambda x: "{0:.4f}".format(x)).to_string()
        )

    time_t2 = time.time()
    log_print.write(
        f"{'=' * 100}\n"
        f'----- t{t} end ----- total time spent:{(time_t2 - time_t0): .2f} ----- t{t} end -----'
        f"{'=' * 100}\n"
    )