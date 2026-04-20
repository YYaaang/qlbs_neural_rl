import torch
import torch.nn.functional as F

from src.rl_models import Critic0, Critic, Actor, ModelsTargGroup
from src.train_state import TrainState
from src.buffer import ReplayBuffer
from utils.log_print import LogPrint

from configs.base.runtime_cfg import RuntimeCFG

class ACTrainer:

    # ================================================================
    # 初始化
    # ================================================================
    def __init__(
        self,
        cfg: RuntimeCFG,
        *,
        actor: Actor,
        critic0: Critic0,
        critic: Critic,
        critic_targ_group: ModelsTargGroup,
        buffer: ReplayBuffer,
    ):
        self.cfg = cfg

        # ---- 注册模型 ----
        self.actor = actor
        self.critic0 = critic0
        self.critic = critic
        self.critic_targ_group = critic_targ_group
        self.buffer = buffer

        # ---- Optimizer ----
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic0_opt = torch.optim.Adam(self.critic0.parameters(), lr=cfg.lr_critic0)
        self.critic_opt  = torch.optim.Adam(self.critic.parameters(),  lr=cfg.lr_critic)

        # ---- Scheduler（ReduceLROnPlateau）----
        self.actor_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_opt,
            mode='min',
            factor=cfg.sched_factor,
            patience=cfg.sched_patience,
            threshold=cfg.sched_threshold,
            threshold_mode=cfg.sched_threshold_mode,
            cooldown=cfg.sched_cooldown,
            min_lr=cfg.sched_min_lr,
        )

        self.critic0_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.critic0_opt,
            mode='min',
            factor=cfg.sched_factor,
            patience=cfg.sched_patience,
            threshold=cfg.sched_threshold,
            threshold_mode=cfg.sched_threshold_mode,
            cooldown=cfg.sched_cooldown,
            min_lr=cfg.sched_min_lr,
        )

        self.critic_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_opt,
            mode='min',
            factor=cfg.sched_factor,
            patience=cfg.sched_patience,
            threshold=cfg.sched_threshold,
            threshold_mode=cfg.sched_threshold_mode,
            cooldown=cfg.sched_cooldown,
            min_lr=cfg.sched_min_lr,
        )

        # ---- TrainState（三套独立状态机）----
        self.state_actor = TrainState(
            ema_beta=cfg.ema_beta,
            patience=cfg.stable_patience,
            warmup_ignore=cfg.warmup_ignore,
        )

        self.state_critic0 = TrainState(
            ema_beta=cfg.ema_beta,
            patience=cfg.stable_patience,
            warmup_ignore=cfg.warmup_ignore,
        )

        self.state_critic = TrainState(
            ema_beta=cfg.ema_beta,
            patience=cfg.stable_patience,
            warmup_ignore=cfg.warmup_ignore,
        )

        self.max_grad_norm = cfg.max_grad_norm
        self.print_debug = cfg.print_debug


    # ================================================================
    # 单步 Critic 更新（Critic0 与 Critic 复用）
    # ================================================================
    def _critic_single_step(self, model, optimizer, S_input, A_input, Y_target):
        model.train()

        q_pred = model(S_input, A_input)
        loss_q = F.mse_loss(q_pred, Y_target)  # Y_target.cpu().detach().numpy()

        optimizer.zero_grad(set_to_none=True)
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        optimizer.step()

        return float(loss_q.item())


    # ================================================================
    # Actor：C‑stage（监督）到 a*
    # ================================================================
    def _actor_supervised_C(self, s_batch):
        self.actor.train()

        with torch.no_grad():
            a_star = self.critic.a_star(
                s_batch,
                clamp_bounds=(
                    float(self.actor.a_min_buf.item()),
                    float(self.actor.a_max_buf.item()),
                )
            )
            eps = 0.01 * (self.actor.a_max_buf - self.actor.a_min_buf)
            # eps = 0.0001 * (self.actor.a_max_buf - self.actor.a_min_buf)
            a_star = a_star + eps * torch.randn_like(a_star)

        a_pred = self.actor.mean(s_batch)

        polyak = 0.8
        a_tgt = polyak * a_star + (1 - polyak) * a_pred.detach()

        loss_raw = F.mse_loss(a_pred, a_tgt)
        loss = 0.1 * loss_raw

        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        return float(loss.item())


    # ================================================================
    # Actor：DPG stage
    # ================================================================
    def _actor_DPG(self, s_batch):
        self.actor.train()

        s_in = s_batch.detach()
        a_det = self.actor.mean(s_in)
        a_det = a_det + 1e-4 * torch.randn_like(a_det)

        q_val = self.critic(s_in, a_det)
        loss_pi = -q_val.mean()

        self.actor_opt.zero_grad(set_to_none=True)
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        return float(loss_pi.item())


    # ================================================================
    # Actor —— 单步融合更新（C + DPG）
    # ================================================================
    def _actor_update_single_step(self, s_t0_all, c_loops:int = None):
        if c_loops is not None:
            c_loops = c_loops
            dpg_loops = 1
        elif not self.state_critic.is_stable:
            c_loops = 2
            dpg_loops = 1
        else:
            c_loops = 1
            dpg_loops = 2

        c_losses = [self._actor_supervised_C(s_t0_all) for _ in range(c_loops)]
        dpg_losses = [self._actor_DPG(s_t0_all) for _ in range(dpg_loops)]

        return float((sum(c_losses) + sum(dpg_losses)) / (c_loops + dpg_loops))


    # ================================================================
    # train_one_timestep —— 三阶段结构 + 独立 early stop
    # ================================================================
    def train_one_timestep(
            self,
            *,
            t_step:int,
            max_iters_critic0:int,
            min_iters_critic0:int,
            max_iters_critic:int,
            min_iters_critic:int,
            max_iters_actor:int,
            min_iters_actor:int,
            log_print:LogPrint,
            sched_every:int=1,
            #
            samples_N:int=100,
            ratio_new:float=0.1,
            c_loops:int=None,
            #

    ):
        # -------- reset TrainState（关键要求）--------
        self.state_critic0.reset()
        self.state_critic.reset()
        self.state_actor.reset()

        # =====================================================
        # 1) Critic0
        # =====================================================
        ret_c0 = None
        for it in range(max_iters_critic0):
            if it == 140:
                debug = 1
            S_input, A_input, Y_target = self.buffer.sample_lambda0_mix(t_step=t_step, N=samples_N, ratio_new=ratio_new)
            loss_q = self._critic_single_step(self.critic0, self.critic0_opt,
                                              S_input, A_input, Y_target)

            ema = self.state_critic0.update(loss_q)

            if (it + 1) % sched_every == 0:
                self.critic0_sched.step(loss_q)

            ret_c0 = {
                "loss_last": loss_q,
                "ema": self.state_critic0.ema,
                "best_ema": self.state_critic0.best_ema,
                "streak": self.state_critic0.streak,
                "is_stable": self.state_critic0.is_stable,
            }

            if (it + 1) % self.cfg.log_every == 0:
                self._log_step('critic0', it + 1, loss_q, self.state_critic0, self.critic0_opt, log_print)

            if self.state_critic0.is_stable and it >= min_iters_critic0:
                self._log_step('STABLE|critic0', it + 1, loss_q, self.state_critic0, self.critic0_opt, log_print)
                break
        if max_iters_critic0 > 0:
            del S_input, A_input, Y_target
            self.critic_targ_group.critic0_targ.load_state_dict(self.critic0.state_dict())
        # =====================================================
        # 2) Critic
        # =====================================================
        ret_c = None
        for it in range(max_iters_critic):
            # if it == 100:
            #     print()
            S_input, A_input, Y_target = self.buffer.sample_lambdaN_mix(t_step=t_step, N=samples_N, ratio_new=ratio_new)
            loss_q = self._critic_single_step(self.critic, self.critic_opt,
                                              S_input, A_input, Y_target)

            ema = self.state_critic.update(loss_q)
            if (it + 1) % sched_every == 0:
                self.critic_sched.step(loss_q)

            ret_c = {
                "loss_last": loss_q,
                "ema": self.state_critic.ema,
                "best_ema": self.state_critic.best_ema,
                "streak": self.state_critic.streak,
                "is_stable": self.state_critic.is_stable,
            }

            if (it + 1) % self.cfg.log_every == 0:
                self._log_step('critic', it + 1, loss_q, self.state_critic, self.critic_opt, log_print)

            if self.state_critic.is_stable and it >= min_iters_critic:
                self._log_step('STABLE|critic', it + 1, loss_q, self.state_critic, self.critic_opt, log_print)
                break
        if max_iters_critic > 0:
            del S_input, A_input, Y_target
            self.critic_targ_group.critic_targ.load_state_dict(self.critic.state_dict())
        # =====================================================
        # 3) Actor
        # =====================================================
        ret_a = None
        for it in range(max_iters_actor):

            s_t0_all = self.buffer.sample_actor_lambdaN_min(t_step=t_step, N=samples_N, ratio_new=ratio_new)

            loss_pi = self._actor_update_single_step(s_t0_all, c_loops=c_loops)

            ema = self.state_actor.update(loss_pi)
            if (it + 1) % sched_every == 0:
                self.actor_sched.step(loss_pi)

            ret_a = {
                "loss_last": loss_pi,
                "ema": self.state_actor.ema,
                "best_ema": self.state_actor.best_ema,
                "streak": self.state_actor.streak,
                "is_stable": self.state_actor.is_stable,
            }

            if (it + 1) % self.cfg.log_every == 0:
            # if (it + 1) % 1 == 0:
                self._log_step('actor', it + 1, loss_pi, self.state_actor, self.actor_opt, log_print)


            if self.state_actor.is_stable and it >= min_iters_actor:
                self._log_step('STABLE|actor', it + 1, loss_pi, self.state_actor, self.actor_opt, log_print)
                break
        if max_iters_actor > 0:
            del s_t0_all
            self.critic_targ_group.actor_targ.load_state_dict(self.actor.state_dict())

        return {
            "critic0": ret_c0,
            "critic": ret_c,
            "actor": ret_a,
        }

    def _log_step(self, module_name: str, it: int, loss: float, state: TrainState, optimizer, log_print):
        lr = optimizer.param_groups[0]['lr']
        p = (
            f"[{module_name}] it={it:03d} | "
            f"loss={loss:.6f} | ema={state.ema:.6f} | best={state.best_ema:.6f} | "
            f"streak={state.streak}/{state.patience} | "
            f"stable={'Y' if state.is_stable else 'N'} | lr={lr:.2e}"
        )
        log_print.write(p)
