# fast_group_replay_buffer.py
import torch

def auto_ceil(num):
    import math
    if num == 0:
        return 0
    # 自动计算最高位的单位：100, 1000, 10000...
    digit = 10 ** (len(str(abs(num))) - 1)
    # 向上取整
    return math.ceil(num / digit) * digit

class ReplayBuffer:
    @torch.no_grad()
    def __init__(
            self,
            critic0_state_dim,
            collect_data_times,
            a_grid_size,
            s_over_k_steps,
            T_steps,
            device, dtype=torch.float32
    ):

        self.critic0_state_dim = critic0_state_dim

        group_size = collect_data_times * s_over_k_steps

        max_size =  group_size * (T_steps + 1)

        self.max_size = max_size
        self.group_size = group_size

        self.a_grid_size = a_grid_size

        self.device = device
        self.dtype = dtype

        self.reset_all()

    @torch.no_grad()
    def reset_all(self):

        self.S = torch.empty((self.max_size, self.critic0_state_dim), dtype=self.dtype, device=self.device)
        self.A = torch.empty((self.max_size, self.a_grid_size), dtype=self.dtype, device=self.device)

        # self.size = 0
        # self.ptr = 0

        # ===== λ = 0 buffer =====
        self.Y0 = torch.empty((self.max_size, self.a_grid_size), dtype=self.dtype, device=self.device)
        # ===== λ != 0 buffer =====
        self.YN = torch.empty((self.max_size, self.a_grid_size), dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def reset_size(self):
        # self.size = 0
        # self.ptr = 0
        return

    # =====================================================================
    # add_from_tensor —— 一次性展成 group-level 并批量写入
    # =====================================================================
    @torch.no_grad()
    def add_from_tensor(
            self,
            t_step:int,
            #
            S_input,  # [N0, 2]
            A_input,  # [N1, A_dim]
            #
            Y0_input,  # [N0, A_dim]
            #
            Y_input,  # [N1, A_dim]
    ):
        """
        - S_input
        - A0_input / Y0_input 对应 λ=0。
        - A_input / Y_input 对应 λ!=0。
        """
        N = S_input.shape[0]

        # new_idx = slice(self.ptr, self.ptr + N)
        start_index = (t_step - 2) * self.group_size
        new_idx = slice(start_index, start_index + N)

        self.S[new_idx] = S_input  # [N0, 2]
        self.A[new_idx] = A_input  # [N1, A_dim]

        # self.ptr += N
        # self.size = max(self.size, self.ptr)

        # ----------------------- λ = 0 批量写入 -----------------------
        # 直接连续写入
        if Y0_input.numel() > 0:
            self.Y0[new_idx] = Y0_input  # [N0, A_dim]

        # ----------------------- λ != 0 批量写入 -----------------------
        self.YN[new_idx] = Y_input  # [N1, A_dim]
        return

    # =====================================================================
    # 采样逻辑：按 t_max 分出 recent / history（无 mask-filter 循环）
    # =====================================================================
    def _sample_index(self,t_step, N, ratio_new):
        #
        start_index = (t_step - 2) * self.group_size
        end_index = (t_step -1) * self.group_size

        # size = self.size
        # group_size = self.group_size

        # if size == 0:
        #     return None, None, None

        recent_idx = torch.arange(start_index, end_index, device=self.device)
        history_idx = torch.arange(0, start_index, device=self.device)

        k_old = min(int(N * (1 - ratio_new)), history_idx.numel())
        k_new = N - k_old

        # random selection
        if k_old > 0:
            old_sel = history_idx[torch.randint(0, history_idx.numel(), (k_old,), device=self.device)]
        else:
            old_sel = torch.empty((0,), dtype=torch.int, device=self.device)

        new_sel = recent_idx[torch.randint(0, recent_idx.numel(), (k_new,), device=self.device)]

        sel = torch.cat([new_sel, old_sel], dim=0)
        return sel

    @torch.no_grad()
    def _sample_from(self, t_step, S_buf, A_buf, Y_buf, N, ratio_new):

        sel = self._sample_index(t_step, N, ratio_new)

        return (S_buf[sel].unsqueeze(1).expand(-1, A_buf.shape[1], -1),
                A_buf[sel],
                Y_buf[sel])

    # λ = 0
    def sample_lambda0_mix(self, t_step, N, ratio_new=0.1):
        return self._sample_from(t_step, self.S, self.A, self.Y0, N, ratio_new)

    # λ != 0
    def sample_lambdaN_mix(self, t_step, N, ratio_new=0.1):
        return self._sample_from(t_step, self.S, self.A, self.YN, N, ratio_new)

    def sample_actor_lambdaN_min(self, t_step, N, ratio_new=0.1):
        sel = self._sample_index(t_step, N, ratio_new)
        data = self.S[sel]
        return data
