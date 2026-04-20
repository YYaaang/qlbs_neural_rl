# ===================== 统一 Trainer =====================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from configs.specifications.actor_cfg import ActorCFG
from configs.specifications.critic_cfg import CriticCFG

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

torch_dtype = torch.float32 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else torch.float


class Actor(nn.Module):
    """
    Squashed Gaussian Actor:
    """

    def __init__(self,
                 cfgs: ActorCFG,
                 #
                 device=device,
                 torch_dtype=torch_dtype
                 ):
        super().__init__()
        self.risk_lambda = cfgs.risk_lambda
        self.state_dim = cfgs.state_dim
        self.net = build_mlp(
            cfgs.state_dim, cfgs.hidden, out_dim=2, device=device
        )
        self.log_std_min = cfgs.log_std_min
        self.log_std_max = cfgs.log_std_max
        self.min_sigma = cfgs.min_sigma

        self.device = device
        self.torch_dtype = torch_dtype

        # 动作边界（可在训练中动态更新）
        self.register_buffer('a_min_buf', torch.tensor(float(cfgs.a_min), device=device, dtype=torch_dtype))
        self.register_buffer('a_max_buf', torch.tensor(float(cfgs.a_max), device=device, dtype=torch_dtype))

    # ---- 工具：把 (-1,1) 映射到 [a_min, a_max]，以及反向映射 ----
    def _squash_to_bounds(self, a_tanh):
        # a_tanh ∈ (-1, 1)  ->  a ∈ [a_min, a_max]
        a_min, a_max = self.a_min_buf, self.a_max_buf
        return 0.5 * (a_max - a_min) * (a_tanh + 1.0) + a_min

    def _unsquash_from_bounds(self, a):
        # a ∈ [a_min, a_max] -> a_tanh ∈ (-1, 1)
        a_min, a_max = self.a_min_buf, self.a_max_buf
        a_tanh = (a - a_min) / (0.5 * (a_max - a_min)) - 1.0
        return a_tanh.clamp(-0.999999, 0.999999)

    def forward(self, s):
        """返回 (mu, log_std)，定义在未压缩空间（tanh 之前的 u 空间）"""
        if s.shape[-1] == self.state_dim + 1:
            x_tau = s[..., :2]
        else:
            x_tau = s

        out = self.net(x_tau)  # [..., n]
        mu = out[..., 0]
        log_std = out[..., 1].clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def mean(self, s):
        """
        确定性动作：a_det = squash(mu) -> 映射到 [a_min, a_max]
        注意：不是严格意义上的 E[a]，但工程里常用作确定性代表。
        """
        mu, _ = self(s)
        a_tanh = torch.tanh(mu)
        return self._squash_to_bounds(a_tanh)

    def sample(self, s, return_pre_tanh=False):
        """
        重参数化采样：
          u = mu + std * eps
          a_tanh = tanh(u)
          a = squash_to_bounds(a_tanh)
        同时返回 log_prob(a)（带 tanh 雅可比修正，仍是对未映射[-1,1]空间的密度）。
        """
        mu, log_std = self(s)
        std = log_std.exp().clamp_min(self.min_sigma)
        dist = Normal(mu, std)

        # reparameterization trick
        u = dist.rsample()  # 未压缩空间样本
        a_tanh = torch.tanh(u)  # (-1,1)
        a = self._squash_to_bounds(a_tanh)  # [a_min, a_max]

        # log_prob 修正（tanh 的雅可比）
        # 对 u 的密度：log N(u; mu, std)
        logp_u = dist.log_prob(u)

        # tanh 的雅可比：da_tanh/du = 1 - tanh(u)^2
        # 对多维动作需要 sum；这里动作是一维，所以不需要再 sum(-1)
        # 稳定处理：加上一个很小 eps 防止 log(0)
        eps = 1e-6
        log_det_jacobian = torch.log(1.0 - a_tanh.pow(2) + eps)

        # a_tanh 的 log_prob（未映射到 [a_min, a_max] 的那步）
        logp_a_tanh = logp_u - log_det_jacobian

        # 由于之后是线性映射到 [a_min, a_max]，线性变换的雅可比是常数
        # scale = (a_max - a_min)/2；加上常数项对优化只差一个常数，可以省略
        # 若你要绝对数值，可在此再减去 log(scale)
        # scale = 0.5 * (self.a_max_buf - self.a_min_buf)
        # logp_a = logp_a_tanh - torch.log(scale + eps)
        logp_a = logp_a_tanh  # 常数项省略

        if return_pre_tanh:
            return a, logp_a, u
        return a, logp_a

    @torch.no_grad()
    def sample_n(self, s, num_actions, mode="quantile", k_sigma=2.5):
        """
        为每个状态一次性构造一个动作“网格”：shape [batch, num_actions]
        - mode="quantile": 用 z∈linspace(-k,k) 的确定性节点；更稳定可复现
        - mode="sample":   真随机采样
        返回：a_grid（已在 [a_min, a_max] 内）
        """
        mu, log_std = self(s)
        std = log_std.exp().clamp_min(self.min_sigma)

        if mode == "quantile":
            z = torch.linspace(-k_sigma, k_sigma, steps=num_actions, device=self.device, dtype=self.torch_dtype)  # [A]
            u = mu.unsqueeze(-1) + std.unsqueeze(-1) * z.view(1, -1)  # [B, A]
        elif mode == "sample":
            z = torch.randn(s.size(0), num_actions, device=self.device, dtype=self.torch_dtype)  # [B, A]
            u = mu.unsqueeze(-1) + std.unsqueeze(-1) * z  # [B, A]
        else:
            raise ValueError(f"Unknown mode={mode}")

        a_tanh = torch.tanh(u)  # [B, A]
        a = self._squash_to_bounds(a_tanh)  # [B, A]
        return a

    # ---- 可选：暴露一个接口，动态调整边界（比如随训练退火） ----
    def set_action_bounds(self, a_min: float, a_max: float):
        with torch.no_grad():
            self.a_min_buf.fill_(float(a_min))
            self.a_max_buf.fill_(float(a_max))


class Critic0(
    nn.Module,
):
    """
    λ = 0
    Q(s,a) = w0(s) + w1(s)*a

    """

    def __init__(
            self,
            cfgs: CriticCFG,
            #
            device=device
    ):
        super().__init__()
        self.model_type = 'critic0'
        self.risk_lambda = cfgs.risk_lambda
        self.state_dim = cfgs.state_dim
        head_dim = cfgs.head_dim

        self.eps_curv = cfgs.eps_curv

        # ---------------------------
        #  Two completely separate MLPs
        # ---------------------------
        self.backbone_base = build_mlp(
            in_dim=cfgs.state_dim,
            hidden_dims=cfgs.hidden,
            out_dim=head_dim,
            device=device
        )

        # ---------------------------
        #  Heads for w^(0) —— base backbone
        # ---------------------------
        self.w0_head = nn.Linear(head_dim, 1, device=device)
        self.w1_head = nn.Linear(head_dim, 1, device=device)
        # init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)

    def heads(self, s_full):
        """
        s_full: [..., 2] = [log(S/K), tau]
        """

        # -------- base backbone (no λ info) --------
        h_base = self.backbone_base(s_full)  # [..., head_dim]
        w0 = self.w0_head(h_base).squeeze(-1)  # w0^(0)
        w1 = self.w1_head(h_base).squeeze(-1)  # w1^(0)

        return w0, w1

    def forward(self, s_full, a):
        w0, w1 = self.heads(s_full)
        return w0 + w1 * a


class Critic(nn.Module):
    """
    Q(s,a) = w0(s) + w1(s)*a + 0.5*w2(s)*a^2
    """

    def __init__(self,
                 cfgs: CriticCFG,
                 #
                 device=device,
                 ):
        super().__init__()
        self.model_type = 'critic'
        self.risk_lambda = cfgs.risk_lambda
        self.state_dim = cfgs.state_dim
        head_dim = cfgs.head_dim
        #
        self.eps_curv = cfgs.eps_curv

        # ---------------------------
        # ---------------------------
        self.backbone_full = build_mlp(
            in_dim=cfgs.state_dim,
            hidden_dims=cfgs.hidden,
            out_dim=head_dim,
            device=device
        )

        # ---------------------------
        #  Heads for w^(1) —— full backbone
        # ---------------------------
        self.w0_head_1 = nn.Linear(head_dim, 1, device=device)
        self.w1_head_1 = nn.Linear(head_dim, 1, device=device)
        self.w2_head_1 = nn.Linear(head_dim, 1, device=device)

        # init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)

    def heads(self, s_full):
        """
        s_full: [..., 2] = [log(S/K), tau]
        """
        # -------- full backbone (with λ info) --------
        h_full = self.backbone_full(s_full)  # [..., head_dim]
        w0_1 = self.w0_head_1(h_full).squeeze(-1)  # w0^(1)
        w1_1 = self.w1_head_1(h_full).squeeze(-1)  # w1^(1)
        w2_1 = self.w2_head_1(h_full).squeeze(-1)  # w2^(1)
        w2 = -  (F.softplus(w2_1) + self.eps_curv)

        return w0_1, w1_1, w2

    def forward(self, s_full, a):
        w0, w1, w2 = self.heads(s_full)
        return w0 + w1 * a + 0.5 * w2 * (a * a)

    @torch.no_grad()
    def a_star(self, s_full, clamp_bounds=None, eps=1e-6):
        w0, w1, w2 = self.heads(s_full)
        w2_safe = torch.where(w2 < -eps, w2, -eps)
        a = -w1 / w2_safe
        if clamp_bounds is not None:
            a = a.clamp(*clamp_bounds)
        return a

# ===================== Actor-Critic网络（连续动作+QLBS Q函数）=====================
def build_mlp(
        in_dim,
        hidden_dims=(128, 128),
        out_dim=1,
        activation=nn.ReLU,
        device=device
):
    layers = []
    last = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(last, h, device=device), activation()]
        last = h
    layers += [nn.Linear(last, out_dim, device=device)]
    return nn.Sequential(*layers).to(device)

# =====================================================
#   精确匹配函数：将 lambda_raw → lambda_norm
# =====================================================
def map_lambda(lambda_raw):
    """

    """
    log_lam = torch.log(
        torch.where(lambda_raw < 1e-8, lambda_raw.max(), lambda_raw)
    )
    return log_lam / torch.abs(log_lam).max()


class ModelsTargGroup:
    def __init__(
            self,
            actor_targ: Actor,
            critic0_targ: Critic0,
            critic_targ:Critic):
        self.actor_targ = actor_targ
        self.critic0_targ = critic0_targ
        self.critic_targ = critic_targ


def extract_theta_from_outputs(
    u0_0, u0_1,
    u1_0, u1_1, u1_2,
    risk_lambda: float,
):
    """
    Extract risk-sensitive structural coefficients Θ from critic outputs.

    Parameters
    ----------
    u0_0, u0_1, u0_2 : torch.Tensor or float
        Outputs from critic-0 (λ = 0), corresponding to U^{(0)}, U^{(1)}, U^{(2)}.
        Typically u0_2 = 0.
    u1_0, u1_1, u1_2 : torch.Tensor or float
        Outputs from critic-λ, corresponding to U^{(0)}, U^{(1)}, U^{(2)}.
    risk_lambda : float
        Risk aversion parameter λ.

    Returns
    -------
    theta0, theta1, theta2 : same type as inputs
        Structural risk coefficients Θ^{(0)}, Θ^{(1)}, Θ^{(2)}.
    """
    if risk_lambda <= 0:
        raise ValueError("risk_lambda must be positive")

    theta0 = (u0_0 - u1_0) / risk_lambda
    theta1 = (u1_1 - u0_1) / risk_lambda
    theta2 = u1_2 / risk_lambda

    return theta0, theta1, theta2

def q_diff_critic_risk_lambda(
        critic0: Critic0,
        critic: Critic,
        actions: torch.Tensor,
        actor_risk_lambda:float,
        all_critic_risk_lambda: list,
        state: torch.Tensor,
):
    import numpy as np
    all_critic_risk_lambda = np.array(all_critic_risk_lambda)
    Q0 = critic0(state, actions)
    Q = critic(state, actions)

    risk_lambda = torch.tensor(all_critic_risk_lambda, device=Q.device, dtype=Q.dtype).view(-1, 1, 1)

    Q_all = Q0 + (Q - Q0) / actor_risk_lambda * risk_lambda
    return Q_all

def q_diff_risk_lambda(
        actions,
        u0_0, u0_1,
        u1_0, u1_1, u1_2,
        actor_lambda: float,
        risk_lambda: list,
):
    risk_lambda = torch.tensor(risk_lambda, device=u0_0.device, dtype=u0_0.dtype)
    # 向量化计算，自动广播，无循环
    risk_lambda = risk_lambda.view(-1, 1, 1)
    term1 = u0_0 - (u0_0 - u1_0) / actor_lambda * risk_lambda
    term2 = u0_1 + (u1_1 - u0_1) / actor_lambda * risk_lambda
    term3 = 0.5 * u1_2 / actor_lambda * risk_lambda

    q = term1 + term2 * actions + term3 * actions

    return q
