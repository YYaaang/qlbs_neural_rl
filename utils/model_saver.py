import os
import json
import time

import numpy as np
import torch
from typing import Tuple, Optional

from src.rl_models import Critic0, Actor, Critic

from configs.specifications.market_bs_cfg import MarketCFG
from configs.specifications.actor_cfg import ActorCFG
from configs.specifications.critic_cfg import CriticCFG
from configs.base.runtime_cfg import RuntimeCFG
from utils.log_print import LogPrint
# =====================================================
# utils
# =====================================================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _archive_if_exists(path: str):
    """
    如果文件已存在，改名为 *_YYYYMMDD_HHMM.ext
    """
    if os.path.exists(path):
        ts = time.strftime("%Y%m%d_%H%M")
        base, ext = os.path.splitext(path)
        os.rename(path, f"{base}__{ts}{ext}")


def _actor_dir(market_dir: str, actor_lambda: float) -> str:
    return os.path.join(
        market_dir,
        f"actor_lambda__{str(actor_lambda).replace('.', '_')}"
    )


def _critic_name(lam: float) -> str:
    return f"critic_{str(lam).replace('.', '_')}.pt"

# =====================================================
# log
# =====================================================


# =====================================================
# Critic0, market_cfg, runtime_cfg layer
# =====================================================
def save_market_artifact(
        *,
        log_print:LogPrint,
        save_dir: str,
        market_cfg: MarketCFG,
        critic0: torch.nn.Module,
        critic0_cfg: CriticCFG,
        runtime_cfg: RuntimeCFG,
) -> str:
    """
    """

    # -------------------------------------------------
    # sanity checks
    # -------------------------------------------------
    if market_cfg.level != "market":
        log_print.raise_value_error("MarketCFG.level must be 'market'")

    if critic0_cfg.level != "critic":
        log_print.raise_value_error("CriticCFG.level must be 'critic'")

    # -------------------------------------------------
    # create market directory
    # -------------------------------------------------
    market_dir = save_dir
    _ensure_dir(market_dir)

    market_cfg_path   = os.path.join(market_dir, "market_cfg.json")
    critic0_ckpt_path = os.path.join(market_dir, "critic0.pt")
    runtime_cfg_path  = os.path.join(market_dir, "runtime_cfg.json")

    # -------------------------------------------------
    # archive old critic0 if exists
    # -------------------------------------------------
    _archive_if_exists(critic0_ckpt_path)

    # -------------------------------------------------
    # 1️⃣ Save Critic0 checkpoint (reproducible unit)
    # -------------------------------------------------
    critic0_ckpt = {
        # ---- semantic identity ----
        "model_type": "critic0",
        "form": critic0_cfg.form,

        # ---- function class (architecture / assumptions) ----
        "critic_cfg": critic0_cfg.to_dict(),

        # ---- numerical solution ----
        "state_dict": critic0.state_dict(),

        # ---- optional debug metadata (NOT identity) ----
        "meta": {
            "saved_device": str(next(critic0.parameters()).device),
            "saved_torch_dtype": str(next(critic0.parameters()).dtype),
        },
    }

    torch.save(critic0_ckpt, critic0_ckpt_path)

    # -------------------------------------------------
    # 2️⃣ Save MarketCFG (pure problem definition)
    # -------------------------------------------------
    with open(market_cfg_path, "w") as f:
        json.dump(market_cfg.to_dict(), f, indent=2)

    # -------------------------------------------------
    # 3️⃣ Save RuntimeCFG (optional provenance)
    # -------------------------------------------------
    if runtime_cfg is not None:
        with open(runtime_cfg_path, "w") as f:
            json.dump(runtime_cfg.to_dict(), f, indent=2)

    # -------------------------------------------------
    # summary
    # -------------------------------------------------
    log_print.write(
        f"[MarketArtifactSaver] Saved market artifact:\n"
        f"  path           : {market_dir}\n"
        f"  market cfg     : market_cfg.json\n"
        f"  critic0 ckpt   : critic0.pt\n"
        f"  runtime cfg    : {'runtime_cfg.json' if runtime_cfg is not None else '(not saved)'}"
    )

    return market_dir


def load_market_and_critic0(
        *,
        log_print: LogPrint,
        load_dir: str,
        map_location: Optional[torch.device] = None,
        device: torch.device,
        torch_dtype: torch.dtype,
) -> Tuple[Critic0, MarketCFG, CriticCFG, Optional[RuntimeCFG]]:
    """
    Load a frozen market experiment artifact.

    Returns:
        - critic0      : baseline Critic0 (nn.Module)
        - market_cfg   : MarketCFG (problem definition)
        - critic_cfg   : CriticCFG (function-class for critic0)
        - runtime_cfg  : RuntimeCFG or None (experiment provenance)
    """

    market_dir = load_dir

    market_cfg_path   = os.path.join(market_dir, "market_cfg.json")
    critic0_path      = os.path.join(market_dir, "critic0.pt")
    runtime_cfg_path  = os.path.join(market_dir, "runtime_cfg.json")

    # ------------------------------
    # sanity check
    # ------------------------------
    switch_to_qlbs_rl_root()  # os.getcwd()
    if not os.path.isdir(market_dir):
        log_print.raise_file_not_found_error(f"Market directory not found: {market_dir}")
    if not os.path.isfile(market_cfg_path):
        log_print.raise_file_not_found_error(f"Missing market_cfg.json: {market_cfg_path}")
    if not os.path.isfile(critic0_path):
        log_print.raise_file_not_found_error(f"Missing critic0 checkpoint: {critic0_path}")

    # ------------------------------
    # load MarketCFG
    # ------------------------------
    with open(market_cfg_path, "r") as f:
        info = json.load(f)
        if info["model"] == "heston":
            from configs.specifications.market_heston_cfg import MarketCFG
        else:
            from configs.specifications.market_bs_cfg import MarketCFG
        market_cfg = MarketCFG.from_dict(
            info,
            device=device,
            torch_dtype=torch_dtype,
        )

    # ------------------------------
    # load Critic0 checkpoint
    # ------------------------------
    ckpt = torch.load(critic0_path, map_location=map_location)

    if ckpt.get("model_type") != "critic0":
        log_print.raise_value_error(
            f"Invalid checkpoint type: expected 'critic0', "
            f"got {ckpt.get('model_type')}"
        )

    critic_cfg = CriticCFG.from_dict(ckpt["critic_cfg"])

    critic0 = Critic0(
        cfgs=critic_cfg,
        device=device,
    )

    critic0.load_state_dict(ckpt["state_dict"])
    critic0.eval()

    # ------------------------------
    # load RuntimeCFG (optional)
    # ------------------------------
    runtime_cfg: Optional[RuntimeCFG] = None
    if os.path.isfile(runtime_cfg_path):
        with open(runtime_cfg_path, "r") as f:
            runtime_cfg = RuntimeCFG.from_dict(
                json.load(f),
                # device=device,
                # torch_dtype=torch_dtype,
            )

    # ------------------------------
    # summary
    # ------------------------------
    log_print.write(
        f"[MarketArtifactLoader] Loaded market artifact: \n"
        f"path            : {market_dir}\n"
        f"  - MarketCFG   : market_cfg.json\n"
        f"  - Critic0     : critic0.pt\n"
        f"  - RuntimeCFG  : runtime_cfg.json\n"
    )

    return critic0, market_cfg, critic_cfg, runtime_cfg


# =====================================================
# Actor a layer
# =====================================================
def save_actor(
        *,
        log_print:LogPrint,
        save_dir: str,
        actor_cfg: ActorCFG,
        actor: torch.nn.Module,
) -> str:
    """
    保存 Actor checkpoint（可完整复现）

    保存内容包括：
    - ActorCFG（函数族假设）
    - actor.state_dict（数值解）
    """

    if actor_cfg.level != "actor":
        log_print.raise_value_error("ActorCFG.level must be 'actor'")

    market_dir = save_dir

    actor_dir = os.path.join(
        market_dir,
        f"actor_lambda_{actor.risk_lambda:.4f}"
    )
    os.makedirs(actor_dir, exist_ok=True)

    actor_path = os.path.join(actor_dir, "actor.pt")

    _archive_if_exists(actor_path)

    actor_ckpt = {
        # ---- semantic identity ----
        "model_type": "actor",

        # ---- function class ----
        "actor_cfg": actor_cfg.to_dict(),

        # ---- numerical solution ----
        "state_dict": actor.state_dict(),

        # ---- optional debug meta ----
        "device": str(next(actor.parameters()).device),
        "torch_dtype": str(next(actor.parameters()).dtype),
    }

    torch.save(actor_ckpt, actor_path)

    log_print.write(
        f"[ActorSaver] Saved actor λ={actor.risk_lambda}\n"
        f"  path: {actor_path}"
    )

    return actor_path


def load_actor(
        *,
        log_print:LogPrint,
        load_dir: str,
        actor_lambda: float,
        device: torch.device,
        torch_dtype: torch.dtype,
        map_location: Optional[torch.device] = None,
) -> Tuple[Actor, ActorCFG]:
    """
    """
    market_dir = load_dir

    actor_dir = os.path.join(
        market_dir,
        f"actor_lambda_{actor_lambda:.4f}"
    )
    actor_path = os.path.join(actor_dir, "actor.pt")

    if not os.path.isdir(actor_dir):
        log_print.raise_file_not_found_error(f"Actor directory not found: {actor_dir}")
    if not os.path.isfile(actor_path):
        log_print.raise_file_not_found_error(f"Actor checkpoint not found: {actor_path}")

    # --------------------------------------------------
    # load checkpoint
    # --------------------------------------------------
    ckpt = torch.load(actor_path, map_location=map_location)

    if ckpt.get("model_type") != "actor":
        log_print.raise_value_error(
            f"Invalid checkpoint type: expected 'actor', "
            f"got {ckpt.get('model_type')}"
        )

    # --------------------------------------------------
    # rebuild ActorCFG
    # --------------------------------------------------
    actor_cfg = ActorCFG.from_dict(ckpt["actor_cfg"])

    # --------------------------------------------------
    # rebuild Actor
    # --------------------------------------------------
    actor = Actor(
        cfgs=actor_cfg,
        device=device,
        torch_dtype=torch_dtype,
    )

    # --------------------------------------------------
    # load parameters
    # --------------------------------------------------
    actor.load_state_dict(ckpt["state_dict"])
    actor.eval()

    log_print.write(
        f"[ActorLoader] Loaded actor λ={actor_lambda}\n"
        f"  path: {actor_path}\n"
        f"  device: {device}, dtype: {torch_dtype}"
    )

    return actor, actor_cfg

# =====================================================
# Critic λ layer
# =====================================================
def save_critic_lambda(
    *,
    log_print:LogPrint,
    save_dir: str,
    actor_lambda: float,
    critic_cfg: CriticCFG,
    critic: torch.nn.Module,
) -> str:
    """
    保存 critic(λ) checkpoint（可完整复现）

    保存内容包括：
    - CriticCFG（函数族 / 结构假设）
    - critic.state_dict（数值解）

    不更新 PolicyCFG（registry 由外层控制）
    """

    if critic_cfg.level != "critic":
        log_print.raise_value_error("CriticCFG.level must be 'critic'")

    critic_lambda = critic_cfg.risk_lambda

    # -------------------------------------------------
    # 目录结构：
    # models/
    #   bs_model_{sigma}/
    #     actor_lambda_{actor_lambda}/
    #       critic_{lambda}.pt
    # -------------------------------------------------
    actor_dir = os.path.join(
        save_dir,
        f"actor_lambda_{actor_lambda:.4f}",
    )
    os.makedirs(actor_dir, exist_ok=True)

    critic_path = os.path.join(
        actor_dir,
        f"critic_{critic_lambda:.4f}.pt"
    )

    _archive_if_exists(critic_path)

    # -------------------------------------------------
    # 组织 checkpoint
    # -------------------------------------------------
    critic_ckpt = {
        # ---- semantic identity ----
        "model_type": "critic",
        "form": critic_cfg.form,

        # ---- function class ----
        "critic_cfg": critic_cfg.to_dict(),

        # ---- numerical solution ----
        "state_dict": critic.state_dict(),

        # ---- optional debug meta ----
        "device": str(next(critic.parameters()).device),
        "torch_dtype": str(next(critic.parameters()).dtype),
    }

    torch.save(critic_ckpt, critic_path)

    log_print.write(
        f"[CriticSaver] Saved critic λ={critic_lambda}\n"
        f"  path: {critic_path}"
    )

    return critic_path


def load_critic_lambda(
    *,
        log_print: LogPrint,
        load_dir: str,
        actor_lambda: float,
        critic_lambda: float,
        device: torch.device,
        torch_dtype: torch.dtype,
        map_location: Optional[torch.device] = None,
) -> Tuple[Critic, CriticCFG]:
    """
    """

    # -------------------------------------------------
    # 路径推导
    # -------------------------------------------------
    actor_dir = os.path.join(
        load_dir,
        f"actor_lambda_{actor_lambda:.4f}",
    )

    critic_path = os.path.join(
        actor_dir,
        f"critic_{critic_lambda:.4f}.pt"
    )

    if not os.path.isdir(actor_dir):
        log_print.raise_file_not_found_error(f"Actor directory not found: {actor_dir}")

    if not os.path.isfile(critic_path):
        log_print.raise_file_not_found_error(f"Critic checkpoint not found: {critic_path}")

    # -------------------------------------------------
    # load checkpoint
    # -------------------------------------------------
    ckpt = torch.load(critic_path, map_location=map_location)

    if ckpt.get("model_type") != "critic":
        raise ValueError(
            f"Invalid checkpoint type: expected 'critic', "
            f"got {ckpt.get('model_type')}"
        )

    # -------------------------------------------------
    # rebuild CriticCFG
    # -------------------------------------------------
    critic_cfg = CriticCFG.from_dict(ckpt["critic_cfg"])

    # sanity check（可选但强烈建议）
    if critic_cfg.risk_lambda != critic_lambda:
        raise ValueError(
            f"Risk lambda mismatch: "
            f"path λ={critic_lambda}, "
            f"cfg λ={critic_cfg.risk_lambda}"
        )

    # -------------------------------------------------
    # rebuild Critic
    # -------------------------------------------------
    critic = Critic(
        cfgs=critic_cfg,
        device=device,
    )

    # -------------------------------------------------
    # load parameters
    # -------------------------------------------------
    critic.load_state_dict(ckpt["state_dict"])
    critic.eval()

    log_print.write(
        f"[CriticLoader] Loaded critic λ={critic_lambda}\n"
        f"  path: {critic_path}\n"
        f"  device: {device}"
    )

    return critic, critic_cfg


# =====================================================
# Policy layer
# =====================================================
def generate_policy_cfg(
        *,
        log_print: LogPrint,
        dir: str,
        actor_lambda: float,
):
    """
    为已经存在的 actor / critic 生成 policy_cfg.json
    不重新训练，只做登记
    """

    actor_dir = os.path.join(
        dir,
        f"actor_lambda_{actor_lambda:.4f}"
    )

    if not os.path.isdir(actor_dir):
        log_print.raise_file_not_found_error(f"Actor directory not found: {actor_dir}")

    actor_path = os.path.join(actor_dir, "actor.pt")
    if not os.path.isfile(actor_path):
        log_print.raise_file_not_found_error(f"Missing actor.pt in {actor_dir}")

    # --------------------------------------------------
    # 扫描已有的 critic_*.pt
    # --------------------------------------------------
    critics = {}
    for fname in os.listdir(actor_dir):
        if fname.startswith("critic_") and fname.endswith(".pt"):
            # critic_10000.pt -> 10000
            name = fname.replace("critic_", "").replace(".pt", "")
            if '__'in name:
                lam = float(name.split('__')[0])
                history = name.split('__')[1]
                critics[name] = {
                    "critic_lambda": lam,
                    "checkpoint": fname,
                    "history": history,
                }
            else:
                lam = float(fname.replace("critic_", "").replace(".pt", ""))
                critics[name] = {
                    "critic_lambda": lam,
                    "checkpoint": fname,
                }

    if not critics:
        log_print.raise_file_not_found_error(
            f"No critic_*.pt found under {actor_dir}, "
            "cannot generate PolicyCFG."
        )

    # --------------------------------------------------
    # 构造 PolicyCFG
    # --------------------------------------------------
    policy_cfg = {
        "level": "policy",
        "actor_lambda": actor_lambda,
        "critics": critics,
    }

    # --------------------------------------------------
    # 保存
    # --------------------------------------------------
    policy_cfg_path = os.path.join(actor_dir, "policy_cfg.json")
    with open(policy_cfg_path, "w") as f:
        json.dump(policy_cfg, f, indent=2)

    log_print.write(
        "✅ PolicyCFG generated successfully\n"
        f"   path: {policy_cfg_path}\n"
        f"   actor_lambda: {actor_lambda}\n"
        f"   critics: {list(critics.keys())}\n"
    )


# =====================================================
# Save all
# =====================================================
def save_full_experiment(
    *,
    log_print: LogPrint,
    save_dir: str,
    market_cfg: MarketCFG,
    critic0: Critic0,
    critic0_cfg: CriticCFG,
    runtime_cfg: RuntimeCFG,
    actor_cfg: ActorCFG,
    actor: Actor,
    critic_cfg: CriticCFG,
    critic: Critic,
    stock_path_generator = None,
):
    """
    Save a complete experiment state:
    - market artifact
    - actor checkpoint
    - critic(lambda) checkpoint
    - policy registry
    """

    log_print.write("=== Saving full experiment ===")

    # 1️⃣ Market-level artifact
    save_market_artifact(
        log_print=log_print,
        save_dir=save_dir,
        market_cfg=market_cfg,
        critic0=critic0,
        critic0_cfg=critic0_cfg,
        runtime_cfg=runtime_cfg,
    )
    log_print.write("Market artifact saved")

    # 2️⃣ Actor
    save_actor(
        log_print=log_print,
        save_dir=save_dir,
        actor_cfg=actor_cfg,
        actor=actor,
    )
    log_print.write(f"Actor saved (λ={actor_cfg.risk_lambda})")

    # 3️⃣ Critic(lambda)
    save_critic_lambda(
        log_print=log_print,
        save_dir=save_dir,
        actor_lambda=actor_cfg.risk_lambda,
        critic_cfg=critic_cfg,
        critic=critic,
    )
    log_print.write(f"Critic saved (λ={critic_cfg.risk_lambda})")

    # 4️⃣ Policy registry
    generate_policy_cfg(
        log_print=log_print,
        dir=save_dir,
        actor_lambda=actor_cfg.risk_lambda,
    )
    log_print.write("PolicyCFG generated")
    if stock_path_generator:
        from datetime import datetime
        np.save(
            f'{save_dir}/trained_S_path.npy',
            stock_path_generator.S_path.cpu().numpy()
        )
        if stock_path_generator.cfg.model == 'heston':
            np.save(
                f'{save_dir}/trained_V_path.npy',
                stock_path_generator.V_path.cpu().numpy()
            )

    log_print.write("=== Full experiment saved ===")

# =====================================================
# Save all
# =====================================================
def load_full_experiment(
    *,
    load_dir: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    log_print: LogPrint = None,
    actor_lambda: Optional[float] = None,
    critic_lambda: Optional[float] = None,
):
    """
    Load a complete experiment state.

    Returns:
      critic0,
      market_cfg,
      critic0_cfg,
      runtime_cfg,
      actor,
      actor_cfg,
      critic (optional),
      critic_cfg (optional)
    """
    if log_print is None:
        log_print = LogPrint(True)
    log_print.write("=== Loading full experiment ===")

    # 1️⃣ Market artifact
    critic0, market_cfg, critic0_cfg, runtime_cfg = load_market_and_critic0(
        log_print=log_print,
        load_dir=load_dir,
        device=device,
        torch_dtype=torch_dtype,
    )
    log_print.write("Market artifact loaded")

    # 2️⃣ Actor
    actor, actor_cfg = None, None
    if actor_lambda is not None:
        actor, actor_cfg = load_actor(
            log_print=log_print,
            load_dir=load_dir,
            actor_lambda=actor_lambda,
            device=device,
            torch_dtype=torch_dtype,
        )
        log_print.write(f"Actor loaded (λ={actor_lambda})")
    else:
        log_print.write("No actor specified, actor not loaded")

    # 3️⃣ Optional critic(lambda)
    critic, critic_cfg = None, None

    if critic_lambda is not None:
        critic, critic_cfg = load_critic_lambda(
            log_print=log_print,
            load_dir=load_dir,
            actor_lambda=actor_lambda,
            critic_lambda=critic_lambda,
            device=device,
            torch_dtype=torch_dtype,
        )
        log_print.write(f"Critic loaded (λ={critic_lambda})")
    else:
        log_print.write("No critic_lambda specified, critic not loaded")

    log_print.write("=== Full experiment loaded ===")

    return (
        critic0,
        market_cfg,
        critic0_cfg,
        runtime_cfg,
        actor,
        actor_cfg,
        critic,
        critic_cfg,
    )


import os


def switch_to_qlbs_rl_root():
    """
    检测当前工作目录是否在qlbs_rl下，若不是则切换到qlbs_rl根目录
    """
    # 1. 获取当前工作目录
    current_dir = os.getcwd()
    print(f"当前工作目录：{current_dir}")

    # 2. 定义目标根目录标识（qlbs_rl）
    root_dir_name = "qlbs_rl"

    # 3. 拆分当前路径，找到qlbs_rl的位置
    path_parts = current_dir.split(os.sep)  # 按系统分隔符拆分路径（兼容Windows/Mac/Linux）

    try:
        # 找到qlbs_rl在路径中的索引
        root_index = path_parts.index(root_dir_name)
        # 拼接qlbs_rl根目录路径
        root_dir = os.sep.join(path_parts[:root_index + 1])

        # 4. 判断是否需要切换目录
        if current_dir != root_dir:
            os.chdir(root_dir)
            print(f"已切换到qlbs_rl根目录：{os.getcwd()}")
        else:
            print("当前已是qlbs_rl根目录，无需切换")

    except ValueError:
        # 路径中没有qlbs_rl，抛出友好提示
        raise FileNotFoundError(f"错误：当前路径 {current_dir} 中未找到 {root_dir_name} 目录！")
