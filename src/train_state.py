class TrainState:
    """
    纯训练状态管理器：
    - EMA
    - best_ema
    - streak
    - patience
    - warmup_ignore
    - steps
    - is_stable
    """

    def __init__(
        self,
        *,
        ema_beta: float = 0.9,
        patience: int = 5,
        warmup_ignore: int = 0,
    ):
        self.ema_beta = float(ema_beta)
        self.patience = int(max(1, patience))
        self.warmup_ignore = int(max(0, warmup_ignore))

        self.reset()

    # ------------------------------------------------------
    # reset：每个 timestep t 必须调用
    # ------------------------------------------------------
    def reset(self):
        self.ema = float("inf")         # 根据你的选择 ema_init="inf"
        self.best_ema = float("inf")
        self.streak = 0
        self.is_stable = False
        self.steps = 0

    # ------------------------------------------------------
    # update：更新状态并返回当前 EMA（给 Trainer 用）
    # Trainer 决定是否 scheduler.step(ema)
    # ------------------------------------------------------
    def update(self, loss_val: float):
        self.steps += 1

        # -----------------------------
        # (1) EMA 更新
        # -----------------------------
        if self.ema == float("inf"):
            # 第一帧：EMA 初始化为 loss
            self.ema = loss_val
        else:
            # 经典 EMA
            self.ema = self.ema_beta * self.ema + (1.0 - self.ema_beta) * loss_val

        ema_now = self.ema

        # -----------------------------
        # (2) warmup_ignore：前 N 步不算 streak
        # -----------------------------
        if self.steps <= self.warmup_ignore:
            self.streak = 0
            self.is_stable = False
            return ema_now

        # -----------------------------
        # (3) pure_improvement 判定
        # ema < best_ema → 改善
        # -----------------------------
        if ema_now < self.best_ema:
            self.best_ema = ema_now
            self.streak = 0
        else:
            self.streak += 1

        # -----------------------------
        # (4) 判断是否稳定（早停条件的一部分）
        # -----------------------------
        self.is_stable = (self.streak >= self.patience)

        return ema_now
