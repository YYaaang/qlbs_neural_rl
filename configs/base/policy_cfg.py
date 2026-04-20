from dataclasses import dataclass, field, asdict
from typing import Dict, Any
from datetime import datetime


@dataclass
class CriticEntry:
    lambda_value: float
    path: str
    trained_with_actor_lambda: float
    status: str = "trained"
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M")
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CriticEntry":
        return cls(**d)


@dataclass
class PolicyCFG:
    # ---------- identity ----------
    level: str                    # must be "policy"
    actor_lambda: float

    # ---------- registry ----------
    critics: Dict[str, CriticEntry] = field(default_factory=dict)

    # ===============================
    # registry ops
    # ===============================
    def register_critic(self, entry: CriticEntry):
        key = str(entry.lambda_value)
        self.critics[key] = entry

    # ===============================
    # serialization
    # ===============================
    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "actor_lambda": self.actor_lambda,
            "critics": {
                k: v.to_dict() for k, v in self.critics.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PolicyCFG":
        if d.get("level") != "policy":
            raise ValueError("PolicyCFG.level must be 'policy'")
        critics_raw = d.get("critics", {})
        critics = {
            k: CriticEntry.from_dict(v)
            for k, v in critics_raw.items()
        }
        return cls(
            level="policy",
            actor_lambda=d["actor_lambda"],
            critics=critics,
        )