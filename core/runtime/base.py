from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
import os


@dataclass
class RuntimeContext:
    config: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=lambda: dict(os.environ))
    # place to stash clients/tools
    tools: Dict[str, Any] = field(default_factory=dict)
    # runtime policy / safety gating
    policy: Optional["RuntimePolicy"] = None


@dataclass
class AgentResult:
    name: str
    status: str  # e.g. ok, warning, error
    summary: str
    data: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class SharedLLMConfig:
    """Shared local LLM reference used by runtime agents."""

    client: Any
    model: str
    lock: Any = None


class BaseRuntimeAgent(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run_once(self, ctx: RuntimeContext) -> AgentResult:
        """Run a single polling/decision cycle.

        Must return an AgentResult and avoid throwing on predictable runtime failures.
        """
        raise NotImplementedError()


@dataclass
class RuntimePolicy:
    # maximum aggregate value of positions allowed for autonomous decisions
    max_position_value: float = 1_000_000.0
    # maximum single order size/value
    max_order_value: float = 10_000.0
    # require human approval before executing any trade
    require_trade_approval: bool = True
    # autonomy flags for individual agent domains
    autonomy_enabled: Dict[str, bool] = field(default_factory=lambda: {
        "news": True,
        "price": True,
        "stats": True,
        "portfolio": False,
    })
