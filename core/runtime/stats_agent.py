from typing import Any
from .base import BaseRuntimeAgent, AgentResult, RuntimeContext
from . import tools


class StatsAgent(BaseRuntimeAgent):
    def __init__(self):
        super().__init__("stats")

    def run_once(self, ctx: RuntimeContext) -> AgentResult:
        try:
            # Lightweight stats: try to read a small progress CSV if present
            try:
                from pathlib import Path
                p = Path("results/ddpg/progress.csv")
                if p.exists():
                    lines = p.read_text().splitlines()
                    summary = f"progress_lines={len(lines)}"
                    return AgentResult(self.name, "ok", summary, {"lines": len(lines)})
            except Exception:
                pass
            return AgentResult(self.name, "warning", "no stats available")
        except Exception as e:
            return AgentResult(self.name, "error", f"exception: {e}")
