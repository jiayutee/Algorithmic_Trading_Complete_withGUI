from typing import Any, List
from .base import BaseRuntimeAgent, AgentResult, RuntimeContext
from . import tools


class PriceAgent(BaseRuntimeAgent):
    def __init__(self, symbols: List[str] | None = None):
        super().__init__("price")
        self.symbols = symbols or ["AAPL", "SPY"]

    def run_once(self, ctx: RuntimeContext) -> AgentResult:
        try:
            results = {}
            for s in self.symbols:
                p = tools.get_latest_price(s)
                results[s] = p
            summary = f"prices_checked={len(results)}"
            return AgentResult(self.name, "ok", summary, {"prices": results})
        except Exception as e:
            return AgentResult(self.name, "error", f"exception: {e}")
