from typing import Any
from .base import BaseRuntimeAgent, AgentResult, RuntimeContext
from . import tools


class PortfolioAgent(BaseRuntimeAgent):
    def __init__(self):
        super().__init__("portfolio")

    def run_once(self, ctx: RuntimeContext) -> AgentResult:
        try:
            # include policy information in portfolio output; portfolio actions
            # must be explicitly enabled via RuntimePolicy.autonomy_enabled['portfolio']
            res = tools.get_portfolio()
            if not res.get("ok"):
                return AgentResult(self.name, "warning", "broker manager unavailable", {"raw": res})
            port = res.get("portfolio")
            summary = f"positions={len(port.get('positions',[]))} cash={port.get('cash')!s}"
            shared_llm = ctx.tools.get("shared_llm")
            meta = {
                "portfolio": port,
                "autonomous_allowed": False,
                "shared_llm_model": getattr(shared_llm, "model", None),
            }
            try:
                if getattr(ctx, "policy", None) and ctx.policy.autonomy_enabled.get("portfolio", False):
                    meta["autonomous_allowed"] = True
            except Exception:
                pass
            return AgentResult(self.name, "ok", summary, meta)
        except Exception as e:
            return AgentResult(self.name, "error", f"exception: {e}")
