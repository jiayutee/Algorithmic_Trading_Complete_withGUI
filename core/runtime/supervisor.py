"""Simple supervisor that coordinates runtime agents and an Ollama client."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any
import time
import os
import threading
from .ollama_client import OllamaClient
from .base import RuntimeContext, AgentResult, RuntimePolicy, SharedLLMConfig
from .portfolio_agent import PortfolioAgent
from .news_agent import NewsAgent
from .price_agent import PriceAgent
from .stats_agent import StatsAgent
from .sql_writer import SQLWriter


class Supervisor:
    """Supervisor runs agents in a simple periodic loop and exposes status snapshots.

    The supervisor maintains an in-memory history (capped) of AgentResult dictionaries
    so a UI can poll `status()` or `snapshot()` without blocking agent execution.
    """

    def __init__(self, ollama_url: str | None = None, ollama_model: str | None = None):
        self.ollama_url = ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        # instantiate Ollama client first so we can query available models
        self.ollama = OllamaClient(self.ollama_url)
        self._llm_lock = threading.Lock()

        # Determine which Ollama model to use. Precedence:
        # 1) explicit constructor argument `ollama_model`
        # 2) `OLLAMA_MODEL` env var
        # 3) query the Ollama server for available models and pick the first
        #    matching one from the preferred list
        # 4) fallback to a sensible default
        preferred_order = ("qwen3:8b", "qwen2.5:7b")
        allowed_models = set(preferred_order)
        chosen_model: str | None = None

        if ollama_model:
            chosen_model = ollama_model if ollama_model in allowed_models else None
        else:
            env_model = os.environ.get("OLLAMA_MODEL")
            if env_model and env_model in allowed_models:
                chosen_model = env_model
            else:
                # try to query the Ollama server for available models; be defensive
                try:
                    lm = self.ollama.list_models()
                    models_list: list = []
                    if isinstance(lm, dict):
                        # support {'ok': True, 'models': [...]} or {'models': [...]} shapes
                        if "models" in lm and isinstance(lm.get("models"), list):
                            models_list = lm.get("models") or []
                        else:
                            # if dict but not the expected shape, try to extract values
                            # (defensive fallback)
                            maybe = lm.get("models") if isinstance(lm, dict) else None
                            if isinstance(maybe, list):
                                models_list = maybe
                    elif isinstance(lm, list):
                        models_list = lm

                    # normalize model entries to strings
                    normalized = []
                    for m in models_list:
                        if isinstance(m, str):
                            normalized.append(m)
                        elif isinstance(m, dict):
                            # common keys might be 'name' or 'model'
                            name = m.get("name") or m.get("model")
                            if isinstance(name, str):
                                normalized.append(name)
                    # pick the first preferred model that appears in the normalized list
                    for pref in preferred_order:
                        for m in normalized:
                            if pref == m or pref in m:
                                chosen_model = m
                                break
                        if chosen_model:
                            break
                except Exception:
                    # keep chosen_model as None and fall through to default
                    chosen_model = None

        if not chosen_model:
            chosen_model = "qwen3:8b"

        self.ollama_model = chosen_model
        self.shared_llm = SharedLLMConfig(client=self.ollama, model=self.ollama_model, lock=self._llm_lock)
        self.sql_writer = SQLWriter()
        # last short summary text returned from the model (or fallback message)
        self.last_summary: str = "(no summary yet)"
        # instantiate agents
        self.agents = [
            PortfolioAgent(),
            NewsAgent(),
            PriceAgent(),
            StatsAgent(),
        ]
        # history: agent_name -> list[AgentResult]
        self.status_history: Dict[str, List[AgentResult]] = {a.name: [] for a in self.agents}
        self._history_limit = 200
        self._history_lock = threading.Lock()
        # loop control
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _update_history(self, result: AgentResult) -> None:
        with self._history_lock:
            hist = self.status_history.setdefault(result.name, [])
            hist.append(result)
            if len(hist) > self._history_limit:
                hist[:] = hist[-self._history_limit:]

    def _chat_llm(self, messages: list[dict[str, str]]) -> dict:
        with self._llm_lock:
            return self.ollama.chat(self.ollama_model, messages)

    def health(self) -> dict:
        try:
            return self.ollama.health()
        except Exception:
            return {"ok": False, "error": "ollama_unavailable"}

    def run_cycle(self, ctx: Optional[RuntimeContext] = None) -> List[AgentResult]:
        ctx = ctx or RuntimeContext()
        # ensure runtime policy exists
        if ctx.policy is None:
            ctx.policy = RuntimePolicy()
        # expose clients/tools
        ctx.tools["ollama"] = self.ollama
        ctx.tools["ollama_model"] = self.ollama_model
        ctx.tools["shared_llm"] = self.shared_llm
        ctx.tools["sql_writer"] = self.sql_writer
        results: List[AgentResult] = []

        deterministic_agents = [a for a in self.agents if a.name in ("price", "stats")]
        sequential_agents = [a for a in self.agents if a.name not in ("price", "stats")]

        def _run_agent(agent) -> AgentResult:
            try:
                return agent.run_once(ctx)
            except Exception as e:
                return AgentResult(agent.name, "error", f"exception: {e}")

        if deterministic_agents:
            with ThreadPoolExecutor(max_workers=len(deterministic_agents)) as executor:
                future_map = {executor.submit(_run_agent, agent): agent for agent in deterministic_agents}
                for future in as_completed(future_map):
                    result = future.result()
                    results.append(result)
                    self._update_history(result)

        for agent in sequential_agents:
            result = _run_agent(agent)
            results.append(result)
            self._update_history(result)

        # After collecting agent results, ask Ollama for a short summary.
        try:
            # build concise prompt summarizing agent statuses
            lines = []
            warnings = []
            for r in results:
                lines.append(f"{r.name}: {r.status} - {str(r.summary)}")
                if isinstance(r.status, str) and r.status.lower() in ("warning", "error"):
                    warnings.append(f"{r.name}: {r.summary}")

            prompt = "Provide a one-sentence summary of the following agent statuses. Highlight warnings or errors if present.\n\n" + "\n".join(lines)
            if warnings:
                prompt += "\n\nWarnings/Errors:\n" + "\n".join(warnings)

            messages = [
                {"role": "system", "content": "You are a concise monitoring assistant for a financial runtime. Reply with one short paragraph."},
                {"role": "user", "content": prompt},
            ]

            resp = self._chat_llm(messages)
            summary_text = None
            if resp.get("ok"):
                response = resp.get("response")
                # parse common response shapes defensively
                if isinstance(response, str):
                    summary_text = response
                elif isinstance(response, dict):
                    # Ollama-like: {'choices': [{'message': {'role':'assistant','content': '...'}}]}
                    choices = response.get("choices")
                    if isinstance(choices, list) and choices:
                        first = choices[0]
                        # message may be nested
                        msg = first.get("message") or first
                        if isinstance(msg, dict):
                            # try a few keys
                            summary_text = msg.get("content") or msg.get("text") or str(msg)
                        else:
                            summary_text = str(msg)
                    else:
                        # sometimes direct 'text' key
                        summary_text = response.get("text") or str(response)
                else:
                    summary_text = str(response)

            if not summary_text:
                # fallback: produce a simple aggregated status line
                summary_text = "Summary unavailable (Ollama returned no usable output). Recent statuses: " + ", ".join([f"{r.name}={r.status}" for r in results])

            self.last_summary = summary_text
        except Exception:
            # defensive fallback: keep runtime working
            self.last_summary = "Ollama summary unavailable; using local fallback."

        return results

    def start(self, loop_delay: float = 1.0) -> None:
        """Start background polling thread. Idempotent."""
        if self._thread and self._thread.is_alive():
            return

        def _run():
            ctx = RuntimeContext()
            ctx.policy = RuntimePolicy()
            while not self._stop_event.is_set():
                try:
                    self.run_cycle(ctx)
                except Exception:
                    pass
                time.sleep(loop_delay)

        self._stop_event.clear()
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        try:
            self.sql_writer.stop()
        except Exception:
            pass

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return latest status per agent and history length metadata."""
        out: Dict[str, Dict[str, Any]] = {}
        with self._history_lock:
            for name, hist in self.status_history.items():
                latest = hist[-1] if hist else None
                out[name] = {
                    "latest": latest,
                    "history_len": len(hist),
                }
        # include last_summary for external callers/UI
        out["__meta__"] = {"last_summary": getattr(self, "last_summary", "(no summary)")}
        return out

    # compatibility alias
    status = snapshot
