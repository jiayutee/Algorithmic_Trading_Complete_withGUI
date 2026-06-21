"""Runnable demo that instantiates the Supervisor and runs a short cycle.

Usage:
    python scripts/run_runtime_supervisor.py
"""
from core.runtime.supervisor import Supervisor
from core.runtime.base import RuntimeContext


def main():
    s = Supervisor()
    print("Ollama health:", s.health())
    ctx = RuntimeContext()
    results = s.run_cycle(ctx)
    for r in results:
        print(f"[{r.name}] {r.status} - {r.summary}")
    # print short model summary if available
    try:
        if getattr(s, "last_summary", None):
            print("Supervisor summary:", s.last_summary)
    except Exception:
        pass


if __name__ == "__main__":
    main()
