Runtime Agents (Ollama) — Notes

Overview
- The runtime agents package is a lightweight scaffold that lets you run small autonomous agents
  backed by a local Ollama inference server.

Ollama connectivity
- Yes: the supervisor connects to Ollama at http://localhost:11434 by default.
  - You can configure the URL by setting the `OLLAMA_URL` environment variable.

Tooling and MCP caveat
- No: models served by Ollama do NOT automatically gain access to Python/MCP tools.
  - Your Python runtime must explicitly expose and call tools (adapters/wrappers) when needed.
  - The runtime here provides simple `tools` adapters that call existing repo components.

Agents provided
- `portfolio` — queries broker manager for current positions and cash.
- `news` — fetches recent news via the repo pipeline and stores it.
- `price` — samples latest prices for a small watchlist.
- `stats` — lightweight health/stats scraping of local artifacts.

Autonomy vs gated controls
- Autonomous parts: polling, read-only summarization, local logging and light storage.
- Gated parts (must be risk-controlled): order placement, automated portfolio rebalancing,
  or any action that executes trades or moves money. The scaffold intentionally omits
  any autonomous execution of trades — use the `BrokerManager` directly and gate it
  behind approval workflows and explicit risk checks.

File map
- core/runtime/__init__.py
- core/runtime/base.py
- core/runtime/ollama_client.py
- core/runtime/tools.py
- core/runtime/portfolio_agent.py
- core/runtime/news_agent.py
- core/runtime/price_agent.py
- core/runtime/stats_agent.py
- core/runtime/supervisor.py
- scripts/run_runtime_supervisor.py

Launch example
1) Start Ollama locally (if you use it). Default host: `http://localhost:11434`.
2) Run the supervisor demo:

```bash
python scripts/run_runtime_supervisor.py
```

Notes
- The implementation is intentionally minimal and defensive: if Ollama or repo modules
  are unavailable, agents return clear statuses instead of raising. Extend the agents
  to add richer prompts, tool invocation patterns, and risk gating as needed.
 - The supervisor can now ask Ollama for a short status summary after each polling
   cycle; this summary is recorded for UIs but the model is not granted direct tool access.
