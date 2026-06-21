# Orchestrator Refactor Plan

## Gaps
- `strategy_manager.py` still mixes orchestration, source selection, and execution concerns.
- Live trading, backtesting, and ML/RL strategy wrappers are not cleanly separated.
- Strategy wiring is implicit, which makes testing and extension harder.

## Next Steps
1. Add a plugin-style strategy registry with a stable strategy interface.
2. Split live execution adapters from backtest runners and shared signal logic.
3. Wrap ML/RL strategies behind thin adapters for `FinRL`, `DDPG`, and `TD3`.
4. Keep broker, data, and order-state handling out of strategy classes.
5. Add tests for news dedupe/merge behavior and strategy registration/wiring.

## Test Focus
- Deduping repeated news items across sources.
- Merge rules for canonicalized URLs and fuzzy headline matches.
- Registry lookup, adapter selection, and live/backtest dispatch paths.
