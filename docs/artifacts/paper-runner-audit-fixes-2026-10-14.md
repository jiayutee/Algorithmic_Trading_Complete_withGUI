# Paper runner audit findings and paper-account follow-ups — fixes, 2026-10-14

Source: `docs/artifacts/paper-execution-audit-2026-10-03.md` (findings 1-3) and the "still open" list of the desktop-backtest
isolation work (PR #40). Branch `fix/paper-runner-audit-findings`, base main `47a9ded`. Paper only; no live connector touched.

## Fixed
| # | Finding | Change | Test |
|---|---|---|---|
| 1 | Peak equity and the day's starting equity moved only when an order was about to be sent, so after weeks of holds the drawdown (15%) and daily-loss (3%) entry limits were measured from stale numbers (journal still had 100,000 from 09-19) | `ExecutionService._refresh_baselines()` runs at the start of every poll: marks each held symbol to a fresh price, then updates the baselines. If any held symbol has no fresh price, or anything raises, the baselines are left alone (a cost-basis equity would be wrong in either direction) and the poll continues | 4 tests in `tests/test_execution_service.py`; the first fails on the old code |
| 2 | BNBUSDT had no Yahoo fallback (the map covered BTC/ETH/SOL/ADA only) | any `<BASE>USDT` maps to `<BASE>-USD` (`DataLoader._yahoo_crypto_symbol`) | `tests/test_data_loader_crypto_fallback.py` |
| 3 | The Binance batch-failure warning did not say what failed | the warning now names the symbol and the exception class + message | same file |
| 4 | `SimulatedBroker` did not reload fees from the account file, so a differently configured instance charged other fees and wrote them back | saved fees win over constructor fees, like the starting balance | `tests/test_paper_account.py` (fails on old code) |
| 5 | `reset()` with no argument used this instance's in-memory starting balance, not the file's | reloads from the file first | same (fails on old code) |
| 6 | Order ids were `simorder_<ms>_<4 random digits>`: two orders in the same millisecond with the same digits (about 1 in 9,000 such pairs) collided and the second silently overwrote the first in the shared store | suffix is now 48 random bits | same (fails on old code) |

## Effect on the running system
Nothing changes until the always-on runner is restarted on a checkout that contains this branch
(`launchctl kickstart -k gui/$(id -u)/com.algotrader.paper-execution`). After a restart the runner's first poll will move
`peak_equity` and `day_baseline` to the current marked equity (it was about 98,150 on 10-09 against a stale 100,000 peak), so the
drawdown/daily-loss limits become as strict as the documentation says. They block new entries only; exits are never blocked.
The four open positions are unaffected.

## Flaky test: `test_paper_account.py::test_two_processes_share_one_account`
It failed in two other sessions (one full-suite run, one single-file run) and I could not reproduce it: 90 isolated runs and 30 runs of
the file under load all passed. Item 6 above is a plausible cause (five orders across two instances in a few milliseconds), but it is
not proven, and the original failure messages were not kept. If it fails again, keep the assertion text.

## Not done
- Python 3.11 env and `test_gui.py` were not run for this branch.
- Binance first-request failures (129 in 7 days) are still unexplained; finding 3 only makes the next one diagnosable.
