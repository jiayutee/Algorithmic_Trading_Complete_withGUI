"""Risk gate rules, in isolation."""
import pytest

from core.execution.risk import RiskConfig, RiskGate, RiskInput


def inp(**kw):
    base = dict(symbol="BTCUSDT", side="buy", qty=1.0, price=100.0, current_qty=0.0, equity=100_000.0, day_start_equity=100_000.0,
                peak_equity=100_000.0, gross_exposure=0.0, orders_today=0, data_age_bars=0.5, halted=None, is_crypto=True)
    base.update(kw)
    return RiskInput(**base)


gate = RiskGate(RiskConfig())


def test_a_normal_entry_is_approved_unchanged():
    v = gate.evaluate(inp(qty=10))
    assert v.approved and v.qty == 10 and not v.is_exit


@pytest.mark.parametrize("bad", [dict(halted={"reason": "x"}), dict(data_age_bars=9), dict(equity=96_000),
                                 dict(peak_equity=120_000), dict(orders_today=20)])
def test_each_entry_limit_blocks_a_new_position(bad):
    v = gate.evaluate(inp(qty=10, **bad))
    assert not v.approved and v.qty == 0 and v.reasons


@pytest.mark.parametrize("bad", [dict(halted={"reason": "x"}), dict(data_age_bars=99), dict(equity=50_000, day_start_equity=100_000),
                                 dict(peak_equity=500_000), dict(orders_today=999)])
def test_exits_are_never_blocked_by_any_limit(bad):
    long_exit = gate.evaluate(inp(side="sell", qty=5, current_qty=5, **bad))
    short_exit = gate.evaluate(inp(side="buy", qty=5, current_qty=-5, **bad))
    assert long_exit.approved and long_exit.is_exit and long_exit.qty == 5
    assert short_exit.approved and short_exit.is_exit


def test_entry_is_shrunk_to_the_position_limit_and_blocked_when_no_room_is_left():
    v = gate.evaluate(inp(qty=1000, price=100.0))                       # asks for $100k; limit is 25% = $25k
    assert v.approved and v.qty == pytest.approx(250) and "reduced" in v.reasons[0]
    full = gate.evaluate(inp(qty=1, current_qty=250))                   # already at the limit
    assert not full.approved and "position limit" in full.reasons[0]


def test_gross_exposure_limit_applies_across_symbols():
    v = gate.evaluate(inp(qty=1000, gross_exposure=70_000))             # 75% cap leaves $5k room
    assert v.approved and v.qty == pytest.approx(50) and "gross" in v.reasons[0]
    assert not gate.evaluate(inp(qty=1, gross_exposure=75_000)).approved


def test_dust_orders_are_refused():
    v = gate.evaluate(inp(qty=0.05, price=100.0))                       # $5 < $10 minimum
    assert not v.approved and "minimum" in v.reasons[0]


def test_equities_tolerate_longer_gaps_between_bars_than_crypto():
    assert not gate.evaluate(inp(qty=10, data_age_bars=4, is_crypto=True)).approved
    assert gate.evaluate(inp(qty=10, data_age_bars=4, is_crypto=False)).approved
    assert not gate.evaluate(inp(qty=10, data_age_bars=6, is_crypto=False)).approved


def test_kill_switch_file_blocks_entries_but_not_exits(tmp_path):
    f = tmp_path / ".kill_switch"
    g = RiskGate(RiskConfig(kill_switch_file=str(f)))
    assert g.evaluate(inp(qty=10)).approved
    f.write_text("stop")
    assert not g.evaluate(inp(qty=10)).approved
    assert g.evaluate(inp(side="sell", qty=5, current_qty=5)).approved


def test_nonsense_orders_are_refused():
    assert not gate.evaluate(inp(qty=0)).approved
    assert not gate.evaluate(inp(price=0)).approved
    assert not gate.evaluate(inp(qty=-1)).approved
