"""Instrument catalogue: consistent with how the loader routes symbols, and honest about what each kind supports."""
import pytest

from core import instruments as ins
from core import ui_options
from core.chart_builder import is_crypto_symbol

PREVIOUS_LIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "AAPL", "TSLA", "GOLD", "SPY", "QQQ"]


def test_the_catalogue_is_large_unique_and_named():
    syms = ins.all_symbols()
    assert len(syms) >= 80 and len(set(syms)) == len(syms)
    assert all(ins.name_of(s) for s in syms)
    assert set(PREVIOUS_LIST) <= set(syms)                              # nobody loses an instrument they had before


def test_kind_agrees_with_the_loaders_routing_rule_for_every_symbol():
    """The data loader sends any symbol containing 'USDT' to Binance and everything else to Yahoo. A catalogue entry whose kind
    disagrees would load from the wrong place."""
    for s in ins.all_symbols():
        assert is_crypto_symbol(s) == (ins.kind_of(s) == "crypto"), s


def test_kinds_of_catalogue_and_typed_symbols():
    assert [ins.kind_of(s) for s in ("BTCUSDT", "AAPL", "SPY", "GC=F", "^GSPC", "EURUSD=X")] == ["crypto", "stock", "etf", "future", "index", "fx"]
    # symbols the user types that are not in the list are classified by their shape
    assert [ins.kind_of(s) for s in ("pepeusdt", " nvda ", "^N225", "ZC=F", "GBPJPY=X")] == ["crypto", "stock", "index", "future", "fx"]


def test_paper_execution_supports_only_things_that_can_actually_be_bought():
    assert all(ins.paper_tradable(s) for s in ("BTCUSDT", "AAPL", "SPY", "MYSTERYUSDT"))
    for s, word in (("^GSPC", "index"), ("GC=F", "futures"), ("EURUSD=X", "FX")):
        assert not ins.paper_tradable(s)
        msg = ins.paper_refusal(s)
        assert s in msg and word in msg and "chart/backtest-only" in msg
    assert ins.paper_refusal("AAPL") == ""


def test_labels_show_the_name_and_unknown_symbols_fall_back_to_the_bare_symbol():
    assert ins.label_of("AAPL") == "AAPL — Apple" and ins.label_of("ZZZZ") == "ZZZZ"
    assert ui_options.SYMBOLS == ins.all_symbols() and ui_options.SYMBOL_LABELS["BTCUSDT"].startswith("BTCUSDT")


def test_the_one_stock_named_gold_is_labelled_so_nobody_mistakes_it_for_the_metal():
    assert "not gold" in ins.name_of("GOLD") and ins.kind_of("GOLD") == "stock" and ins.kind_of("GC=F") == "future"
