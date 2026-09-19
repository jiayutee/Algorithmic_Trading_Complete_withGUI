"""
conftest.py — repo-wide pytest fixtures.

Keeps retry-with-backoff code (core/yf_session.py's download_with_retry /
fetch_earnings_dates_with_retry) from introducing real wall-clock sleeps into
the test suite. Tests that exercise the empty/error paths of yfinance calls
(mocked) would otherwise trigger genuine exponential-backoff delays between
retry attempts. Forcing a single attempt means "no second attempt" -> no
sleep, while leaving the retry loop's logic itself under test (it still runs
its empty/exception-handling branch once).

This does not affect production behavior: outside pytest, YF_FETCH_MAX_RETRIES
defaults to 3 as documented in core/yf_session.py.
"""

import importlib
import sys
import types
from types import SimpleNamespace as NS
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _fast_yf_retries(monkeypatch):
    monkeypatch.setenv("YF_FETCH_MAX_RETRIES", "1")


@pytest.fixture
def ib_env(monkeypatch):
    """Shared mocked ib_insync (no TWS/Gateway ever needed): installs a fake module, reloads the connector and
    BrokerManager so they import it, and restores the real (not-installed) state afterwards.
    Exposes .ib (the mocked IB instance), .mod, .bm (core.broker_manager), .ibc (brokers.ib_connector)."""
    fake_ib = mock.MagicMock(name="IB()")
    fake_ib.isConnected.return_value = False
    mod = types.ModuleType("ib_insync")
    mod.IB = mock.MagicMock(return_value=fake_ib)
    mod.MarketOrder = lambda action, qty: NS(action=action, totalQuantity=qty)
    mod.Contract = lambda **kw: NS(**kw)
    monkeypatch.setitem(sys.modules, "ib_insync", mod)
    import brokers.ib_connector as ibc
    import core.broker_manager as bm
    importlib.reload(ibc)
    importlib.reload(bm)
    yield NS(ib=fake_ib, mod=mod, bm=bm, ibc=ibc)
    monkeypatch.delitem(sys.modules, "ib_insync", raising=False)
    sys.modules.pop("brokers.ib_connector", None)                # drop the copy bound to the fake ib_insync
    # restore the "ib_insync not installed" state so later tests see the real environment
    importlib.reload(importlib.import_module("core.broker_manager"))


