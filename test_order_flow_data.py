"""Phase 6.5 data features: taker flow + funding rates. Same rule as everywhere: no lookahead."""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import core.order_flow_data as ofd
from core.order_flow_data import fetch_funding_rates, fetch_klines, funding_features, taker_flow_features


def _klines(n=120, seed=0):
    rng = np.random.default_rng(seed)
    vol = rng.uniform(1000, 2000, n)
    return pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": vol,
                         "Trades": rng.integers(5000, 9000, n).astype(float),
                         "TakerBuyBase": vol * rng.uniform(0.35, 0.65, n)},
                        index=pd.date_range("2024-01-01", periods=n, freq="D"))


def _funding(days=120, seed=1, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 3, freq="8h")
    return pd.Series(rng.normal(0.0001, 0.0002, len(idx)), index=idx, name="funding_rate")


# ---------------------------------------------------------------- taker flow

def test_taker_ratio_is_buy_share_of_volume():
    k = _klines(30)
    f = taker_flow_features(k)
    np.testing.assert_allclose(f["tb_ratio"], k["TakerBuyBase"] / k["Volume"])
    assert f["tb_ratio"].between(0, 1).all()


@pytest.mark.parametrize("cut", [30, 61, 99])
def test_taker_features_have_no_lookahead(cut):
    k = _klines()
    pd.testing.assert_frame_equal(taker_flow_features(k).iloc[:cut], taker_flow_features(k.iloc[:cut]),
                                  check_freq=False, rtol=1e-9)


def test_zero_volume_bar_gives_nan_not_inf():
    k = _klines(30); k.iloc[5, k.columns.get_loc("Volume")] = 0.0
    assert np.isnan(taker_flow_features(k)["tb_ratio"].iloc[5])


# -------------------------------------------------------------------- funding

def test_funding_sums_only_settlements_inside_each_bar():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    f = pd.Series([0.001, 0.002, 0.003,          # Jan 1: 00:00, 08:00, 16:00
                   0.010,                        # Jan 2 00:00 -> belongs to Jan 2, NOT Jan 1
                   0.100],                       # Jan 3 00:00
                  index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 08:00", "2024-01-01 16:00",
                                        "2024-01-02 00:00", "2024-01-03 00:00"]))
    out = funding_features(f, idx)
    assert out["funding_bar"].iloc[0] == pytest.approx(0.006)
    assert out["funding_bar"].iloc[1] == pytest.approx(0.010)
    assert out["funding_bar"].iloc[2] == pytest.approx(0.100)


def test_bars_before_the_perpetual_launched_are_nan_not_zero():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    f = pd.Series([0.001], index=pd.to_datetime(["2024-01-06 00:00"]))
    out = funding_features(f, idx)
    assert out["funding_bar"].iloc[:5].isna().all() and out["funding_bar"].iloc[5] == pytest.approx(0.001)


@pytest.mark.parametrize("cut", [40, 77, 110])
def test_funding_features_have_no_lookahead(cut):
    """Features at bar t computed with only the funding events up to the end of bar t must match."""
    full_f, idx = _funding(), pd.date_range("2024-01-01", periods=120, freq="D")
    full = funding_features(full_f, idx)
    end_of_last_bar = idx[cut - 1] + pd.Timedelta(days=1)
    truncated = funding_features(full_f[full_f.index < end_of_last_bar], idx[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], truncated, check_freq=False, rtol=1e-9)


def test_empty_funding_gives_nan_columns_not_a_crash():
    out = funding_features(pd.Series(dtype=float), pd.date_range("2024-01-01", periods=5))
    assert out["funding_bar"].isna().all() and len(out) == 5


# -------------------------------------------------------------------- fetchers

def _resp(payload, status=200):
    r = MagicMock(); r.status_code = status; r.json.return_value = payload; r.text = str(payload)[:50]
    r.raise_for_status = MagicMock()
    return r


def test_fetch_klines_paginates_and_parses_taker_volume(tmp_path, monkeypatch):
    monkeypatch.setattr(ofd, "CACHE_DIR", str(tmp_path))
    day = 86_400_000
    base = 1_600_000_000_000
    def row(i):
        return [base + i * day, "1", "2", "0.5", "1.5", "10", base + (i + 1) * day - 1, "15", 100, "4", "6", "0"]
    pages = [[row(i) for i in range(1000)], [row(i) for i in range(1000, 1050)]]
    with patch.object(ofd.requests, "get", side_effect=[_resp(p) for p in pages]) as g, \
         patch.object(ofd.time, "sleep"):
        df = fetch_klines("XYZUSDT", days=1050)
    assert g.call_count == 2 and len(df) in (1049, 1050)        # newest (possibly unfinished) bar may be dropped
    assert {"Close", "Volume", "Trades", "TakerBuyBase"} <= set(df.columns)
    assert df["TakerBuyBase"].iloc[0] == 4.0 and df.index.is_monotonic_increasing


def test_fetch_klines_uses_the_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ofd, "CACHE_DIR", str(tmp_path))
    _klines(10).to_csv(tmp_path / "klines_CACHED_1d_10.csv")
    with patch.object(ofd.requests, "get") as g:
        df = fetch_klines("CACHED", days=10)
    g.assert_not_called()
    assert len(df) == 10


def test_fetch_funding_parses_and_tolerates_http_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(ofd, "CACHE_DIR", str(tmp_path))
    payload = [{"symbol": "X", "fundingTime": 1_600_000_000_000 + i * 28_800_000, "fundingRate": "0.0001"} for i in range(5)]
    with patch.object(ofd.requests, "get", return_value=_resp(payload)), patch.object(ofd.time, "sleep"):
        s = fetch_funding_rates("XYZUSDT", days=5)
    assert len(s) == 5 and (s == 0.0001).all()
    with patch.object(ofd.requests, "get", return_value=_resp({"code": -1121}, status=400)):
        assert fetch_funding_rates("BADUSDT", days=5).empty
