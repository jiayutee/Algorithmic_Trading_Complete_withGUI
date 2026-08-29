try:
    import ccxt
    _CCXT_AVAILABLE = True
except ImportError:
    _CCXT_AVAILABLE = False
    ccxt = None


class MexcConnector:
    """MEXC connector supporting both live (via ccxt) and local paper-trading modes.

    There is no dedicated ``mexc-python`` client in requirements.txt. This
    codebase already talks to other exchanges without a first-party SDK
    through ccxt (see ``KuCoinConnector``), so this connector follows the
    same convention rather than introducing a new dependency.

    Parameters
    ----------
    api_key : str
        MEXC API key. Ignored (may be empty string) when ``paper_mode=True``.
    secret_key : str
        MEXC API secret. Ignored when ``paper_mode=True``.
    paper_mode : bool
        When *True* (the default), no private/authenticated network calls are
        made. All orders are simulated locally: fills happen immediately at
        the supplied price and an internal portfolio / P&L ledger is
        maintained — this mirrors ``BinanceConnector``/``KuCoinConnector``'s
        ``paper_mode`` exactly (same ledger shape, same
        ``place_order``/``get_portfolio``/``get_position`` contract) so
        callers can treat all three connectors interchangeably.

    Notes
    -----
    Unlike ``KuCoinConnector``, MEXC via ccxt needs only ``apiKey`` and
    ``secret`` — no passphrase/password field.

    ``get_historical_klines`` fetches OHLCV candles from MEXC's *public*
    market-data endpoint via ccxt, which requires no API credentials, so it
    works in both paper and live mode (same as KuCoin; unlike
    ``BinanceConnector.get_historical_klines``, which is gated behind
    ``paper_mode=False`` because python-binance's historical klines call
    uses an authenticated client).
    """

    def __init__(self, api_key: str = "", secret_key: str = "", paper_mode: bool = True):
        self.paper_mode = paper_mode

        # Public client for historical OHLCV — needs no credentials, so it's
        # created whenever ccxt is available regardless of paper_mode.
        self._public_client = ccxt.mexc() if _CCXT_AVAILABLE else None

        if paper_mode:
            # Local simulation — no network required
            self.client = None
            # portfolio: symbol -> {"qty": float, "avg_entry_price": float}
            self._positions: dict = {}
            # Running realised P&L (USDT)
            self._realized_pnl: float = 0.0
            # Cash balance (USDT) for the paper account
            self._cash: float = 100_000.0
            # Order history
            self._orders: list = []
        else:
            if not _CCXT_AVAILABLE:
                raise RuntimeError(
                    "ccxt is not installed. Install it with: pip install ccxt"
                )
            try:
                self.client = ccxt.mexc({
                    "apiKey": api_key,
                    "secret": secret_key,
                })
            except Exception as e:
                raise RuntimeError(f"Failed to initialise MEXC client: {e}") from e

    # ------------------------------------------------------------------
    # Paper-trading helpers (mirrors BinanceConnector/KuCoinConnector.place_order)
    # ------------------------------------------------------------------

    def place_order(self, symbol: str, side: str, qty: float, price: float) -> dict:
        """Place a paper-trading order and fill it immediately at *price*.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. ``"BTC/USDT"`` (ccxt/MEXC unified symbol format).
        side : str
            ``"BUY"`` or ``"SELL"`` (case-insensitive).
        qty : float
            Quantity of the base asset to trade.
        price : float
            Execution price in the quote asset (e.g. USDT).

        Returns
        -------
        dict
            Order receipt with keys: symbol, side, qty, price, status,
            realized_pnl (for SELL orders).

        Raises
        ------
        RuntimeError
            If called when ``paper_mode`` is False.
        ValueError
            If *side* is not ``"BUY"`` or ``"SELL"``.
        """
        if not self.paper_mode:
            raise RuntimeError(
                "place_order() is only available in paper_mode. "
                "Use submit_order() for live trading."
            )

        side_upper = side.upper()
        if side_upper not in ("BUY", "SELL"):
            raise ValueError(f"side must be 'BUY' or 'SELL', got '{side}'")

        order_value = qty * price
        realized_pnl_this_trade: float = 0.0

        if side_upper == "BUY":
            # Deduct cost from cash
            self._cash -= order_value
            # Update position
            if symbol in self._positions:
                pos = self._positions[symbol]
                total_qty = pos["qty"] + qty
                total_cost = pos["qty"] * pos["avg_entry_price"] + order_value
                self._positions[symbol] = {
                    "qty": total_qty,
                    "avg_entry_price": total_cost / total_qty,
                }
            else:
                self._positions[symbol] = {
                    "qty": qty,
                    "avg_entry_price": price,
                }
        else:  # SELL
            pos = self._positions.get(symbol)
            sell_qty = min(qty, pos["qty"]) if pos else 0.0
            if sell_qty > 0:
                realized_pnl_this_trade = sell_qty * (price - pos["avg_entry_price"])
                self._realized_pnl += realized_pnl_this_trade
                self._cash += sell_qty * price
                remaining = pos["qty"] - sell_qty
                if remaining < 1e-10:
                    del self._positions[symbol]
                else:
                    self._positions[symbol]["qty"] = remaining

        receipt = {
            "symbol": symbol,
            "side": side_upper,
            "qty": qty,
            "price": price,
            "status": "FILLED",
            "realized_pnl": realized_pnl_this_trade,
        }
        self._orders.append(receipt)
        return receipt

    def get_portfolio(self) -> dict:
        """Return current paper-trading portfolio state.

        Returns
        -------
        dict
            Keys:
            - ``cash``  : remaining USDT balance.
            - ``positions`` : dict of symbol -> {qty, avg_entry_price}.
            - ``realized_pnl`` : cumulative realised P&L in USDT.

        Raises
        ------
        RuntimeError
            If called when ``paper_mode`` is False.
        """
        if not self.paper_mode:
            raise RuntimeError("get_portfolio() is only available in paper_mode.")
        return {
            "cash": self._cash,
            "positions": dict(self._positions),
            "realized_pnl": self._realized_pnl,
        }

    def get_position(self, symbol):
        if self.paper_mode:
            pos = self._positions.get(symbol)
            if not pos:
                return None
            return {
                "symbol": symbol,
                "qty": pos["qty"],
                "side": "long",
                "avg_entry_price": pos["avg_entry_price"],
                "current_price": None,
                "unrealized_pl": None,
                "leverage": 1,
            }

        if not _CCXT_AVAILABLE:
            raise RuntimeError("ccxt is not installed. Install it with: pip install ccxt")
        balance = self.client.fetch_balance()
        base_asset = symbol.split("/")[0] if "/" in symbol else symbol.replace("USDT", "")
        free = balance.get("free", {}).get(base_asset, 0.0)
        total = balance.get("total", {}).get(base_asset, 0.0)
        if not total:
            return None
        ticker = self.client.fetch_ticker(symbol)
        return {
            "symbol": symbol,
            "qty": total,
            "side": "long",
            "avg_entry_price": None,  # MEXC spot balances don't expose avg entry price
            "current_price": ticker.get("last"),
            "unrealized_pl": None,
            "leverage": 1,
        }

    # ------------------------------------------------------------------
    # Live methods (via ccxt)
    # ------------------------------------------------------------------

    def submit_order(self, symbol, qty, side, order_type="market"):
        """Submit a live order to MEXC via ccxt. Not available in paper_mode."""
        if self.paper_mode:
            raise RuntimeError(
                "submit_order() is not available in paper_mode. "
                "Use place_order(symbol, side, qty, price) instead."
            )
        order_side = "buy" if side.lower() in ("long", "buy") else "sell"
        return self.client.create_order(symbol=symbol, type=order_type, side=order_side, amount=qty)

    def get_historical_klines(self, symbol: str, timeframe: str = "1d", since=None, limit: int = 100):
        """Fetch historical OHLCV candles from MEXC's public market-data API.

        Parameters
        ----------
        symbol : str
            Unified ccxt symbol, e.g. ``"BTC/USDT"``.
        timeframe : str
            ccxt timeframe string. MEXC supports: 1m, 5m, 15m, 30m, 1h, 4h,
            8h, 1d, 1w, 1M.
        since : int, optional
            Start time in milliseconds since epoch.
        limit : int
            Maximum number of candles to return.

        Returns
        -------
        list
            List of ``[timestamp, open, high, low, close, volume]`` rows, as
            returned by ``ccxt.mexc().fetch_ohlcv``.

        Raises
        ------
        RuntimeError
            If ccxt is not installed.
        """
        if self._public_client is None:
            raise RuntimeError("ccxt is not installed. Install it with: pip install ccxt")
        return self._public_client.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
