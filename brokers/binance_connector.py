try:
    from binance.exceptions import BinanceAPIException
    from binance.client import Client
    _BINANCE_CLIENT_AVAILABLE = True
except ImportError:
    _BINANCE_CLIENT_AVAILABLE = False
    BinanceAPIException = Exception


class BinanceConnector:
    """Binance connector supporting both live/testnet and local paper-trading modes.

    Parameters
    ----------
    api_key : str
        Binance API key.  Ignored (may be empty string) when ``paper_mode=True``.
    secret_key : str
        Binance secret key.  Ignored when ``paper_mode=True``.
    paper : bool
        When *True* and ``paper_mode`` is *False*, the connector uses Binance
        testnet endpoints (``testnet=True`` is passed to the ``Client``).
    paper_mode : bool
        When *True* (the default), no network connection is established.  All
        orders are simulated locally: fills happen immediately at the supplied
        price and an internal portfolio / P&L ledger is maintained.  This is
        safe to use without real API credentials.
    """

    def __init__(self, api_key: str = "", secret_key: str = "", paper: bool = True,
                 paper_mode: bool = True):
        self.paper_mode = paper_mode

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
            if not _BINANCE_CLIENT_AVAILABLE:
                raise RuntimeError(
                    "python-binance is not installed. "
                    "Install it with: pip install python-binance"
                )
            try:
                self.client = Client(api_key, secret_key, testnet=paper)
            except Exception as e:
                raise RuntimeError(f"Failed to initialise Binance client: {e}") from e

    # ------------------------------------------------------------------
    # Paper-trading helpers
    # ------------------------------------------------------------------

    def place_order(self, symbol: str, side: str, qty: float, price: float) -> dict:
        """Place a paper-trading order and fill it immediately at *price*.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. ``"BTCUSDT"``.
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
                "Use submit_order() for live/testnet trading."
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

    # ------------------------------------------------------------------
    # Live / testnet methods (unchanged from original)
    # ------------------------------------------------------------------

    def submit_order(self, symbol, qty, side, order_type='MARKET', futures=True):
        if self.paper_mode:
            # Convenience wrapper: delegate to place_order with a dummy price of 0
            # (not intended for real use — call place_order() directly in paper mode)
            raise RuntimeError(
                "submit_order() is not available in paper_mode. "
                "Use place_order(symbol, side, qty, price) instead."
            )
        order_side = "BUY" if side.lower() in ('long', 'buy') else "SELL"
        if futures:
            return self.client.futures_create_order(
                symbol=symbol,
                quantity=qty,
                side=order_side,
                type=order_type
            )
        else:
            return self.client.create_order(
                symbol=symbol,
                quantity=qty,
                side=order_side,
                type=order_type
            )

    def get_position(self, symbol, futures=True):
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
        if futures:
            positions = self.client.futures_position_information()
            position = next((p for p in positions if p['symbol'] == symbol), None)

            if not position or float(position['positionAmt']) == 0:
                return None

            return {
                'symbol': position['symbol'],
                'qty': float(position['positionAmt']),
                'side': 'long' if float(position['positionAmt']) > 0 else 'short',
                'avg_entry_price': float(position['entryPrice']),
                'current_price': float(position['markPrice']),
                'unrealized_pl': float(position['unRealizedProfit']),
                'leverage': int(position['leverage'])
            }
        else:
            balances = self.client.get_account()['balances']
            asset = symbol.replace('USDT', '') if symbol.endswith('USDT') else symbol
            balance = next((b for b in balances if b['asset'] == asset), None)

            if not balance or (float(balance['free']) == 0 and float(balance['locked']) == 0):
                return None

            return {
                'symbol': asset,
                'qty': float(balance['free']) + float(balance['locked']),
                'side': 'long',  # Spot is always long
                'avg_entry_price': None,  # Binance spot doesn't provide this
                'current_price': float(self.client.get_symbol_ticker(symbol=symbol)['price']),
                'unrealized_pl': None  # Not directly available for spot
            }

    def get_historical_klines(self, symbol, interval, start_str, end_str=None):
        """Get historical klines from Binance"""
        if self.paper_mode:
            raise RuntimeError("get_historical_klines() is not available in paper_mode.")
        return self.client.get_historical_klines(symbol, interval, start_str, end_str)
