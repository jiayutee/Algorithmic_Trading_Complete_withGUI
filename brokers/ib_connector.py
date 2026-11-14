from ib_insync import IB, MarketOrder, LimitOrder, Contract, Option, Future
from typing import Optional, Union


from brokers.execution_guard import guarded_live_order, OPTION_CONTRACT_MULTIPLIER

class IBKRConnector:
    """
    Interactive Brokers Connector for live trading
    Requires IBKR TWS or Gateway running with API access enabled
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
        """
        Initialize connection to TWS/Gateway
        :param host: IP address where TWS/Gateway is running
        :param port: IB defaults -- TWS: 7496 live / 7497 paper; IB Gateway: 4001 live / 4002 paper.
            Ports are user-configurable, so this connector never *assumes* a port means paper: orders always
            go through the live-order guard (brokers/execution_guard.py).
        :param client_id: Client ID for this connection (must be unique per connection)
        """
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connect()

    def connect(self) -> None:
        """Establish connection to TWS/Gateway"""
        if not self.ib.isConnected():
            self.ib.connect(self.host, self.port, self.client_id)

    def disconnect(self) -> None:
        """Disconnect from TWS/Gateway"""
        if self.ib.isConnected():
            self.ib.disconnect()

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

    @guarded_live_order("IBKR")
    def submit_order(
            self,
            symbol: str,
            qty: Union[int, float],
            side: str,
            order_type: str = 'MKT',
            sec_type: str = 'STK',
            currency: str = 'USD',
            exchange: str = 'SMART'
    ) -> dict:
        """
        Submit an order to IBKR
        :param symbol: Ticker symbol
        :param qty: Quantity to trade (positive for buy, negative for sell)
        :param side: 'long' or 'short' (for position) or 'buy'/'sell'
        :param order_type: Order type (MKT, LMT, etc.)
        :param sec_type: Security type (STK, FUT, OPT, etc.)
        :param currency: Currency of the trade
        :param exchange: Exchange where the order should be routed
        :return: Dictionary with order details
        """
        # Create contract
        contract = Contract(
            symbol=symbol,
            secType=sec_type,
            currency=currency,
            exchange=exchange
        )

        # Resolve side to quantity
        if isinstance(side, str):
            side = side.lower()
            if side in ['long', 'buy']:
                qty = abs(qty)
            elif side in ['short', 'sell']:
                qty = -abs(qty)

        # Create and place order
        order = MarketOrder('BUY' if qty > 0 else 'SELL', abs(qty))
        trade = self.ib.placeOrder(contract, order)

        # Wait for execution
        self.ib.sleep(1)  # Give it a moment to process

        return {
            'order_id': trade.order.orderId,
            'symbol': symbol,
            'quantity': qty,
            'side': 'BUY' if qty > 0 else 'SELL',
            'status': trade.orderStatus.status,
            'filled': trade.orderStatus.filled,
            'remaining': trade.orderStatus.remaining,
            'avg_fill_price': trade.orderStatus.avgFillPrice
        }

    # ------------------------------------------------------------------ options and futures (Phases 4.3 / 4.4)
    @staticmethod
    def _asset_class(instrument) -> str:
        ac = getattr(instrument, "asset_class", "")
        return str(getattr(ac, "value", ac)).upper()

    def contract_for(self, instrument):
        """Build the IB contract for an OPTION or FUTURE ``core.instrument.Instrument``. Raises ValueError for anything
        else or for a missing required field, so an incomplete contract is never sent (IB would otherwise guess)."""
        kind = self._asset_class(instrument)
        under = getattr(instrument, "underlying", None)
        expiry = getattr(instrument, "expiry", None)
        if kind == "OPTION":
            right = getattr(getattr(instrument, "option_type", None), "value", getattr(instrument, "option_type", None))
            strike = getattr(instrument, "strike", None)
            if not under or expiry is None or not strike or float(strike) <= 0 or str(right).upper() not in ("CALL", "PUT"):
                raise ValueError("an option needs underlying, expiry, a positive strike and option_type CALL/PUT")
            return Option(symbol=under, lastTradeDateOrContractMonth=expiry.strftime("%Y%m%d"), strike=float(strike),
                          right="C" if str(right).upper() == "CALL" else "P",
                          exchange=getattr(instrument, "exchange", "") or "SMART",
                          multiplier=str(int(OPTION_CONTRACT_MULTIPLIER)), currency=getattr(instrument, "currency", "USD") or "USD")
        if kind == "FUTURE":
            size = getattr(instrument, "contract_size", None)
            exch = getattr(instrument, "exchange", "")
            if not under or expiry is None or not size or float(size) <= 0 or not exch:
                raise ValueError("a future needs underlying, expiry (contract month), a positive contract_size and an exchange")
            return Future(symbol=under, lastTradeDateOrContractMonth=expiry.strftime("%Y%m"), exchange=exch,
                          multiplier=str(int(size) if float(size).is_integer() else size),
                          currency=getattr(instrument, "currency", "USD") or "USD")
        raise ValueError(f"submit_instrument_order handles OPTION and FUTURE only (got {kind or 'unknown'}); use submit_order for stocks")

    def submit_instrument_order(self, instrument, qty: Union[int, float], side: str, price: float,
                                order_type: str = 'MKT') -> dict:
        """Place a single-leg option order (buy/sell call/put) or a futures order for an Instrument.

        ``price`` is REQUIRED: the option premium or futures price. It is the guard's value reference (qty x price x contract
        multiplier, so one option contract = 100 shares) and the limit for ``order_type='LMT'``; for 'MKT' it is only the
        reference and the order still goes at market. Multi-leg spreads are out of scope. Always goes through the live-order guard."""
        contract = self.contract_for(instrument)                    # validates BEFORE the guard so a bad contract is an error, not a dry run
        if str(order_type).upper() not in ("MKT", "LMT"):
            raise ValueError("order_type must be MKT or LMT")
        if not price or float(price) <= 0:
            raise ValueError("price (premium / futures price) is required to value the order")
        return self._place_instrument_order(symbol=instrument.symbol, qty=qty, side=side, instrument=instrument,
                                            order_type=str(order_type).upper(), price=float(price), _contract=contract)

    @guarded_live_order("IBKR")
    def _place_instrument_order(self, symbol, qty, side, instrument, order_type='MKT', price=None, _contract=None) -> dict:
        action = "BUY" if str(side).lower() in ("buy", "long") else "SELL"
        order = LimitOrder(action, abs(qty), price) if order_type == "LMT" else MarketOrder(action, abs(qty))
        trade = self.ib.placeOrder(_contract, order)
        self.ib.sleep(1)
        out = {
            'order_id': trade.order.orderId, 'symbol': symbol,
            'quantity': abs(qty) if action == "BUY" else -abs(qty), 'side': action,
            'status': trade.orderStatus.status, 'filled': trade.orderStatus.filled,
            'remaining': trade.orderStatus.remaining, 'avg_fill_price': trade.orderStatus.avgFillPrice,
            'sec_type': getattr(_contract, 'secType', None),
            'contract': {'underlying': _contract.symbol, 'expiry': _contract.lastTradeDateOrContractMonth,
                         'multiplier': getattr(_contract, 'multiplier', None), 'exchange': _contract.exchange,
                         'strike': getattr(_contract, 'strike', None), 'right': getattr(_contract, 'right', None)},
        }
        return out

    def get_position(self, symbol: str, sec_type: str = 'STK', currency: str = 'USD', instrument=None) -> Optional[dict]:
        """
        Get current position for a symbol
        :param symbol: Ticker symbol
        :param sec_type: Security type
        :param currency: Currency
        :return: Dictionary with position details or None if no position
        """
        positions = self.ib.positions()
        want = self.contract_for(instrument) if instrument is not None else None
        if want is not None:                                        # match the exact contract, not just the underlying
            symbol, sec_type, currency = want.symbol, want.secType, want.currency

        for position in positions:
            contract = position.contract
            if (contract.symbol == symbol and
                    contract.secType == sec_type and
                    contract.currency == currency):
                if want is not None and not self._same_contract(contract, want):
                    continue
                info = {
                    'symbol': symbol,
                    'position': position.position,
                    'avg_cost': position.avgCost,
                    'contract': {
                        'sec_type': sec_type,
                        'currency': currency,
                        'exchange': contract.exchange
                    }
                }
                if sec_type in ('OPT', 'FUT', 'FOP'):
                    info['contract'].update({k: getattr(contract, k, None) for k in
                                             ('lastTradeDateOrContractMonth', 'strike', 'right', 'multiplier')})
                return info
        return None

    @staticmethod
    def _same_contract(held, want) -> bool:
        """Expiry compared on the shared prefix (a future is requested by YYYYMM, IB reports YYYYMMDD); strike/right for options."""
        a, b = str(getattr(held, 'lastTradeDateOrContractMonth', '') or ''), str(want.lastTradeDateOrContractMonth or '')
        if not a or not b or not (a.startswith(b) or b.startswith(a)):
            return False
        if want.secType == 'OPT':
            if abs(float(getattr(held, 'strike', 0) or 0) - float(want.strike)) > 1e-9:
                return False
            if str(getattr(held, 'right', '')).upper()[:1] != str(want.right).upper()[:1]:
                return False
        return True

    def get_positions(self) -> dict:
        """All open positions as {symbol: {qty, avg_cost, sec_type, currency, exchange}} (same keyed shape as the
        other brokers' portfolio entries). Multiple contracts on one symbol (e.g. option legs) are keyed
        ``SYMBOL:secType`` after the first so none is silently overwritten."""
        out: dict = {}
        for position in self.ib.positions():
            c = position.contract
            key = c.symbol if c.symbol not in out else f"{c.symbol}:{c.secType}"
            out[key] = {"qty": float(position.position), "avg_cost": float(position.avgCost),
                        "sec_type": c.secType, "currency": c.currency, "exchange": c.exchange}
        return out

    def get_account_info(self) -> dict:
        """Get basic account information"""
        account = self.ib.accountSummary()
        return {
            'net_liquidation': next((a.value for a in account if a.tag == 'NetLiquidation'), None),
            'buying_power': next((a.value for a in account if a.tag == 'BuyingPower'), None),
            'leverage': next((a.value for a in account if a.tag == 'GrossPositionValue'), None),
            'available_funds': next((a.value for a in account if a.tag == 'AvailableFunds'), None),
            # Margin fields (Phase 4.4), None where IBKR does not report them for this account
            'gross_position_value': next((a.value for a in account if a.tag == 'GrossPositionValue'), None),
            'initial_margin_req': next((a.value for a in account if a.tag == 'InitMarginReq'), None),
            'maintenance_margin_req': next((a.value for a in account if a.tag == 'MaintMarginReq'), None),
            'excess_liquidity': next((a.value for a in account if a.tag == 'ExcessLiquidity'), None),
            'cushion': next((a.value for a in account if a.tag == 'Cushion'), None),
        }