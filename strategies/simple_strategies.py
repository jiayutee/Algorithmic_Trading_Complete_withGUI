import backtrader as bt
from core.logger import get_logger
from core.trade_rationale import RationaleMixin

logger = get_logger(__name__)


class MACD_RSI_Strategy(RationaleMixin, bt.Strategy):
    params = (
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('risk_per_trade', 0.1)  # 10% risk per trade
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )
        self.signals = []
        self.order_count = 0
        self.closed_trades = []       # populated in notify_trade(); consumed by
                                       # core/backtester.py::_generate_report() for
                                       # per-trade P&L / cumulative_pnl
        self._closing_long = False   # True when close() was called to exit a long
        self._closing_short = False  # True when close() was called to exit a short

    def _macd_state(self):
        return {
            "rsi": self.rsi[0],
            "macd": self.macd.macd[0],
            "macd_signal": self.macd.signal[0],
            "close": self.data.close[0],
        }

    def _thresholds(self):
        return {
            "rsi_oversold": self.params.rsi_oversold,
            "rsi_overbought": self.params.rsi_overbought,
        }

    def next(self):
        # Calculate position size
        size = (self.broker.getcash() * self.params.risk_per_trade) / self.data.close[0]
        rsi, macd, sig = self.rsi[0], self.macd.macd[0], self.macd.signal[0]

        if not self.position:  # No position
            # LONG signal: RSI oversold + MACD bullish crossover
            if rsi < self.params.rsi_oversold and macd > sig:
                if size > 0.0001:
                    self._set_rationale(
                        action="open_long", strategy="MACD_RSI", signal="rsi_oversold_macd_bullish",
                        summary=(f"Opened LONG: RSI {rsi:.1f} < {self.params.rsi_oversold} (oversold) "
                                 f"and MACD {macd:.4f} above its signal {sig:.4f} (bullish)"),
                        features=self._macd_state(), thresholds=self._thresholds())
                    self.buy(size=size)
                    self.order_count += 1
                    logger.debug("LONG SIGNAL: Size=%.6f, RSI=%.2f, MACD=%.4f", size, self.rsi[0], self.macd.macd[0])

            # SHORT signal: RSI overbought + MACD bearish crossover
            elif rsi > self.params.rsi_overbought and macd < sig:
                if size > 0.0001:
                    self._set_rationale(
                        action="open_short", strategy="MACD_RSI", signal="rsi_overbought_macd_bearish",
                        summary=(f"Opened SHORT: RSI {rsi:.1f} > {self.params.rsi_overbought} (overbought) "
                                 f"and MACD {macd:.4f} below its signal {sig:.4f} (bearish)"),
                        features=self._macd_state(), thresholds=self._thresholds())
                    self.sell(size=size)
                    self.order_count += 1
                    logger.debug("SHORT SIGNAL: Size=%.6f, RSI=%.2f, MACD=%.4f", size, self.rsi[0], self.macd.macd[0])

        elif self.position.size > 0:  # Long position
            # Exit long: RSI overbought OR MACD bearish
            if rsi > self.params.rsi_overbought or macd < sig:
                reasons = []
                if rsi > self.params.rsi_overbought:
                    reasons.append(f"RSI {rsi:.1f} > {self.params.rsi_overbought} (overbought)")
                if macd < sig:
                    reasons.append(f"MACD {macd:.4f} fell below its signal {sig:.4f} (bearish)")
                self._set_rationale(
                    action="close_long", strategy="MACD_RSI", signal="exit_long",
                    summary="Closed LONG: " + " and ".join(reasons),
                    features=self._macd_state(), thresholds=self._thresholds())
                logger.debug("EXIT LONG: Closing position of %.6f", self.position.size)
                self._closing_long = True
                self.close()
                self.order_count += 1

        elif self.position.size < 0:  # Short position
            # Exit short: RSI oversold OR MACD bullish
            if rsi < self.params.rsi_oversold or macd > sig:
                reasons = []
                if rsi < self.params.rsi_oversold:
                    reasons.append(f"RSI {rsi:.1f} < {self.params.rsi_oversold} (oversold)")
                if macd > sig:
                    reasons.append(f"MACD {macd:.4f} rose above its signal {sig:.4f} (bullish)")
                self._set_rationale(
                    action="close_short", strategy="MACD_RSI", signal="exit_short",
                    summary="Closed SHORT: " + " and ".join(reasons),
                    features=self._macd_state(), thresholds=self._thresholds())
                logger.debug("EXIT SHORT: Closing position of %.6f", abs(self.position.size))
                self._closing_short = True
                self.close()
                self.order_count += 1

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                if self._closing_short:
                    self._closing_short = False
                    self.signals.append({
                        'date': self.data.datetime.datetime(0),
                        'type': 'buy_cover',
                        'price': order.executed.price,
                        'qty': order.executed.size
                    })
                    logger.debug("SHORT CLOSED: %.6f @ %.2f", order.executed.size, order.executed.price)
                else:
                    self.signals.append({
                        'date': self.data.datetime.datetime(0),
                        'type': 'buy',
                        'price': order.executed.price,
                        'qty': order.executed.size
                    })
                    logger.debug("LONG EXECUTED: %.6f @ %.2f", order.executed.size, order.executed.price)

            elif order.issell():
                if self._closing_long:
                    self._closing_long = False
                    self.signals.append({
                        'date': self.data.datetime.datetime(0),
                        'type': 'sell',
                        'price': order.executed.price,
                        'qty': order.executed.size
                    })
                    logger.debug("LONG CLOSED: %.6f @ %.2f", order.executed.size, order.executed.price)
                else:
                    self.signals.append({
                        'date': self.data.datetime.datetime(0),
                        'type': 'sell_short',
                        'price': order.executed.price,
                        'qty': order.executed.size
                    })
                    logger.debug("SHORT EXECUTED: %.6f @ %.2f", order.executed.size, order.executed.price)
            self._attach_rationale_to_last_signal()

    def notify_trade(self, trade):
        if trade.isclosed:
            self.closed_trades.append(trade)

    def stop(self):
        logger.info("Strategy finished. Total orders: %d, Total signals: %d", self.order_count, len(self.signals))

class EMACrossoverStrategy(RationaleMixin, bt.Strategy):
    params = (
        ('ema_short', 12),
        ('ema_long', 26),
        ('risk_per_trade', 0.1)  # 10% risk per trade
    )

    def __init__(self):
        self.ema_short = bt.indicators.EMA(self.data.close, period=self.params.ema_short)
        self.ema_long = bt.indicators.EMA(self.data.close, period=self.params.ema_long)
        self.crossover = bt.indicators.CrossOver(self.ema_short, self.ema_long)
        self.signals = []
        self.order_count = 0
        self.closed_trades = []       # populated in notify_trade(); consumed by
                                       # core/backtester.py::_generate_report() for
                                       # per-trade P&L / cumulative_pnl
        self._closing_long = False
        self._closing_short = False

    def _ema_state(self):
        return {
            "ema_short": self.ema_short[0],
            "ema_long": self.ema_long[0],
            "close": self.data.close[0],
        }

    def _params_dict(self):
        return {"ema_short_period": self.params.ema_short, "ema_long_period": self.params.ema_long}

    def next(self):
        # Calculate position size
        size = (self.broker.getcash() * self.params.risk_per_trade) / self.data.close[0]
        short_p, long_p = self.params.ema_short, self.params.ema_long
        es, el = self.ema_short[0], self.ema_long[0]

        if not self.position:  # No position
            # LONG signal: EMA crossover up
            if self.crossover > 0:
                if size > 0.0001:
                    self._set_rationale(
                        action="open_long", strategy="EMA_Crossover", signal="ema_cross_up",
                        summary=f"Opened LONG: EMA{short_p} {es:.2f} crossed above EMA{long_p} {el:.2f} (bullish trend)",
                        features=self._ema_state(), thresholds=self._params_dict())
                    self.buy(size=size)
                    self.order_count += 1
                    logger.debug("LONG SIGNAL: Size=%.6f, EMA12=%.2f, EMA26=%.2f", size, self.ema_short[0], self.ema_long[0])

            # SHORT signal: EMA crossover down
            elif self.crossover < 0:
                if size > 0.0001:
                    self._set_rationale(
                        action="open_short", strategy="EMA_Crossover", signal="ema_cross_down",
                        summary=f"Opened SHORT: EMA{short_p} {es:.2f} crossed below EMA{long_p} {el:.2f} (bearish trend)",
                        features=self._ema_state(), thresholds=self._params_dict())
                    self.sell(size=size)
                    self.order_count += 1
                    logger.debug("SHORT SIGNAL: Size=%.6f, EMA12=%.2f, EMA26=%.2f", size, self.ema_short[0], self.ema_long[0])

        elif self.position.size > 0:  # Long position
            # Exit long when crossover turns negative
            if self.crossover < 0:
                self._set_rationale(
                    action="close_long", strategy="EMA_Crossover", signal="ema_cross_down",
                    summary=f"Closed LONG: EMA{short_p} {es:.2f} crossed below EMA{long_p} {el:.2f} (trend reversed)",
                    features=self._ema_state(), thresholds=self._params_dict())
                logger.debug("EXIT LONG: Closing position of %.6f", self.position.size)
                self._closing_long = True
                self.close()
                self.order_count += 1

        elif self.position.size < 0:  # Short position
            # Exit short when crossover turns positive
            if self.crossover > 0:
                self._set_rationale(
                    action="close_short", strategy="EMA_Crossover", signal="ema_cross_up",
                    summary=f"Closed SHORT: EMA{short_p} {es:.2f} crossed above EMA{long_p} {el:.2f} (trend reversed)",
                    features=self._ema_state(), thresholds=self._params_dict())
                logger.debug("EXIT SHORT: Closing position of %.6f", abs(self.position.size))
                self._closing_short = True
                self.close()
                self.order_count += 1

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                if self._closing_short:
                    self._closing_short = False
                    self.signals.append({
                        'date': self.data.datetime.datetime(0),
                        'type': 'buy_cover',
                        'price': order.executed.price,
                        'qty': order.executed.size
                    })
                    logger.debug("SHORT CLOSED: %.6f @ %.2f", order.executed.size, order.executed.price)
                else:
                    self.signals.append({
                        'date': self.data.datetime.datetime(0),
                        'type': 'buy',
                        'price': order.executed.price,
                        'qty': order.executed.size
                    })
                    logger.debug("LONG EXECUTED: %.6f @ %.2f", order.executed.size, order.executed.price)
            elif order.issell():
                if self._closing_long:
                    self._closing_long = False
                    self.signals.append({
                        'date': self.data.datetime.datetime(0),
                        'type': 'sell',
                        'price': order.executed.price,
                        'qty': order.executed.size
                    })
                    logger.debug("LONG CLOSED: %.6f @ %.2f", order.executed.size, order.executed.price)
                else:
                    self.signals.append({
                        'date': self.data.datetime.datetime(0),
                        'type': 'sell_short',
                        'price': order.executed.price,
                        'qty': order.executed.size
                    })
                    logger.debug("SHORT EXECUTED: %.6f @ %.2f", order.executed.size, order.executed.price)
            self._attach_rationale_to_last_signal()

    def notify_trade(self, trade):
        if trade.isclosed:
            self.closed_trades.append(trade)

    def stop(self):
        logger.info("Strategy finished. Total orders: %d, Total signals: %d", self.order_count, len(self.signals))


class StochasticStrategy(RationaleMixin, bt.Strategy):
    params = (
        ('k_period', 14),
        ('d_period', 3),
        ('oversold', 20),
        ('overbought', 80),
        ('risk_per_trade', 0.1)  # 10% risk per trade
    )

    def __init__(self):
        self.stochastic = bt.indicators.Stochastic(self.data,
                                                    period=self.params.k_period,
                                                    period_dslow=self.params.d_period)
        self.k_line = self.stochastic.percK
        self.d_line = self.stochastic.percD
        self.k_cross_d = bt.indicators.CrossOver(self.k_line, self.d_line)
        self.signals = []
        self.order_count = 0
        self.closed_trades = []       # populated in notify_trade(); consumed by
                                       # core/backtester.py::_generate_report() for
                                       # per-trade P&L / cumulative_pnl
        self._closing_long = False
        self._closing_short = False

    def _sto_state(self):
        return {
            "stoch_k": self.k_line[0],
            "stoch_d": self.d_line[0],
            "close": self.data.close[0],
        }

    def _thresholds(self):
        return {"oversold": self.params.oversold, "overbought": self.params.overbought}

    def next(self):
        # Calculate position size
        size = (self.broker.getcash() * self.params.risk_per_trade) / self.data.close[0]
        k, d = self.k_line[0], self.d_line[0]
        os_, ob = self.params.oversold, self.params.overbought

        if not self.position:  # No position
            # LONG signal: K crosses above D in oversold territory
            if (self.k_cross_d > 0 and
                self.k_line[0] < self.params.oversold and
                self.k_line[-1] <= self.d_line[-1]):
                if size > 0.0001:
                    self._set_rationale(
                        action="open_long", strategy="Stochastic", signal="k_cross_above_d_oversold",
                        summary=f"Opened LONG: %K {k:.1f} crossed above %D {d:.1f} while below {os_} (oversold)",
                        features=self._sto_state(), thresholds=self._thresholds())
                    self.buy(size=size)
                    self.order_count += 1
                    logger.debug("LONG SIGNAL: Size=%.6f, K=%.2f, D=%.2f", size, self.k_line[0], self.d_line[0])

            # SHORT signal: K crosses below D in overbought territory
            elif (self.k_cross_d < 0 and
                  self.k_line[0] > self.params.overbought and
                  self.k_line[-1] >= self.d_line[-1]):
                if size > 0.0001:
                    self._set_rationale(
                        action="open_short", strategy="Stochastic", signal="k_cross_below_d_overbought",
                        summary=f"Opened SHORT: %K {k:.1f} crossed below %D {d:.1f} while above {ob} (overbought)",
                        features=self._sto_state(), thresholds=self._thresholds())
                    self.sell(size=size)
                    self.order_count += 1
                    logger.debug("SHORT SIGNAL: Size=%.6f, K=%.2f, D=%.2f", size, self.k_line[0], self.d_line[0])

        elif self.position.size > 0:  # Long position
            # Exit long: K crosses below D in overbought territory
            if (self.k_cross_d < 0 and
                self.k_line[0] > self.params.overbought and
                self.k_line[-1] >= self.d_line[-1]):
                self._set_rationale(
                    action="close_long", strategy="Stochastic", signal="k_cross_below_d_overbought",
                    summary=f"Closed LONG: %K {k:.1f} crossed below %D {d:.1f} while above {ob} (overbought)",
                    features=self._sto_state(), thresholds=self._thresholds())
                logger.debug("EXIT LONG: Closing position of %.6f", self.position.size)
                self._closing_long = True
                self.close()
                self.order_count += 1

        elif self.position.size < 0:  # Short position
            # Exit short: K crosses above D in oversold territory
            if (self.k_cross_d > 0 and
                self.k_line[0] < self.params.oversold and
                self.k_line[-1] <= self.d_line[-1]):
                self._set_rationale(
                    action="close_short", strategy="Stochastic", signal="k_cross_above_d_oversold",
                    summary=f"Closed SHORT: %K {k:.1f} crossed above %D {d:.1f} while below {os_} (oversold)",
                    features=self._sto_state(), thresholds=self._thresholds())
                logger.debug("EXIT SHORT: Closing position of %.6f", abs(self.position.size))
                self._closing_short = True
                self.close()
                self.order_count += 1

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                if self._closing_short:
                    self._closing_short = False
                    self.signals.append({
                        'date': self.data.datetime.datetime(0),
                        'type': 'buy_cover',
                        'price': order.executed.price,
                        'qty': order.executed.size
                    })
                    logger.debug("SHORT CLOSED: %.6f @ %.2f", order.executed.size, order.executed.price)
                else:
                    self.signals.append({
                        'date': self.data.datetime.datetime(0),
                        'type': 'buy',
                        'price': order.executed.price,
                        'qty': order.executed.size
                    })
                    logger.debug("LONG EXECUTED: %.6f @ %.2f", order.executed.size, order.executed.price)
            elif order.issell():
                if self._closing_long:
                    self._closing_long = False
                    self.signals.append({
                        'date': self.data.datetime.datetime(0),
                        'type': 'sell',
                        'price': order.executed.price,
                        'qty': order.executed.size
                    })
                    logger.debug("LONG CLOSED: %.6f @ %.2f", order.executed.size, order.executed.price)
                else:
                    self.signals.append({
                        'date': self.data.datetime.datetime(0),
                        'type': 'sell_short',
                        'price': order.executed.price,
                        'qty': order.executed.size
                    })
                    logger.debug("SHORT EXECUTED: %.6f @ %.2f", order.executed.size, order.executed.price)
            self._attach_rationale_to_last_signal()

    def notify_trade(self, trade):
        if trade.isclosed:
            self.closed_trades.append(trade)

    def stop(self):
        logger.info("Strategy finished. Total orders: %d, Total signals: %d", self.order_count, len(self.signals))