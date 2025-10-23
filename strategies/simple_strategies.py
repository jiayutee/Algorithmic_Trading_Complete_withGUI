#Classic Strategies

import backtrader as bt


class MACD_RSI_Strategy(bt.Strategy):
    params = (
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )
        self.signals = []
        self.last_buy_price = 0
        self.cumulative_pnl = 0

    def next(self):
        if not self.position:
            if self.rsi[0] > self.p.rsi_oversold and self.macd.macd[0] > self.macd.signal[0]:
                self.buy()
        else:
            if self.rsi[0] > self.p.rsi_overbought or self.macd.macd[0] < self.macd.signal[0]:
                self.close()

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.signals.append({'date': self.data.datetime.date(), 'type': 'buy', 'price': order.executed.price})
                self.last_buy_price = order.executed.price
                print(f"BUY: {order.executed.size} @ {order.executed.price:.2f}")
            elif order.issell():
                self.signals.append({'date': self.data.datetime.date(), 'type': 'sell', 'price': order.executed.price})
                pnl = (order.executed.price - self.last_buy_price) * abs(order.executed.size)
                self.cumulative_pnl += pnl
                print(f"SELL: {order.executed.size} @ {order.executed.price:.2f}, PnL: {pnl:.2f}, Cumulative PnL: {self.cumulative_pnl:.2f}")
                self.last_buy_price = 0


class EMACrossoverStrategy(bt.Strategy):
    params = (
        ('ema_short', 12),
        ('ema_long', 26)
    )

    def __init__(self):
        self.ema_short = bt.indicators.EMA(self.data.close, period=self.p.ema_short)
        self.ema_long = bt.indicators.EMA(self.data.close, period=self.p.ema_long)
        self.crossover = bt.indicators.CrossOver(self.ema_short, self.ema_long)
        self.signals = []
        self.last_buy_price = 0
        self.cumulative_pnl = 0

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.signals.append({'date': self.data.datetime.date(), 'type': 'buy', 'price': order.executed.price})
                self.last_buy_price = order.executed.price
                print(f"BUY: {order.executed.size} @ {order.executed.price:.2f}")
            elif order.issell():
                self.signals.append({'date': self.data.datetime.date(), 'type': 'sell', 'price': order.executed.price})
                pnl = (order.executed.price - self.last_buy_price) * abs(order.executed.size)
                self.cumulative_pnl += pnl
                print(f"SELL: {order.executed.size} @ {order.executed.price:.2f}, PnL: {pnl:.2f}, Cumulative PnL: {self.cumulative_pnl:.2f}")
                self.last_buy_price = 0


class StochasticStrategy(bt.Strategy):
    params = (
        ('k_period', 14),
        ('d_period', 3),
        ('oversold', 20),
        ('overbought', 80)
    )

    def __init__(self):
        self.stochastic = bt.indicators.Stochastic(self.data,
                                                    period=self.p.k_period,
                                                    period_dslow=self.p.d_period)
        self.k_line = self.stochastic.percK
        self.d_line = self.stochastic.percD
        self.signals = []
        self.last_buy_price = 0
        self.cumulative_pnl = 0

    def next(self):
        if not self.position:
            if self.k_line[0] > self.d_line[0] and self.k_line[-1] <= self.d_line[-1] and self.k_line[0] < self.p.oversold:
                self.buy()
        else:
            if self.k_line[0] < self.d_line[0] and self.k_line[-1] >= self.d_line[-1] and self.k_line[0] > self.p.overbought:
                self.close()

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.signals.append({'date': self.data.datetime.date(), 'type': 'buy', 'price': order.executed.price})
                self.last_buy_price = order.executed.price
                print(f"BUY: {order.executed.size} @ {order.executed.price:.2f}")
            elif order.issell():
                self.signals.append({'date': self.data.datetime.date(), 'type': 'sell', 'price': order.executed.price})
                pnl = (order.executed.price - self.last_buy_price) * abs(order.executed.size)
                self.cumulative_pnl += pnl
                print(f"SELL: {order.executed.size} @ {order.executed.price:.2f}, PnL: {pnl:.2f}, Cumulative PnL: {self.cumulative_pnl:.2f}")
                self.last_buy_price = 0
