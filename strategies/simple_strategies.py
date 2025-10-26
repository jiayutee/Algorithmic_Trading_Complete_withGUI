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

    def next(self):
        # Calculate position size
        size = (self.broker.getcash() * self.params.risk_per_trade) / self.data.close[0]
        
        if not self.position:  # No position
            # LONG signal: RSI oversold + MACD bullish crossover
            if self.rsi[0] < self.params.rsi_oversold and self.macd.macd[0] > self.macd.signal[0]:
                if size > 0.0001:
                    self.buy(size=size)
                    self.order_count += 1
                    print(f"LONG SIGNAL: Size={size:.6f}, RSI={self.rsi[0]:.2f}, MACD={self.macd.macd[0]:.4f}")
            
            # SHORT signal: RSI overbought + MACD bearish crossover  
            elif self.rsi[0] > self.params.rsi_overbought and self.macd.macd[0] < self.macd.signal[0]:
                if size > 0.0001:
                    self.sell(size=size)
                    self.order_count += 1
                    print(f"SHORT SIGNAL: Size={size:.6f}, RSI={self.rsi[0]:.2f}, MACD={self.macd.macd[0]:.4f}")

        elif self.position.size > 0:  # Long position
            # Exit long: RSI overbought OR MACD bearish
            if self.rsi[0] > self.params.rsi_overbought or self.macd.macd[0] < self.macd.signal[0]:
                print(f"EXIT LONG: Closing position of {self.position.size:.6f}")
                self.close()
                self.order_count += 1

        elif self.position.size < 0:  # Short position
            # Exit short: RSI oversold OR MACD bullish
            if self.rsi[0] < self.params.rsi_oversold or self.macd.macd[0] > self.macd.signal[0]:
                print(f"EXIT SHORT: Closing position of {abs(self.position.size):.6f}")
                self.close()
                self.order_count += 1

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.signals.append({
                    'date': self.data.datetime.datetime(0), 
                    'type': 'buy', 
                    'price': order.executed.price, 
                    'qty': order.executed.size
                })
                print(f"LONG EXECUTED: {order.executed.size:.6f} @ {order.executed.price:.2f}")
            
            elif order.issell():
                # Determine if this is a short open or long close
                if not self.position or self.position.size <= 0:  # Opening short
                    self.signals.append({
                        'date': self.data.datetime.datetime(0), 
                        'type': 'sell_short', 
                        'price': order.executed.price, 
                        'qty': order.executed.size
                    })
                    print(f"SHORT EXECUTED: {order.executed.size:.6f} @ {order.executed.price:.2f}")
                else:  # Closing long
                    self.signals.append({
                        'date': self.data.datetime.datetime(0), 
                        'type': 'sell', 
                        'price': order.executed.price, 
                        'qty': order.executed.size
                    })
                    print(f"LONG CLOSED: {order.executed.size:.6f} @ {order.executed.price:.2f}")

    def stop(self):
        print(f"Strategy finished. Total orders: {self.order_count}, Total signals: {len(self.signals)}")

class EMACrossoverStrategy(bt.Strategy):
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

    def next(self):
        # Calculate position size
        size = (self.broker.getcash() * self.params.risk_per_trade) / self.data.close[0]
        
        if not self.position:  # No position
            # LONG signal: EMA crossover up
            if self.crossover > 0:
                if size > 0.0001:
                    self.buy(size=size)
                    self.order_count += 1
                    print(f"LONG SIGNAL: Size={size:.6f}, EMA12={self.ema_short[0]:.2f}, EMA26={self.ema_long[0]:.2f}")
            
            # SHORT signal: EMA crossover down
            elif self.crossover < 0:
                if size > 0.0001:
                    self.sell(size=size)
                    self.order_count += 1
                    print(f"SHORT SIGNAL: Size={size:.6f}, EMA12={self.ema_short[0]:.2f}, EMA26={self.ema_long[0]:.2f}")

        elif self.position.size > 0:  # Long position
            # Exit long when crossover turns negative
            if self.crossover < 0:
                print(f"EXIT LONG: Closing position of {self.position.size:.6f}")
                self.close()
                self.order_count += 1

        elif self.position.size < 0:  # Short position
            # Exit short when crossover turns positive
            if self.crossover > 0:
                print(f"EXIT SHORT: Closing position of {abs(self.position.size):.6f}")
                self.close()
                self.order_count += 1

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.signals.append({
                    'date': self.data.datetime.datetime(0), 
                    'type': 'buy', 
                    'price': order.executed.price, 
                    'qty': order.executed.size
                })
                print(f"LONG EXECUTED: {order.executed.size:.6f} @ {order.executed.price:.2f}")
            
            elif order.issell():
                if not self.position or self.position.size <= 0:  # Opening short
                    self.signals.append({
                        'date': self.data.datetime.datetime(0), 
                        'type': 'sell_short', 
                        'price': order.executed.price, 
                        'qty': order.executed.size
                    })
                    print(f"SHORT EXECUTED: {order.executed.size:.6f} @ {order.executed.price:.2f}")
                else:  # Closing long
                    self.signals.append({
                        'date': self.data.datetime.datetime(0), 
                        'type': 'sell', 
                        'price': order.executed.price, 
                        'qty': order.executed.size
                    })
                    print(f"LONG CLOSED: {order.executed.size:.6f} @ {order.executed.price:.2f}")

    def stop(self):
        print(f"Strategy finished. Total orders: {self.order_count}, Total signals: {len(self.signals)}")


class StochasticStrategy(bt.Strategy):
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

    def next(self):
        # Calculate position size
        size = (self.broker.getcash() * self.params.risk_per_trade) / self.data.close[0]
        
        if not self.position:  # No position
            # LONG signal: K crosses above D in oversold territory
            if (self.k_cross_d > 0 and 
                self.k_line[0] < self.params.oversold and 
                self.k_line[-1] <= self.d_line[-1]):
                if size > 0.0001:
                    self.buy(size=size)
                    self.order_count += 1
                    print(f"LONG SIGNAL: Size={size:.6f}, K={self.k_line[0]:.2f}, D={self.d_line[0]:.2f}")
            
            # SHORT signal: K crosses below D in overbought territory
            elif (self.k_cross_d < 0 and 
                  self.k_line[0] > self.params.overbought and 
                  self.k_line[-1] >= self.d_line[-1]):
                if size > 0.0001:
                    self.sell(size=size)
                    self.order_count += 1
                    print(f"SHORT SIGNAL: Size={size:.6f}, K={self.k_line[0]:.2f}, D={self.d_line[0]:.2f}")

        elif self.position.size > 0:  # Long position
            # Exit long: K crosses below D in overbought territory
            if (self.k_cross_d < 0 and 
                self.k_line[0] > self.params.overbought and 
                self.k_line[-1] >= self.d_line[-1]):
                print(f"EXIT LONG: Closing position of {self.position.size:.6f}")
                self.close()
                self.order_count += 1

        elif self.position.size < 0:  # Short position
            # Exit short: K crosses above D in oversold territory
            if (self.k_cross_d > 0 and 
                self.k_line[0] < self.params.oversold and 
                self.k_line[-1] <= self.d_line[-1]):
                print(f"EXIT SHORT: Closing position of {abs(self.position.size):.6f}")
                self.close()
                self.order_count += 1

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.signals.append({
                    'date': self.data.datetime.datetime(0), 
                    'type': 'buy', 
                    'price': order.executed.price, 
                    'qty': order.executed.size
                })
                print(f"LONG EXECUTED: {order.executed.size:.6f} @ {order.executed.price:.2f}")
            
            elif order.issell():
                if not self.position or self.position.size <= 0:  # Opening short
                    self.signals.append({
                        'date': self.data.datetime.datetime(0), 
                        'type': 'sell_short', 
                        'price': order.executed.price, 
                        'qty': order.executed.size
                    })
                    print(f"SHORT EXECUTED: {order.executed.size:.6f} @ {order.executed.price:.2f}")
                else:  # Closing long
                    self.signals.append({
                        'date': self.data.datetime.datetime(0), 
                        'type': 'sell', 
                        'price': order.executed.price, 
                        'qty': order.executed.size
                    })
                    print(f"LONG CLOSED: {order.executed.size:.6f} @ {order.executed.price:.2f}")

    def stop(self):
        print(f"Strategy finished. Total orders: {self.order_count}, Total signals: {len(self.signals)}")