# core/ta_engine.py
# Uses the 'ta' library (pip install ta) instead of 'pandas_ta' which requires Python >=3.12.
import pandas as pd
try:
    import ta as _ta
    _TA_AVAILABLE = True
except ImportError:
    _ta = None
    _TA_AVAILABLE = False


class TAEngine:
    @staticmethod
    def calculate_rsi(data, window=14):
        """Calculate Relative Strength Index"""
        if _TA_AVAILABLE:
            return _ta.momentum.RSIIndicator(close=data['Close'], window=window).rsi()
        # Pure-pandas fallback
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
        rs = gain / loss.replace(0, float('nan'))
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        if _TA_AVAILABLE:
            macd = _ta.trend.MACD(
                close=data['Close'],
                window_fast=fast,
                window_slow=slow,
                window_sign=signal,
            )
            return {
                'macd_line': macd.macd(),
                'signal_line': macd.macd_signal(),
                'histogram': macd.macd_diff(),
            }
        # Pure-pandas fallback
        ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': macd_line - signal_line,
        }

    @staticmethod
    def calculate_ema(data, window=20):
        """Calculate Exponential Moving Average"""
        if _TA_AVAILABLE:
            return _ta.trend.EMAIndicator(close=data['Close'], window=window).ema_indicator()
        return data['Close'].ewm(span=window, adjust=False).mean()

    @staticmethod
    def calculate_stochastic(data, k_window=14, d_window=3):
        """Calculate Stochastic Oscillator"""
        if _TA_AVAILABLE:
            stoch = _ta.momentum.StochasticOscillator(
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                window=k_window,
                smooth_window=d_window,
            )
            return {
                'percent_k': stoch.stoch(),
                'percent_d': stoch.stoch_signal(),
            }
        # Pure-pandas fallback
        low_k = data['Low'].rolling(k_window).min()
        high_k = data['High'].rolling(k_window).max()
        k = 100 * (data['Close'] - low_k) / (high_k - low_k).replace(0, float('nan'))
        d = k.rolling(d_window).mean()
        return {'percent_k': k, 'percent_d': d}
