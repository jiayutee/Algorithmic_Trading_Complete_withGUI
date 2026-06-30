# core/ta_engine.py
# Uses the 'ta' library (pip install ta) instead of 'pandas_ta' which requires Python >=3.12.
import pandas as pd
import ta as _ta


class TAEngine:
    @staticmethod
    def calculate_rsi(data, window=14):
        """Calculate Relative Strength Index"""
        return _ta.momentum.RSIIndicator(close=data['Close'], window=window).rsi()

    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        """Calculate MACD"""
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

    @staticmethod
    def calculate_ema(data, window=20):
        """Calculate Exponential Moving Average"""
        return _ta.trend.EMAIndicator(close=data['Close'], window=window).ema_indicator()

    @staticmethod
    def calculate_stochastic(data, k_window=14, d_window=3):
        """Calculate Stochastic Oscillator"""
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
