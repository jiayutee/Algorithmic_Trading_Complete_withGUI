# core/ta_engine.py
import pandas_ta as ta
import pandas as pd


class TAEngine:
    @staticmethod
    def calculate_rsi(data, window=14):
        """Calculate Relative Strength Index"""
        return ta.rsi(data['Close'], length=window)

    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        macd = ta.macd(data['Close'], fast=fast, slow=slow, signal=signal)
        return {
            'macd_line': macd.iloc[:, 0],  # MACD line
            'signal_line': macd.iloc[:, 1],  # Signal line
            'histogram': macd.iloc[:, 2]  # MACD histogram
        }

    @staticmethod
    def calculate_ema(data, window=20):
        """Calculate Exponential Moving Average"""
        return ta.ema(data['Close'], length=window)

    @staticmethod
    def calculate_stochastic(data, k_window=14, d_window=3):
        """Calculate Stochastic Oscillator"""
        stoch = ta.stoch(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            k=k_window,
            d=d_window
        )
        return {
            'percent_k': stoch.iloc[:, 0],  # %K line
            'percent_d': stoch.iloc[:, 1]  # %D line (signal)
        }