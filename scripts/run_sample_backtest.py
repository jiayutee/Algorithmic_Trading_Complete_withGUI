#!/usr/bin/env python3
"""Simple EMA crossover backtest for AAPL.

Creates a Plotly figure with candlesticks + buy/sell markers and equity curve,
then saves to results/backtest_aapl.html.
"""
import sys
import os
import pandas as pd

# Ensure repo root is on sys.path so `core` imports work when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from core.data_loader import DataLoader


def run_backtest(symbol='AAPL', days=365, interval='1d', start_cash=100000, outpath='results/backtest_aapl.html'):
    dl = DataLoader()
    df = dl.load_data(symbol, days=days, interval=interval)

    # Ensure we have required columns and sort chronologically
    df = df.sort_index().copy()
    df = df.dropna(subset=['Close'])

    # Compute EMAs
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # Detect crossover signals
    df['prev_EMA12'] = df['EMA12'].shift(1)
    df['prev_EMA26'] = df['EMA26'].shift(1)
    df['signal'] = 0
    df.loc[(df['EMA12'] > df['EMA26']) & (df['prev_EMA12'] <= df['prev_EMA26']), 'signal'] = 1
    df.loc[(df['EMA12'] < df['EMA26']) & (df['prev_EMA12'] >= df['prev_EMA26']), 'signal'] = -1

    # Backtest state
    cash = float(start_cash)
    shares = 0
    in_position = False

    equity_curve = []
    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []

    for idx, row in df.iterrows():
        price = float(row['Close'])
        sig = int(row.get('signal', 0))

        # Buy: enter long with all cash when signal=1 and not in position
        if sig == 1 and not in_position:
            num_shares = int(cash // price)
            if num_shares > 0:
                shares = num_shares
                cash -= shares * price
                in_position = True
                buy_dates.append(idx)
                buy_prices.append(price)

        # Sell: exit to cash when signal=-1 and in position
        if sig == -1 and in_position and shares > 0:
            cash += shares * price
            shares = 0
            in_position = False
            sell_dates.append(idx)
            sell_prices.append(price)

        equity = cash + shares * price
        equity_curve.append({'datetime': idx, 'equity': equity})

    # If still holding at the end, record final equity
    if in_position and shares > 0:
        last_price = float(df['Close'].iloc[-1])
        final_value = cash + shares * last_price
    else:
        final_value = cash

    # Prepare plotting
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                 name=f'{symbol}'), row=1, col=1)

    # Add EMAs to price subplot
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA12'], mode='lines', name='EMA12', line=dict(width=1, color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA26'], mode='lines', name='EMA26', line=dict(width=1, color='orange')), row=1, col=1)

    # Buy markers
    if buy_dates:
        fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', name='Buy',
                                 marker=dict(symbol='triangle-up', color='green', size=10)), row=1, col=1)

    # Sell markers
    if sell_dates:
        fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', name='Sell',
                                 marker=dict(symbol='triangle-down', color='red', size=10)), row=1, col=1)

    # Equity curve
    eq_df = pd.DataFrame(equity_curve).set_index('datetime')
    fig.add_trace(go.Scatter(x=eq_df.index, y=eq_df['equity'], mode='lines', name='Equity', line=dict(color='purple')), row=2, col=1)

    fig.update_layout(title_text=f'{symbol} EMA12/EMA26 Crossover Backtest — Final Value: ${final_value:,.2f}',
                      xaxis_rangeslider_visible=False)

    fig.write_html(outpath)

    return {'final_value': final_value, 'trades': {'buys': len(buy_dates), 'sells': len(sell_dates)}, 'outpath': outpath}


if __name__ == '__main__':
    try:
        res = run_backtest()
        print(f"Backtest complete. Output saved to: {res['outpath']}")
        print(f"Final portfolio value: ${res['final_value']:,.2f}")
    except Exception as e:
        print(f"Backtest failed: {e}")
        raise
