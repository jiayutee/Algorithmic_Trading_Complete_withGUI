import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import time
from collections import deque, namedtuple

# Gym & RL
import gym
from gym import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

# Metrics
from scipy.stats import linregress
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from config.settings import KUCOIN_API_KEY, KUCOIN_SECRET_KEY


def fetch_kucoin_ohlcv(kucoin_key, kucoin_secret, symbol="BTC/USDT", timeframe="1m", since=None, limit=1000, max_iters=50):
    """
    Fetch OHLCV in batches of up to `limit` (max 1000 for KuCoin).
    since: unix ms timestamp to start from, or None to fetch most recent.
    Returns combined DataFrame with timestamp index (UTC).
    """
    exchange = ccxt.kucoin({
        'apiKey': kucoin_key,
        'secret': kucoin_secret,
        'enableRateLimit': True,
    })
    all_ohlcv = []
    params = {}
    if since is None:
        # fetch the most recent `limit` candles as a single batch
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    else:
        ts = int(since)
        for i in range(max_iters):
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=ts, limit=limit, params=params)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            ts = ohlcv[-1][0] + 1
            # avoid rate limits
            time.sleep(0.2)
            if len(ohlcv) < limit:
                break
        df = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        # drop duplicates and sort
        df = df[~df.index.duplicated(keep='first')].sort_index()
        return df


def fit_obs_scaler(df, window=50):
    # build sample observations for scaler fitting (prices normalized by last close)
    rows = []
    for i in range(window, len(df), max(1, len(df)//2000)):  # sample up to ~2000 windows
        start = i-window
        block = df.iloc[start:i][['open','high','low','close','volume','return']].copy()
        # normalize relative to last close
        block[['open','high','low','close']] = block[['open','high','low','close']].div(block['close'].iloc[-1])
        vol = block['volume'] / (block['volume'].mean() + 1e-9)
        rows.append(np.hstack([block[['open','high','low','close']].values.flatten(),
                               vol.values.flatten(), block['return'].values.flatten()]))
    scaler = StandardScaler()
    scaler.fit(np.vstack(rows))
    return scaler


class ContinuousPositionEnv(gym.Env):
    """
    Continuous trading environment for DDPG, using a window-normalized observation scaler.
    Tracks per-trade PnL, cumulative PnL, and total asset value.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size=50, initial_cash=10000.0, commission=0.0005,
                 allow_short=True, obs_scaler=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window = window_size
        self.initial_cash = initial_cash
        self.commission = commission
        self.allow_short = allow_short
        self.done = False

        # pre-fitted scaler
        self.scaler = obs_scaler

        self.n_features = 6  # open, high, low, close, volume, return
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window * self.n_features,),
            dtype=np.float32
        )

        self.reset()

    def _make_obs(self):
        """Build normalized observation using fitted scaler (window-based)."""
        start = max(0, self.current_step - self.window + 1)
        block = self.df.iloc[start:self.current_step+1][['open','high','low','close','volume','return']].copy()

        if len(block) < self.window:
            pad = pd.DataFrame([block.iloc[0].values] * (self.window - len(block)), columns=block.columns)
            block = pd.concat([pad, block], ignore_index=True)

        # normalize relative to last close
        block[['open','high','low','close']] = block[['open','high','low','close']].div(block['close'].iloc[-1])
        block['volume'] = block['volume'] / (block['volume'].mean() + 1e-9)
        block['return'] = block['return'].fillna(0)

        obs = np.hstack([
            block[['open','high','low','close']].values.flatten(),
            block['volume'].values.flatten(),
            block['return'].values.flatten()
        ])

        if self.scaler is not None:
            obs = self.scaler.transform(obs.reshape(1, -1))[0]

        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window
        self.cash = self.initial_cash
        self.position = 0.0
        self.prev_action = 0.0
        self.total_asset = self.initial_cash
        self.done = False

        # trade tracking
        self.open_trade = None
        self.trades = []  # list of dicts {'entry_price','exit_price','size','side','pnl'}
        self.per_trade_pnl = []
        self.cum_pnl = []
        self.asset_history = [self.initial_cash]
        self.position_history = []

        self.smoothed_action = 0.0
        return self._make_obs(), {}

    def step(self, action):
        action = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        smoothed_action = 0.2 * action + 0.8 * self.prev_action
        self.prev_action = smoothed_action

        if not self.allow_short:
            smoothed_action = np.clip(smoothed_action, 0.0, 1.0)

        price = float(self.df.loc[self.current_step, 'close'])
        next_price = float(self.df.loc[self.current_step + 1, 'close']) if self.current_step < len(self.df)-1 else price

        total_asset = self.cash + self.position * price
        desired_position_value = smoothed_action * total_asset
        desired_units = desired_position_value / price
        delta_units = desired_units - self.position

        # trade execution
        trade_value = abs(delta_units) * price
        fee = trade_value * self.commission
        self.cash -= delta_units * price + fee
        self.position = desired_units
        self.total_asset = self.cash + self.position * price

        if abs(delta_units) > 1e-12:
            side = 'long' if delta_units > 0 else 'short'
            if self.open_trade is None:
                self.open_trade = {'entry_price': price, 'size': abs(delta_units), 'side': side}
            else:
                prev_side = self.open_trade['side']
                if side != prev_side:
                    entry = self.open_trade['entry_price']
                    size = self.open_trade['size']
                    pnl = (price - entry) * size if prev_side == 'long' else (entry - price) * size
                    pnl -= fee
                    self.per_trade_pnl.append(pnl)
                    self.trades.append({**self.open_trade, 'exit_price': price, 'pnl': pnl})
                    self.open_trade = {'entry_price': price, 'size': abs(delta_units), 'side': side}

        new_total_asset = self.cash + self.position * next_price
        reward_raw = new_total_asset - self.total_asset
        position_penalty = -0.001 * abs(delta_units)
        reward = (reward_raw / (self.total_asset + 1e-9)) + position_penalty
        reward = np.clip(reward, -1, 1)

        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            self.done = True
            close_price = next_price
            if abs(self.position) > 1e-12:
                if self.open_trade is not None:
                    entry = self.open_trade['entry_price']
                    size = self.open_trade['size']
                    side = self.open_trade['side']
                    pnl = (close_price - entry) * size if side == 'long' else (entry - close_price) * size
                    pnl -= fee
                    self.per_trade_pnl.append(pnl)
                    self.trades.append({**self.open_trade, 'exit_price': close_price, 'pnl': pnl})
                    self.open_trade = None

                self.cash += self.position * close_price
                self.position = 0.0
                self.total_asset = self.cash

        self.asset_history.append(self.total_asset)
        self.position_history.append(self.position)
        self.cum_pnl.append(self.total_asset - self.initial_cash)

        obs = self._make_obs() if not self.done else np.zeros_like(self._make_obs())
        info = {
            'step': self.current_step,
            'price': price,
            'total_asset': self.total_asset,
            'cash': self.cash,
            'position': self.position,
            'reward_raw': reward_raw,
            'smoothed_action': smoothed_action
        }

        return obs, reward, self.done, False, info

    def render(self, mode="human"):
        print(f"Step {self.current_step} | Asset={self.total_asset:.2f} | Position={self.position:.4f}")


def compute_performance_metrics(asset_values, price_series, initial_cash=10000.0):
    ts_asset = pd.Series(asset_values, index=price_series.index[:len(asset_values)])
    strat_ret = ts_asset.pct_change().fillna(0)
    price = price_series[:len(asset_values)]
    btc_units = initial_cash / price.iloc[0]
    benchmark_value = btc_units * price
    bench_ret = benchmark_value.pct_change().fillna(0)

    strat_daily = (1 + strat_ret).resample('D').prod() - 1
    bench_daily = (1 + bench_ret).resample('D').prod() - 1

    dfm = pd.concat([strat_daily, bench_daily], axis=1).dropna()
    print("Daily returns dataframe (dfm):")
    print(dfm)
    dfm.columns = ['strategy','benchmark']
    if dfm['strategy'].std() > 0:
        sharpe = dfm['strategy'].mean() / dfm['strategy'].std() * np.sqrt(252)
    else:
        sharpe = np.nan

    if dfm.empty:
        print("Daily returns dataframe is empty, cannot compute OLS regression.")
        alpha = np.nan
        beta = np.nan
    elif len(dfm) < 2:
        print("Not enough data points in daily returns dataframe for OLS regression (need at least 2).")
        alpha = np.nan    
        beta = np.nan
    elif dfm['benchmark'].std() == 0:
        print("Benchmark returns have no variance, cannot compute OLS regression.")
        alpha = np.nan
        beta = np.nan
    else:
        X = sm.add_constant(dfm['benchmark'])
        y = dfm['strategy']
        try:
            res = sm.OLS(y, X).fit()
            alpha = res.params['const']
            beta = res.params['benchmark']
        except Exception as e:
            print(f"Error in OLS regression: {e}")
            alpha = np.nan
            beta = np.nan

    return {
        'sharpe': sharpe,
        'alpha': alpha,
        'beta': beta,
        'strategy_daily_mean': dfm['strategy'].mean(),
        'benchmark_daily_mean': dfm['benchmark'].mean(),
        'df_daily': dfm
    }


def main():
    SYMBOL = "BTC/USDT"
    TIMEFRAME = "5m"
    
    # Fetch data
    since = dt.datetime.now() - dt.timedelta(days=8)
    since_ms = int(since.timestamp() * 1000)
    df = fetch_kucoin_ohlcv(KUCOIN_API_KEY, KUCOIN_SECRET_KEY, symbol=SYMBOL, timeframe=TIMEFRAME, since=since_ms, max_iters=100)
    df['return'] = df['close'].pct_change().fillna(0)

    # Split data
    train_size = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

    # Create gym env for training & wrap
    train_env = ContinuousPositionEnv(train_df, window_size=50, initial_cash=10000.0, commission=0.0005)
    vec_train_env = DummyVecEnv([lambda: train_env])

    # Train TD3 agent
    n_actions = vec_train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))

    model = TD3("MlpPolicy", vec_train_env, verbose=1, buffer_size=200000, learning_starts=5000,
                batch_size=256, action_noise=action_noise, learning_rate=1e-4, device='mps')

    train_timesteps = train_size*5
    model.learn(total_timesteps=train_timesteps, log_interval=10)

    # Evaluate on test data
    eval_env = ContinuousPositionEnv(test_df, window_size=50, initial_cash=10000.0, commission=0.0005)
    obs, _ = eval_env.reset()

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)

    asset_values = np.array(eval_env.asset_history)
    cum_pnl = np.array(eval_env.cum_pnl)
    per_trade_pnl = np.array(eval_env.per_trade_pnl) if len(eval_env.per_trade_pnl) > 0 else np.array([])

    print("Final total asset:", asset_values[-1])
    print("Number of trades (closed):", len(per_trade_pnl))

    # Compute performance metrics
    price_series = test_df['close']
    metrics = compute_performance_metrics(asset_values, price_series, initial_cash=10000.0)
    
    print("Sharpe (annualized):", metrics['sharpe'])
    print("Alpha (daily):", metrics['alpha'])
    print("Beta:", metrics['beta'])

    if len(per_trade_pnl) > 0:
        print("Number of closed trades:", len(per_trade_pnl))
        print("Avg profit per trade:", per_trade_pnl.mean())
        print("Median profit per trade:", np.median(per_trade_pnl))
        print("Win rate:", (per_trade_pnl > 0).mean())
    else:
        print("No closed trades recorded in test period.")

    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

    axs[0].plot(test_df.index[:len(cum_pnl)], cum_pnl, label='Cumulative PnL', color='tab:blue')
    axs[0].axhline(0, color='k', linewidth=0.5)
    axs[0].set_title('Cumulative PnL (Test)')
    axs[0].legend()

    axs[1].plot(test_df.index[:len(asset_values)], asset_values, label='Total Asset', color='tab:orange')
    axs[1].axhline(10000.0, color='k', linestyle='--', linewidth=0.5)
    axs[1].set_title('Total Asset Value (Test)')
    axs[1].legend()

    if len(per_trade_pnl) > 0:
        axs[2].bar(range(len(per_trade_pnl)), per_trade_pnl, color=['g' if p>=0 else 'r' for p in per_trade_pnl])
        axs[2].set_title('Profit per Trade (Test)')
        axs[2].set_xlabel('Trade #')
    else:
        axs[2].text(0.5, 0.5, 'No closed trades during test', ha='center', va='center')
        axs[2].set_title('Profit per Trade (Test)')

    plt.tight_layout()
    plt.show()

    # Save key outputs
    pd.Series(asset_values, index=test_df.index[:len(asset_values)]).to_csv("ddpg_asset_values_test.csv")
    if len(per_trade_pnl) > 0:
        pd.Series(per_trade_pnl).to_csv("ddpg_per_trade_pnl_test.csv")
    print("Saved outputs to CSV.")


if __name__ == "__main__":
    main()
