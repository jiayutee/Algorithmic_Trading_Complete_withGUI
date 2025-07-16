
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from finrl.config import INDICATORS

class FinRLStrategy:
    def __init__(self, model_path="trained_models/agent_ppo.zip"):
        self.model = PPO.load(model_path)

    def predict(self, data):
        """
        Predict actions for the given data.
        """
        df = data.copy()
        # Ensure the data has the required columns
        for indicator in INDICATORS:
            if indicator not in df.columns:
                # You might want to calculate the indicator if it's missing
                # For now, I'll just add a column of zeros as a placeholder
                df[indicator] = 0

        # Create the observation
        observation = self._create_observation(df)

        # Get the model's prediction
        action, _ = self.model.predict(observation)

        # Convert the action to signals
        signals = self._action_to_signals(action, df.index)

        return signals

    def _create_observation(self, data):
        """
        Create the observation required by the model.
        """
        # This is a simplified observation space based on the training script.
        # You may need to adjust this based on your specific training environment.
        # The state space is: balance + (shares_owned + closing_price) * num_stocks + indicators * num_stocks

        # For simplicity, I'm assuming a single stock and placeholder values for balance and shares.
        balance = np.array([1000000])  # Initial balance
        shares = np.array([0])  # Initial shares

        # Extract the relevant data
        closing_prices = data['Close'].values
        indicators = data[INDICATORS].values

        # Create the observation for each time step
        observations = []
        for i in range(len(data)):
            obs = np.hstack(
                (
                    balance,
                    shares,
                    closing_prices[i],
                    indicators[i],
                )
            )
            observations.append(obs)

        return np.array(observations)

    def _action_to_signals(self, action, index):
        """
        Convert the model's action to buy/sell/hold signals.
        """
        # The action space is continuous, representing the number of shares to trade.
        # Positive values are buys, negative values are sells.
        # We need to convert this to discrete signals (0: hold, 1: buy, 2: sell).

        signals = []
        for a in action:
            if a > 0:
                signals.append(1)  # Buy
            elif a < 0:
                signals.append(2)  # Sell
            else:
                signals.append(0)  # Hold

        return pd.Series(signals, index=index)
