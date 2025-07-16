import backtrader as bt
from core.ta_engine import TAEngine
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import os

class LSTMPredictor(bt.Strategy):
    params = (('ticker', 'AAPL'), ('sequence_length', 60),)

    def __init__(self):
        self.model = None
        self.scaler = None
        self.data_buffer = np.array([])
        self.load_trained_model()

    def load_trained_model(self):
        model_path = os.path.join('/Users/jiayutee/Dev/Projects/Algorithmic_Trading_Complete_withGUI/trained_models', f'lstm_model_{self.p.ticker}.h5')
        scaler_path = os.path.join('/Users/jiayutee/Dev/Projects/Algorithmic_Trading_Complete_withGUI/trained_models', f'scaler_{self.p.ticker}.pkl')

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"LSTM model and scaler for {self.p.ticker} loaded successfully.")
        else:
            print(f"Error: Model or scaler for {self.p.ticker} not found.")
            self.model = None
            self.scaler = None

    def predict(self, data_sequence):
        if self.model is None or self.scaler is None:
            print("Model not loaded. Cannot make predictions.")
            return None

        # Ensure data_sequence is a 2D array for scaling
        data_sequence_reshaped = np.array(data_sequence).reshape(-1, 1)
        scaled_data = self.scaler.transform(data_sequence_reshaped)

        # Reshape for LSTM input: [1, sequence_length, 1]
        X = scaled_data.reshape(1, self.p.sequence_length, 1)

        # Make prediction
        predicted_scaled_price = self.model.predict(X, verbose=0)[0][0]

        # Inverse transform to get actual price
        predicted_price = self.scaler.inverse_transform(np.array([[predicted_scaled_price]]))[0][0]
        return predicted_price

    def next(self):
        # This is a Backtrader strategy, but for prediction, we'll call predict directly.
        # In a real backtest, you would feed data to the predict method.
        pass


# Example of how you might integrate this into a strategy manager
# This part is illustrative and might need adjustment based on your actual strategy_manager
class StrategyManager:
    def __init__(self):
        self.strategies = {
            "LSTM Predictor": LSTMPredictor,
            # ... other strategies
        }

    def get_strategy(self, name):
        return self.strategies.get(name)

    def run_backtest(self, strategy, data):
        # This is a placeholder. Actual backtesting logic would go here.
        print(f"Running backtest with {strategy.__class__.__name__}")
        # Example of how to use the predictor within a backtest
        # if isinstance(strategy, LSTMPredictor):
        #     # Assuming 'data' is a pandas DataFrame with a 'close' column
        #     # and you want to predict the next day's close price
        #     # You'd need to manage the data buffer within the strategy's next() method
        #     # For a simple test:
        #     if len(data) > strategy.p.sequence_length:
        #         last_sequence = data['close'].tail(strategy.p.sequence_length).values
        #         prediction = strategy.predict(last_sequence)
        #         print(f"Predicted next close: {prediction}")
        return {'final_value': 100000, 'sharpe': 1.5, 'signals': []}

# You might need to adjust your main app.py to pass the correct strategy class
# For example:
# from strategies.ml_strategies import LSTMPredictor
# strategy_instance = LSTMPredictor()
# self.strategy_manager.run_backtest(strategy_instance, self.df)
