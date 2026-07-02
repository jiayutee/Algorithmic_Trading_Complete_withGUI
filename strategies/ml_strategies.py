import backtrader as bt
from core.ta_engine import TAEngine
try:
    from sklearn.preprocessing import MinMaxScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    MinMaxScaler = None
    _SKLEARN_AVAILABLE = False
try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    joblib = None
    _JOBLIB_AVAILABLE = False
import numpy as np
import os
from core.logger import get_logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = get_logger(__name__)

class LSTMPredictor(bt.Strategy):
    params = (('ticker', 'AAPL'), ('sequence_length', 60),)

    def __init__(self):
        self.model = None
        self.scaler = None
        self.data_buffer = np.array([])
        self.load_trained_model()

    def load_trained_model(self):
        model_path = os.path.join(BASE_DIR, 'trained_models', f'lstm_model_{self.p.ticker}.h5')
        scaler_path = os.path.join(BASE_DIR, 'trained_models', f'scaler_{self.p.ticker}.pkl')

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                from tensorflow.keras.models import load_model  # lazy import — TF is optional
            except ImportError:
                logger.warning("TensorFlow not installed; LSTM strategy will be disabled.")
                self.model = None
                self.scaler = None
                return
            if not _SKLEARN_AVAILABLE or not _JOBLIB_AVAILABLE:
                logger.warning("sklearn/joblib not installed; LSTM strategy will be disabled.")
                self.model = None
                self.scaler = None
                return
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("LSTM model and scaler for %s loaded successfully.", self.p.ticker)
        else:
            logger.warning("Model or scaler for %s not found at expected path.", self.p.ticker)
            self.model = None
            self.scaler = None

    def predict(self, data_sequence):
        if self.model is None or self.scaler is None:
            logger.warning("Model not loaded, skipping prediction.")
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
        if self.model is None:
            return
        self.data_buffer = np.append(self.data_buffer, self.data.close[0])
        if len(self.data_buffer) < self.p.sequence_length:
            return
        self.data_buffer = self.data_buffer[-self.p.sequence_length:]
        predicted = self.predict(self.data_buffer)
        if predicted is None:
            return
        if predicted > self.data.close[0] and not self.position:
            self.buy()
        elif predicted < self.data.close[0] and self.position:
            self.sell()
