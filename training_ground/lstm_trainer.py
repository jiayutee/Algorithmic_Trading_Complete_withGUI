import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_lstm_model(ticker='AAPL', sequence_length=60, epochs=50, batch_size=32):
    # Load data
    df = pd.read_csv('/Users/jiayutee/Dev/Projects/Algorithmic_Trading_Complete_withGUI/training_ground/train_data.csv', parse_dates=['date'])
    df = df[df['tic'] == ticker].copy()
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # Select features (using 'close' price for simplicity, can expand to other indicators)
    features = ['close']
    data = df[features].values

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length)

    # Reshape X for LSTM [samples, time_steps, features]
    X = X.reshape(X.shape[0], X.shape[1], len(features))

    # Build LSTM model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(sequence_length, len(features))),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1) # Predicting one value (next close price)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Split data for training and validation
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Train model
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        verbose=1)

    # Save model and scaler
    model_dir = '/Users/jiayutee/Dev/Projects/Algorithmic_Trading_Complete_withGUI/trained_models'
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, f'lstm_model_{ticker}.h5'))
    joblib.dump(scaler, os.path.join(model_dir, f'scaler_{ticker}.pkl'))

    print(f"Model and scaler for {ticker} saved successfully.")

if __name__ == '__main__':
    # Example usage:
    train_lstm_model(ticker='AAPL', sequence_length=60, epochs=50, batch_size=32)
    # You can add more tickers here if needed
    # train_lstm_model(ticker='MSFT', sequence_length=60, epochs=50, batch_size=32)
