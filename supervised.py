import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from utilities import fetch_binance_klines

def label_data(data, threshold=0.001):
    labels = []
    for i in range(1, len(data)):
        change = (data[i] - data[i-1]) / data[i-1]
        if change > 0:
            labels.append(1)  # Up
        elif change < 0:
            labels.append(0)  # Down
        # Removed the 'No significant change' case
    return np.array(labels)

def prepare_data(symbol, interval, limit, look_back=60, threshold=0.01):
    # Fetch data
    df = fetch_binance_klines(symbol, interval, limit)
    
    # Use all columns except 'timestamp' for prediction
    data = df.values
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Prepare the dataset for LSTM
    X, y = [], []
    labels = label_data(df['close'].values, threshold)
    for i in range(look_back, len(labels)):  # Adjusted loop to len(labels)
        X.append(scaled_data[i-look_back:i])
        y.append(labels[i])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
    
    return X, to_categorical(y, num_classes=2), scaler  # Specify num_classes=2

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=2, activation='softmax'))  # 2 classes: up, down
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(train_symbol, test_symbol, interval='1h', limit=1000, look_back=60, epochs=50, batch_size=32):
    # Prepare training data
    X_train, y_train, _ = prepare_data(train_symbol, interval, limit, look_back)
    
    # Prepare testing data
    X_test, y_test, _ = prepare_data(test_symbol, interval, limit, look_back)
    
    # Build model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # Train model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    print(classification_report(y_true_classes, y_pred_classes, target_names=['Down', 'Up']))  # Removed 'No Change'

# Example usage
train_symbol = 'BTCUSDT'
test_symbol = 'ETHUSDT'
train_and_evaluate(train_symbol, test_symbol)
