import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utilities import fetch_binance_klines

# Data Conversion
def convert_data(training_data, threshold=1.0):
    # Convert DataFrame to list of dictionaries if necessary
    if isinstance(training_data, pd.DataFrame):
        training_data = training_data.to_dict('records')

    # Check if training_data is a list of dictionaries
    if not isinstance(training_data, list) or not all(isinstance(data, dict) for data in training_data):
        raise ValueError("training_data should be a list of dictionaries with a 'close' key.")

    # Prepare X and y
    X = [data['close'] for data in training_data[:-1]]  # Extract 'close' values
    y = []

    for i in range(len(training_data) - 1):
        price_change = ((training_data[i + 1]['close'] - training_data[i]['close']) / training_data[i]['close']) * 100
        if price_change > threshold:
            y.append(1)  # Price increased
        elif price_change < -threshold:
            y.append(-1)  # Price decreased
        else:
            y.append(0)  # No significant change

    # One-hot encode y labels
    y_one_hot = [[1, 0, 0] if label == -1 else [0, 1, 0] if label == 0 else [0, 0, 1] for label in y]

    # Reshape X for LSTM [samples, timesteps, features]
    X = np.array(X).reshape(len(X), 1, 1)  # Adjusted to have one feature

    # Convert X and y to tensors
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y_one_hot, dtype=tf.float32)

    # Split data into training (80%) and validation (20%) sets
    split_index = int(0.8 * len(X))
    X_train, y_train = X_tensor[:split_index], y_tensor[:split_index]
    X_val, y_val = X_tensor[split_index:], y_tensor[split_index:]

    return X_train, y_train, X_val, y_val

# Custom metrics calculation functions
def calculate_precision(y_true, y_pred):
    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives > 0 else 0

def calculate_recall(y_true, y_pred):
    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives > 0 else 0

def calculate_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Train the model
def train_model_supervised(training_data):
    X_train, y_train, X_val, y_val = convert_data(training_data)

    # Convert y_train from one-hot encoding to class labels
    y_train_labels = np.argmax(y_train.numpy(), axis=1)

    # Build LSTM model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=25, input_shape=(1, X_train.shape[2])))
    model.add(tf.keras.layers.Dense(units=3, activation='softmax'))

    # Calculate class weights
    from sklearn.utils.class_weight import compute_class_weight

    # Compute class weights using the class labels
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)

    # Convert to dictionary format for Keras
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Custom callback to calculate precision, recall, and F1-score after each epoch
    class MetricsCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            y_pred = model.predict(X_val)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_val, axis=1)

            precision = calculate_precision(y_true_classes, y_pred_classes)
            recall = calculate_recall(y_true_classes, y_pred_classes)
            f1 = calculate_f1(precision, recall)

            print(f"Epoch {epoch + 1}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            print(f"Predicted: {y_pred_classes}, True: {y_true_classes}")

    # Train model with class weights
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), callbacks=[MetricsCallback()], class_weight=class_weight_dict)

    return model

# Example training data (you'll need to replace this with actual stock price data)
# example_data = [{'close': 100}, {'close': 102}, {'close': 101}, {'close': 105}, {'close': 104}]

example_data = fetch_binance_klines('BTCUSDT', '1d', 100)
print(example_data.head())

# Train the model
trained_model = train_model_supervised(example_data)

# Assume unseen_data is your new data to test
unseen_data = fetch_binance_klines('ETHUSDT', '1d', 20)  # Fetch new data

# Convert unseen data
X_unseen, y_unseen, _, _ = convert_data(unseen_data)

# Make predictions
y_pred_unseen = trained_model.predict(X_unseen)
y_pred_classes_unseen = np.argmax(y_pred_unseen, axis=1)
y_true_classes_unseen = np.argmax(y_unseen, axis=1)

# Evaluate predictions
precision_unseen = calculate_precision(y_true_classes_unseen, y_pred_classes_unseen)
recall_unseen = calculate_recall(y_true_classes_unseen, y_pred_classes_unseen)
f1_unseen = calculate_f1(precision_unseen, recall_unseen)

print(f"Unseen Data - Precision: {precision_unseen:.4f}, Recall: {recall_unseen:.4f}, F1: {f1_unseen:.4f}")
print(f"Predicted: {y_pred_classes_unseen}, True: {y_true_classes_unseen}")

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true_classes_unseen, y_pred_classes_unseen)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
