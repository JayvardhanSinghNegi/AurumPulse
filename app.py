from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Serve frontend from ./frontend folder
app = Flask(__name__, static_folder='frontend', static_url_path='')

# === GoldHeart LSTM Model Setup ===
gold_data = pd.read_csv('gp_final.csv')
gold_data['close'] = gold_data['close'].str.replace(',', '').astype(float)
price_values = gold_data['close'].values.reshape(-1, 1)

GoldHeartScaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = GoldHeartScaler.fit_transform(price_values)

gold_seq_len = 30
gold_target_len = 1
gold_sequences = []
gold_targets = []

for i in range(len(scaled_values) - gold_seq_len - gold_target_len + 1):
    seq = scaled_values[i:i + gold_seq_len]
    gold_sequences.append(seq)
    gold_targets.append(scaled_values[i + gold_seq_len + gold_target_len - 1])

gold_sequences = np.array(gold_sequences)
gold_targets = np.array(gold_targets)

split_ratio = 0.8
train_size = int(len(gold_sequences) * split_ratio)

X_train = gold_sequences[:train_size]
X_test = gold_sequences[train_size:]
y_train = gold_targets[:train_size]
y_test = gold_targets[train_size:]

# Autoencoder definition
ae_input = tf.keras.layers.Input(shape=(gold_seq_len, 1))
ae_encoded = tf.keras.layers.LSTM(64, activation='relu', return_sequences=True)(ae_input)
ae_decoded = tf.keras.layers.LSTM(1, activation='sigmoid', return_sequences=True)(ae_encoded)
GoldAutoencoder = tf.keras.Model(inputs=ae_input, outputs=ae_decoded)

# LSTM prediction model
lstm_input = tf.keras.layers.Input(shape=(gold_seq_len, 1))
lstm_encoded = tf.keras.layers.LSTM(32, activation='relu')(lstm_input)
lstm_output = tf.keras.layers.Dense(1)(lstm_encoded)
GoldForecaster = tf.keras.Model(inputs=lstm_input, outputs=lstm_output)

# Combined model
full_input = tf.keras.layers.Input(shape=(gold_seq_len, 1))
encoded_seq = GoldAutoencoder(full_input)
final_output = GoldForecaster(encoded_seq)
GoldHeartModel = tf.keras.Model(inputs=full_input, outputs=final_output)
GoldHeartModel.compile(optimizer='adam', loss='mean_squared_error')

X_train_seq = X_train.reshape(X_train.shape[0], gold_seq_len, 1)
X_test_seq = X_test.reshape(X_test.shape[0], gold_seq_len, 1)

# Train once
GoldHeartModel.fit(X_train_seq, y_train, epochs=50, batch_size=32, validation_data=(X_test_seq, y_test), verbose=0)

# === Serve index.html at root ===
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# === Serve other static frontend files ===
@app.route('/<path:path>')
def serve_static_file(path):
    return send_from_directory(app.static_folder, path)

# === Prediction Endpoint with Sentiment Logic ===
@app.route('/predict', methods=['POST'])
def predict_next_price():
    data = request.get_json()
    user_seq = data.get('sequence', [])
    user_type = data.get('user_type')  # <- New: Accept user type

    if len(user_seq) != gold_seq_len:
        return jsonify({'error': f'Sequence must be {gold_seq_len} values'}), 400

    if user_type not in ['buyer', 'seller']:
        return jsonify({'error': 'Invalid user type. Must be "buyer" or "seller".'}), 400

    try:
        user_seq_array = np.array(user_seq).reshape(1, gold_seq_len, 1)
        scaled_seq = GoldHeartScaler.transform(np.array(user_seq).reshape(-1, 1)).reshape(1, gold_seq_len, 1)
        pred_scaled = GoldHeartModel.predict(scaled_seq)
        pred_price = GoldHeartScaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

        # Get current price from last row
        current_price = gold_data['close'].iloc[-1]

        # Emotion-aware sentiment
        if user_type == 'buyer':
            sentiment = "Good time to buy!" if pred_price < current_price else "You might want to wait."
        else:  # seller
            sentiment = "Good time to sell!" if pred_price > current_price else "Consider holding off."

        return jsonify({
            'predicted_price': round(float(pred_price), 2),
            'current_price': round(float(current_price), 2),
            'sentiment': sentiment
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run Flask on all interfaces for Docker
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
