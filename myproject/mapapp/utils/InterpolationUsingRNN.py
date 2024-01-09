import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time
from math import sqrt


class InterpolationUsingRNN:
    def __init__(self, results):
        self.df = pd.DataFrame([{
            'latitude': r.latitude,
            'longitude': r.longitude,
            'signalStrength': r.signalStrength
        } for r in results])
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def _prepare_data(self):
        features = self.df[['latitude', 'longitude']].values
        labels = self.df['signalStrength'].values

        # Normalize features
        features_scaled = self.scaler.fit_transform(features)

        # Reshape for LSTM [samples, time steps, features]
        # Assuming each sample is independent, time steps = 1
        X = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))
        y = labels
        return X, y

    def train_model(self, test_size=0.2, random_state=42, epochs=10, batch_size=32):
        X, y = self._prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # LSTM model
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=(1, 2)))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # Training the model
        start_train_time = time.time()
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
        training_duration = time.time() - start_train_time

        # Prediction and performance metrics
        y_pred = self.model.predict(X_test)
        mbe = np.mean(y_pred - y_test)

        metrics = {
            'Total Data Points': len(self.df),
            'Training Duration (seconds)': training_duration,
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': sqrt(mean_squared_error(y_test, y_pred)),
            'MBE': mbe,
            'R2': r2_score(y_test, y_pred),
            'Standard Deviation of Errors': np.std(y_pred - y_test)
        }

        return metrics

    # def predict(self, coordinates_list):
    #     coordinates_scaled = self.scaler.transform(coordinates_list)
    #     coordinates_reshaped = np.reshape(coordinates_scaled,
    #                                       (coordinates_scaled.shape[0], 1, coordinates_scaled.shape[1]))
    #     predictions = self.model.predict(coordinates_reshaped)
    #     return predictions.flatten()

    def predict(self, coordinates_list):
        predictions = []
        for lat, lon in coordinates_list:
            coordinates_scaled = self.scaler.transform([[lat, lon]])
            coordinates_reshaped = np.reshape(coordinates_scaled, (1, 1, -1))
            predicted_strength = self.model.predict(coordinates_reshaped)
            predictions.append((lat, lon, float(predicted_strength.flatten()[0])))
        print(predictions)
        return predictions
