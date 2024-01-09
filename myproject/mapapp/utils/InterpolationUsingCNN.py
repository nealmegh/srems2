import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import time
from sklearn.model_selection import cross_val_score, KFold
from numpy import sqrt, mean, std

class InterpolationUsingCNN:
    def __init__(self, results):
        self.df = pd.DataFrame([{
            'latitude': r.latitude,
            'longitude': r.longitude,
            'signalStrength': r.signalStrength
        } for r in results])
        self.model = None
        self.grid_size = (100, 100)

    def _create_individual_grid(self, latitude, longitude, signalStrength):
        lat_range = (self.df['latitude'].min(), self.df['latitude'].max())
        lon_range = (self.df['longitude'].min(), self.df['longitude'].max())
        lat_idx = int((latitude - lat_range[0]) / (lat_range[1] - lat_range[0]) * (self.grid_size[0] - 1))
        lon_idx = int((longitude - lon_range[0]) / (lon_range[1] - lon_range[0]) * (self.grid_size[1] - 1))

        # Clamp the indices to ensure they are within grid boundaries
        lat_idx = max(0, min(lat_idx, self.grid_size[0] - 1))
        lon_idx = max(0, min(lon_idx, self.grid_size[1] - 1))

        grid = np.zeros(self.grid_size)
        grid[lat_idx, lon_idx] = signalStrength
        return grid

    def _prepare_dataset(self):
        X = np.array([self._create_individual_grid(row['latitude'], row['longitude'], row['signalStrength']) for index, row in self.df.iterrows()])
        X = np.expand_dims(X, axis=-1)  # Adding the channel dimension
        return X

    def _create_model(self, input_shape):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1))  # Output layer for regression
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    def train_model(self, test_size=0.2, random_state=42, n_splits=5):
        X = self._prepare_dataset()
        y = self.df['signalStrength'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.model = self._create_model(X_train.shape[1:])
        start_train_time = time.time()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_scores = []
        for train_index, val_index in kf.split(X_train):
            X_train_kf, X_val_kf = X_train[train_index], X_train[val_index]
            y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]

            self.model.fit(X_train_kf, y_train_kf, batch_size=32, epochs=10, validation_data=(X_val_kf, y_val_kf))
            y_val_pred = self.model.predict(X_val_kf)
            cv_scores.append(sqrt(mean_squared_error(y_val_kf, y_val_pred)))

        rmse_cv = mean(cv_scores)

        # Final Training

        self.model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.1)
        training_duration = time.time() - start_train_time

        # Prediction and performance metrics
        y_pred = self.model.predict(X_test)
        mbe = mean(y_pred - y_test)

        metrics = {
            'Total Data Points': len(self.df),
            'Training Duration (seconds)': training_duration,
            'Cross-validated RMSE': rmse_cv,
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': sqrt(mean_squared_error(y_test, y_pred)),
            'MBE': mbe,
            'R2': r2_score(y_test, y_pred),
            'Standard Deviation of Errors': std(y_pred - y_test)
        }

        return metrics

    def predict(self, coordinates_list):
        predictions = []
        for lat, lon in coordinates_list:
            grid_input = self._create_individual_grid(lat, lon, 0)
            print("Grid input shape:", grid_input.shape)  # Debugging shape
            grid_input_reshaped = grid_input.reshape((1, *grid_input.shape, 1))
            print("Reshaped input shape:", grid_input_reshaped.shape)  # Debugging reshaped input shape
            predicted_strength = self.model.predict(grid_input_reshaped)
            predictions.append((lat, lon, float(predicted_strength.flatten()[0])))
        print(predictions)
        return predictions

