import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scipy.spatial import distance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from math import sqrt
import numpy as np
import time

# class InterpolationUsingGAN:
#     def __init__(self, results, min_samples_per_pci=6):
#         # Convert results to a DataFrame
#         self.df = pd.DataFrame([{
#             'latitude': r.latitude,
#             'longitude': r.longitude,
#             'PCI': r.cellId_PCI,
#             'signalStrength': r.signalStrength
#         } for r in results])
#         self.generator = None
#         self.zero_points = self._calculate_zero_points()
#
#     def _calculate_zero_points(self):
#         zero_points = {}
#         for pci, group in self.df.groupby('PCI'):
#             max_signal_idx = group['signalStrength'].idxmax()
#             zero_points[pci] = group.loc[max_signal_idx, ['latitude', 'longitude']].values
#         return zero_points
#
#     def _calculate_distance_to_zero_point(self, row):
#         pci = row['PCI']
#         zero_point = self.zero_points[pci]
#         return distance.euclidean((row['latitude'], row['longitude']), zero_point)
#
#     def _build_generator(self):
#         model = Sequential()
#         model.add(Dense(128, activation='relu', input_dim=3))
#         model.add(Dense(64, activation='relu'))
#         model.add(Dense(1, activation='linear'))
#         return model
#
#     def train_model(self):
#         if len(self.df) < 5:
#             print("Not enough data to train the model.")
#             return
#
#         self.df['distance_to_zero'] = self.df.apply(self._calculate_distance_to_zero_point, axis=1)
#
#         X = self.df[['latitude', 'longitude', 'distance_to_zero']]
#         y = self.df['signalStrength']
#
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
#         self.generator = self._build_generator()
#         self.generator.compile(optimizer='adam', loss='mean_squared_error')
#
#         start_train_time = time.time()
#         self.generator.fit(X_train, y_train, epochs=100, batch_size=32)
#         training_duration = time.time() - start_train_time
#         print('Training End')
#         # Prediction and performance metrics
#         y_pred = self.generator.predict(X_test).flatten()  # Ensuring y_pred is 1D
#
#         print(f"Shape of y_pred: {y_pred.shape}, Shape of y_test: {y_test.shape}")  # Debug print
#
#         mbe = (y_pred - y_test).mean()
#
#         metrics = {
#             'Total Data Points': len(self.df),
#             'Training Duration (seconds)': training_duration,
#             'MAE': mean_absolute_error(y_test, y_pred),
#             'RMSE': sqrt(mean_squared_error(y_test, y_pred)),
#             'MBE': mbe,
#             'R2': r2_score(y_test, y_pred),
#             'Standard Deviation of Errors': np.std(y_pred - y_test)
#         }
#
#         print('Training End')
#         return metrics
#
#     def predict(self, coordinates_list):
#         predictions = []
#         for lat, lon in coordinates_list:
#             closest_pci, closest_zero_point = min(self.zero_points.items(),
#                                                   key=lambda item: distance.euclidean((lat, lon), item[1]))
#             dist_to_zero = distance.euclidean((lat, lon), closest_zero_point)
#             predicted_signal_strength = self.generator.predict([[lat, lon, dist_to_zero]])[0][0]
#             # Convert float32 to native Python float
#             predicted_signal_strength = float(predicted_signal_strength)
#             predictions.append((lat, lon, predicted_signal_strength))
#
#         print(predictions)
#         return predictions


class InterpolationUsingGAN:
    def __init__(self, results, min_samples_per_pci=6):
        self.df = pd.DataFrame([{
            'latitude': r.latitude,
            'longitude': r.longitude,
            'signalStrength': r.signalStrength
        } for r in results])
        self.generator = None

    def _build_generator(self):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_dim=2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='linear'))
        return model

    def _cross_validated_rmse(self, X, y, n_splits=5):
        kf = KFold(n_splits=n_splits)
        rmse_scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = self._build_generator()
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

            y_pred = model.predict(X_test).flatten()
            rmse_score = sqrt(mean_squared_error(y_test, y_pred))
            rmse_scores.append(rmse_score)

        return np.mean(rmse_scores)

    def train_model(self):
        if len(self.df) < 5:
            print("Not enough data to train the model.")
            return

        X = self.df[['latitude', 'longitude']]
        y = self.df['signalStrength']

        # Calculate Cross-Validated RMSE
        cross_validated_rmse = self._cross_validated_rmse(X, y)

        # Train-test split for final model training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.generator = self._build_generator()
        self.generator.compile(optimizer='adam', loss='mean_squared_error')

        start_train_time = time.time()
        self.generator.fit(X_train, y_train, epochs=100, batch_size=32)
        training_duration = time.time() - start_train_time

        y_pred = self.generator.predict(X_test).flatten()

        mbe = (y_pred - y_test).mean()

        metrics = {
            'Total Data Points': len(self.df),
            'Training Duration (seconds)': training_duration,
            'Cross-validated RMSE': cross_validated_rmse,
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': sqrt(mean_squared_error(y_test, y_pred)),
            'MBE': mbe,
            'R2': r2_score(y_test, y_pred),
            'Standard Deviation of Errors': np.std(y_pred - y_test)
        }

        return metrics

    def predict(self, coordinates_list):
        predictions = []
        start_prediction_time = time.time()
        for lat, lon in coordinates_list:
            predicted_signal_strength = self.generator.predict([[lat, lon]])[0][0]
            predicted_signal_strength = float(predicted_signal_strength)
            predictions.append((lat, lon, predicted_signal_strength))

        end_prediction_time = time.time()
        prediction_time = end_prediction_time - start_prediction_time
        return predictions, prediction_time
