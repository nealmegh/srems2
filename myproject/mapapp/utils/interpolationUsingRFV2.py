from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial import distance
import numpy as np
import pandas as pd
import time
from math import sqrt
from sklearn.model_selection import cross_val_score

class InterpolationUsingRFV2:
    def __init__(self, results, data_source, min_samples_per_pci=6):
        if data_source != 'csv':
            self.df = pd.DataFrame([{
                'latitude': r.latitude,
                'longitude': r.longitude,
                'PCI': r.cellId_PCI,
                'signalStrength': r.signalStrength
            } for r in results])
        else:
            self.df = pd.DataFrame([{
                'latitude': float(r['latitude']),
                'longitude': float(r['longitude']),
                'PCI': int(r['cellId_PCI']),
                'signalStrength': int(r['signalStrength'])
            } for r in results])
        self.model = None
        self.zero_points = self._calculate_zero_points()

    def _calculate_zero_points(self):
        zero_points = {}
        for pci, group in self.df.groupby('PCI'):
            max_signal_idx = group['signalStrength'].idxmax()
            zero_points[pci] = group.loc[max_signal_idx, ['latitude', 'longitude']].values
        return zero_points

    def _calculate_distance_to_zero_point(self, row):
        pci = row['PCI']
        zero_point = self.zero_points[pci]
        return distance.euclidean((row['latitude'], row['longitude']), zero_point)

    def train_model(self):
        if len(self.df) < 5:
            print("Not enough data to train the model.")
            return

        self.df['distance_to_zero'] = self.df.apply(self._calculate_distance_to_zero_point, axis=1)

        X = self.df[['latitude', 'longitude', 'distance_to_zero']]
        y = self.df['signalStrength']

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Cross-validation
        num_folds = min(5, len(X_train))
        rmse_cv = None
        if num_folds > 1:
            scores = cross_val_score(self.model, X_train, y_train, cv=num_folds, scoring='neg_mean_squared_error')
            rmse_cv = sqrt(-scores.mean())

        # Training
        start_train_time = time.time()
        self.model.fit(X_train, y_train)
        training_duration = time.time() - start_train_time

        # Prediction and performance metrics
        y_pred = self.model.predict(X_test)
        mbe = (y_pred - y_test).mean()

        metrics = {
            'Total Data Points': len(self.df),
            'Training Duration (seconds)': training_duration,
            'Cross-validated RMSE': rmse_cv,
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
            closest_pci, closest_zero_point = min(self.zero_points.items(),
                                                  key=lambda item: distance.euclidean((lat, lon), item[1]))
            dist_to_zero = distance.euclidean((lat, lon), closest_zero_point)
            predicted_signal_strength = self.model.predict([[lat, lon, dist_to_zero]])[0]
            predictions.append((lat, lon, predicted_signal_strength))

        end_prediction_time = time.time()
        prediction_time = end_prediction_time - start_prediction_time
        return predictions, prediction_time
