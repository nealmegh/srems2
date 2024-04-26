import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import time


class RegressionKrigingRFK:

    def __init__(self, results, data_source, variogram_model='linear', min_samples_per_pci=60):
        self.variogram_model = variogram_model
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
        self.OK = None
        self.zero_points = self.calculate_zero_points()
        self.df['distance_to_zero'] = self.calculate_distance_to_zero()

    def calculate_zero_points(self):
        zero_points = {}
        for pci, group in self.df.groupby('PCI'):
            max_signal_idx = group['signalStrength'].idxmax()
            zero_points[pci] = group.loc[max_signal_idx, ['latitude', 'longitude']].values
        return zero_points

    def calculate_distance_to_zero(self):
        return self.df.apply(lambda row: distance.euclidean((row['latitude'], row['longitude']),
                                                            self.zero_points[row['PCI']]), axis=1)

    def train_model(self):
        if len(self.df) < 5:
            print("Not enough data to train the model.")
            return

        X = self.df[['latitude', 'longitude', 'distance_to_zero']]
        y = self.df['signalStrength']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        start_train_time = time.time()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Kriging on the residuals
        regression_predictions = self.model.predict(X_train)
        residuals = y_train - regression_predictions
        self.OK = OrdinaryKriging(X_train['longitude'].values, X_train['latitude'].values, residuals,
                                  variogram_model=self.variogram_model, enable_plotting=False)

        training_duration = time.time() - start_train_time

        # Testing
        y_pred = self.model.predict(X_test)
        kriged_residuals, _ = self.OK.execute('points', X_test['longitude'].values, X_test['latitude'].values)
        final_predictions = y_pred + kriged_residuals

        metrics = {
            'Total Data Points':  len(self.df),
            'Training Duration (seconds)': training_duration,
            'MAE': mean_absolute_error(y_test, final_predictions),
            'RMSE': sqrt(mean_squared_error(y_test, final_predictions)),
            'MBE': np.mean(final_predictions - y_test.values),
            'R2': r2_score(y_test, final_predictions),
            'Standard Deviation of Errors': np.std(final_predictions.data - y_test.to_numpy())
        }
        print(metrics)
        return metrics

    def predict(self, coordinates_list):
        predictions = []
        start_prediction_time = time.time()

        for lat, lon in coordinates_list:
            dist_to_zero = distance.euclidean((lat, lon), self.zero_points[self.closest_pci((lat, lon))])
            # print('prediction', dist_to_zero)
            regression_prediction = self.model.predict([[lat, lon, dist_to_zero]])[0]
            kriged_residual, _ = self.OK.execute('points', [lon], [lat])
            predicted_signal_strength = regression_prediction + kriged_residual[0]
            predictions.append((lat, lon, predicted_signal_strength))

        end_prediction_time = time.time()
        prediction_time = end_prediction_time - start_prediction_time
        print("First few predictions:", predictions[:15])
        return predictions, prediction_time
        # return predictions

    def closest_pci(self, coordinates):
        # Find the PCI with the closest zero point to the given coordinates
        lat, lon = coordinates
        return min(self.zero_points, key=lambda pci: distance.euclidean((lat, lon), self.zero_points[pci]))

