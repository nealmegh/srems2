import pandas as pd
import numpy as np
from pykrige.uk import UniversalKriging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
from math import sqrt

class InterpolationUsingUniKriging:

    def __init__(self, results, data_source, variogram_model, min_samples_per_pci=60):
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

    def cross_validate(self, test_size=0.2, random_state=42):
        if len(self.df) < 5:
            print("Not enough data to perform interpolation.")
            return None

        total_data_points = len(self.df)

        X = self.df[['latitude', 'longitude']]
        y = self.df['signalStrength']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        start_time = time.time()

        UK = UniversalKriging(X_train['longitude'].values, X_train['latitude'].values, y_train.values,
                             variogram_model=self.variogram_model)

        z_pred, _ = UK.execute('points', X_test['longitude'].values, X_test['latitude'].values)
        end_time = time.time()

        # Calculate performance metrics
        mae = mean_absolute_error(y_test.values, z_pred)
        rmse = sqrt(mean_squared_error(y_test.values, z_pred))
        mbe = np.mean(z_pred - y_test.values)
        r2 = r2_score(y_test.values, z_pred)
        std_dev = np.std(z_pred - y_test.values)

        # Calculate RMSE for each prediction and then compute the mean
        # rmse_cv = np.sqrt(np.mean((y_test.values - z_pred) ** 2))



        metrics = {
            'Total Data Points': total_data_points,
            'Training Duration (seconds)': end_time - start_time,
            'Cross-validated RMSE': None,
            'MAE': mae,
            'RMSE': rmse,
            'MBE': mbe,
            'R2': r2,
            'Standard Deviation of Errors': std_dev
        }

        return metrics

    def predict(self, coordinates_list):

        start_prediction_time = time.time()
        UK = UniversalKriging(self.df['longitude'].values, self.df['latitude'].values,
                             self.df['signalStrength'].values, variogram_model=self.variogram_model)
        latitudes, longitudes = zip(*coordinates_list)

        z_pred, _ = UK.execute('points', longitudes, latitudes)
        end_prediction_time = time.time()
        prediction_time = end_prediction_time - start_prediction_time

        return list(zip(latitudes, longitudes, z_pred)), prediction_time
