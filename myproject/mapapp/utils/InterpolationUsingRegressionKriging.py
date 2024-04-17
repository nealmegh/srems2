from scipy.spatial import distance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pykrige.ok import OrdinaryKriging
import numpy as np
import pandas as pd
import time
from math import sqrt

class RegressionKriging:

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

        # Calculate zero points and distance to zero points as auxiliary data
        self.df['distance_to_zero'] = self.calculate_distance_to_zero()

    def calculate_zero_points(self):
        zero_points = {}
        for pci, group in self.df.groupby('PCI'):
            max_signal_idx = group['signalStrength'].idxmax()
            zero_points[pci] = group.loc[max_signal_idx, ['latitude', 'longitude']].values
        return zero_points

    def calculate_distance_to_zero(self):
        zero_points = self.calculate_zero_points()
        distances = []
        for index, row in self.df.iterrows():
            pci = row['PCI']
            zero_point = zero_points[pci]
            dist = distance.euclidean((row['latitude'], row['longitude']), zero_point)
            distances.append(dist)
        return distances

    def cross_validate(self, test_size=0.2, random_state=42):
        if len(self.df) < 5:
            print("Not enough data to perform regression kriging.")
            return None
        total_data_points = len(self.df)

        X = self.df[['latitude', 'longitude', 'distance_to_zero']]
        y = self.df['signalStrength']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        start_time = time.time()
        regression_model = LinearRegression()
        regression_model.fit(X_train, y_train)

        # Predict and calculate residuals
        regression_predictions = regression_model.predict(X_train)
        residuals = y_train - regression_predictions

        # Fit the kriging model to the residuals
        OK = OrdinaryKriging(X_train['longitude'].values, X_train['latitude'].values, residuals,
                             variogram_model=self.variogram_model)

        regression_predictions_test = regression_model.predict(X_test)
        kriged_residuals, _ = OK.execute('points', X_test['longitude'].values, X_test['latitude'].values)

        final_predictions = regression_predictions_test + kriged_residuals
        end_time = time.time()

        metrics = {
            'Total Data Points': total_data_points,
            'Training Duration (seconds)': end_time - start_time,
            'MAE': mean_absolute_error(y_test, final_predictions),
            'RMSE': sqrt(mean_squared_error(y_test, final_predictions)),
            'MBE': np.mean(final_predictions - y_test.values),
            'R2': r2_score(y_test, final_predictions),
            'Standard Deviation of Errors': np.std(final_predictions.data - y_test.to_numpy())
        }

        return metrics
