from sklearn.ensemble import RandomForestRegressor
from pykrige.ok import OrdinaryKriging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from scipy.spatial import distance
import pandas as pd
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
        return self.df.apply(lambda row: distance.euclidean((row['latitude'], row['longitude']), zero_points[row['PCI']]), axis=1)

    def cross_validate(self, test_size=0.2, random_state=42):
        if len(self.df) < 5:
            print("Not enough data to perform regression kriging.")
            return None
        total_data_points = len(self.df)
        # Including 'distance_to_zero' directly in X for simplicity
        X = self.df[['latitude', 'longitude', 'distance_to_zero']]
        y = self.df['signalStrength']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        start_time = time.time()

        regression_model = RandomForestRegressor(n_estimators=100, random_state=42)
        regression_model.fit(X_train, y_train)


        regression_predictions_train = regression_model.predict(X_train)
        residuals = y_train - regression_predictions_train


        OK = OrdinaryKriging(X_train['longitude'].values, X_train['latitude'].values, residuals,
                             variogram_model=self.variogram_model)

        regression_predictions_test = regression_model.predict(X_test)
        kriged_residuals, _ = OK.execute('points', X_test['longitude'].values, X_test['latitude'].values)

        final_predictions = regression_predictions_test + kriged_residuals
        end_time = time.time()
        # Calculate performance metrics
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

    def predict(self, coordinates_list):
        # Predict using both the RandomForestRegressor and the residuals from kriging
        # This step assumes the model and kriging parameters have already been trained
        X_predict = pd.DataFrame(coordinates_list, columns=['latitude', 'longitude'])
        X_predict['distance_to_zero'] = self.calculate_distance_to_zero_predict(X_predict)

        regression_predictions = self.regression_model.predict(X_predict)
        kriged_residuals, _ = self.OK.execute('points', X_predict['longitude'].values, X_predict['latitude'].values)
        final_predictions = regression_predictions + kriged_residuals

        return list(zip(coordinates_list, final_predictions))

    def calculate_distance_to_zero_predict(self, X_predict):
        zero_points = self.calculate_zero_points()
        return X_predict.apply(lambda row: distance.euclidean((row['latitude'], row['longitude']), zero_points[row['PCI']]), axis=1)
