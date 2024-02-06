import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.spatial import distance
import matplotlib.colors as mcolors
from sklearn.metrics import mean_absolute_error
import seaborn as sns
from sklearn.metrics import r2_score
import time
from sklearn.model_selection import cross_val_score


class InterpolationByPciUsingRF:
    # def __init__(self, results):
    #     # Convert results to a DataFrame
    #     self.df = pd.DataFrame([{
    #         'latitude': r.latitude,
    #         'longitude': r.longitude,
    #         'PCI': r.cellId_PCI,
    #         'signalStrength': r.signalStrength
    #     } for r in results])
    #     self.pci_groups = self.df.groupby('PCI')
    #     self.models = {}
    #     self.point_zero = {}
    #     self.boundaries = {}
    def __init__(self, results, data_source, min_samples_per_pci=60):
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

        self.pci_groups = self.df.groupby('PCI').filter(lambda x: len(x) > min_samples_per_pci).groupby('PCI')
        self.models = {}
        self.point_zero = {}
        self.boundaries = {}

    def _prepare_pci_data(self, pci):
        df_pci = self.pci_groups.get_group(pci)
        if len(df_pci) < 2:  # Threshold for minimum number of samples
            return None, None, None, None
        X = df_pci[['latitude', 'longitude']]
        y = df_pci['signalStrength']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    # def train_pci_models(self):
    #         global rmse_cv
    #         metrics_df = pd.DataFrame(
    #             columns=['PCI', 'Data_Points', 'Training_Time', 'Cross_Validated_RMSE', 'MAE', 'RMSE', 'MBE', 'R2',
    #                      'Standard_Deviation']
    #         )
    #
    #         for pci in self.pci_groups.groups.keys():
    #             X_train, X_test, y_train, y_test = self._prepare_pci_data(pci)
    #             if X_train is not None and len(X_test) > 1:
    #                 model = RandomForestRegressor(n_estimators=100, random_state=42)
    #                 num_data_points = len(X_train)
    #                 start_train_time = time.time()
    #
    #                 num_folds = min(5, len(X_test))
    #                 if num_folds > 1:
    #                     scores = cross_val_score(model, X_train, y_train, cv=num_folds, scoring='neg_mean_squared_error')
    #                     rmse_cv = np.sqrt(-np.mean(scores))
    #                     # print(f"Cross-validated RMSE for PCI {pci} with {num_folds} folds: {rmse_cv}")
    #
    #                 model.fit(X_train, y_train)
    #                 self.models[pci] = model
    #
    #                 model.fit(X_train, y_train)
    #                 y_pred = model.predict(X_test)
    #
    #                 # Calculate various performance metrics
    #                 mae = mean_absolute_error(y_test, y_pred)
    #                 rmse = sqrt(mean_squared_error(y_test, y_pred))
    #                 mbe = np.mean(y_pred - y_test)
    #                 r2 = r2_score(y_test, y_pred)
    #                 std_dev = np.std(y_pred - y_test)
    #
    #                 end_train_time = time.time()
    #                 training_duration = end_train_time - start_train_time
    #                 index = len(metrics_df)
    #                 metrics_df.loc[index] = [pci, num_data_points, training_duration, rmse_cv, mae, rmse, mbe, r2,
    #                                          std_dev]
    #
    #             else:
    #                 print(f"Not enough data to train model for PCI {pci}")
    #
    #         self.metrics = metrics_df.to_dict(orient='records')
    #         return self.metrics

    def train_pci_models(self):
        metrics_df = pd.DataFrame(
            columns=['PCI', 'Data_Points', 'Training_Time', 'Cross_Validated_RMSE', 'MAE', 'RMSE', 'MBE', 'R2',
                     'Standard_Deviation']
        )

        # Train a model for each PCI group
        for pci, group in self.pci_groups:
            X_train, X_test, y_train, y_test = train_test_split(
                group[['latitude', 'longitude']],
                group['signalStrength'],
                test_size=0.2,
                random_state=42
            )

            model = RandomForestRegressor(n_estimators=100, random_state=42)

            num_data_points = len(X_train)
            # Cross-validation
            num_folds = min(5, len(X_test))
            rmse_cv = None
            if num_folds > 1:
                scores = cross_val_score(model, X_train, y_train, cv=num_folds, scoring='neg_mean_squared_error')
                rmse_cv = np.sqrt(-np.mean(scores))
            start_train_time = time.time()
            # Training
            model.fit(X_train, y_train)
            end_train_time = time.time()
            self.models[pci] = model

            # Predictions and performance metrics
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = sqrt(mean_squared_error(y_test, y_pred))
            mbe = np.mean(y_pred - y_test)
            r2 = r2_score(y_test, y_pred)
            std_dev = np.std(y_pred - y_test)


            training_duration = end_train_time - start_train_time
            index = len(metrics_df)
            metrics_df.loc[index] = [pci, num_data_points, training_duration, rmse_cv, mae, rmse, mbe, r2,
                                     std_dev]

        self.metrics = metrics_df.to_dict(orient='records')
        return self.metrics

    def _calculate_boundaries_and_point_zero(self):
        for pci, group in self.pci_groups:
            lat_min, lat_max = group['latitude'].min(), group['latitude'].max()
            lon_min, lon_max = group['longitude'].min(), group['longitude'].max()
            self.boundaries[pci] = ((lat_min, lon_min), (lat_max, lon_max))

            strongest_point = group.loc[group['signalStrength'].idxmax()]
            self.point_zero[pci] = (strongest_point['latitude'], strongest_point['longitude'])

    def _find_closest_pci(self, lat, lon):
        closest_pci = None
        min_distance = float('inf')

        for pci, point in self.point_zero.items():
            dist = distance.euclidean((lat, lon), point)
            if dist < min_distance:
                min_distance = dist
                closest_pci = pci

        if closest_pci is None:
            print('NO PCI associated')
        return closest_pci

    def predict(self, coordinates_list):
        self._calculate_boundaries_and_point_zero()
        predictions = []
        count = 0
        model_prediction_times = {}

        for lat, lon in coordinates_list:
            pci = self._find_closest_pci(lat, lon)
            if pci and self.models.get(pci):
                count += 1
                start_pci_prediction_time = time.time()
                signal_strength = self.models[pci].predict([[lat, lon]])[0]
                end_pci_prediction_time = time.time()

                pci_prediction_time = end_pci_prediction_time - start_pci_prediction_time
                model_prediction_times[pci] = model_prediction_times.get(pci, 0) + pci_prediction_time

                predictions.append((lat, lon, signal_strength))

        # Optional: Print the total prediction time for each PCI (can be commented out in production)
        for pci, times in model_prediction_times.items():
            print(f"Total prediction time for PCI {pci}: {times}")

        return predictions, model_prediction_times
