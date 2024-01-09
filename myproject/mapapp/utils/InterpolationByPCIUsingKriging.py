import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.spatial import distance
import time
from math import sqrt

class InterpolationByPCIUsingKriging:

    # def __init__(self, results):
    #     self.df = pd.DataFrame([{
    #         'latitude': r.latitude,
    #         'longitude': r.longitude,
    #         'PCI': r.cellId_PCI,
    #         'signalStrength': r.signalStrength
    #     } for r in results])
    #     self.pci_groups = self.df.groupby('PCI')
    #     self.kriging_models = {}
    #     self.point_zero = {}
    #     self.boundaries = {}
    def __init__(self, results, min_samples_per_pci=60):
        self.df = pd.DataFrame([{
            'latitude': r.latitude,
            'longitude': r.longitude,
            'PCI': r.cellId_PCI,
            'signalStrength': r.signalStrength
        } for r in results])

        # Filter out PCI groups with insufficient data
        self.pci_groups = self.df.groupby('PCI').filter(lambda x: len(x) >= min_samples_per_pci).groupby('PCI')
        self.kriging_models = {}
        self.point_zero = {}
        self.boundaries = {}

    def _prepare_pci_data(self, pci):
        df_pci = self.pci_groups.get_group(pci)
        if len(df_pci) < 5:
            return None, None, None, None
        X = df_pci[['latitude', 'longitude']]
        y = df_pci['signalStrength']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    # def cross_validate(self, variogram_model='exponential'):
    #     metrics_df = pd.DataFrame(columns=['PCI', 'Data_Points', 'Training_Time', 'Cross_Validated_RMSE', 'MAE', 'RMSE', 'MBE', 'R2', 'Standard_Deviation'])
    #
    #     for pci in self.pci_groups.groups.keys():
    #         X_train, X_test, y_train, y_test = self._prepare_pci_data(pci)
    #         if X_train is not None and len(X_test) > 1:
    #             start_train_time = time.time()
    #
    #             OK = OrdinaryKriging(X_train['longitude'].values, X_train['latitude'].values, y_train.values, variogram_model=variogram_model)
    #             self.kriging_models[pci] = OK
    #
    #             z_pred, _ = OK.execute('points', X_test['longitude'].values, X_test['latitude'].values)
    #
    #             mae = mean_absolute_error(y_test.values, z_pred)
    #             rmse = sqrt(mean_squared_error(y_test.values, z_pred))
    #             mbe = np.mean(z_pred - y_test.values)
    #             r2 = r2_score(y_test.values, z_pred)
    #             std_dev = np.std(z_pred - y_test.values)
    #
    #             end_train_time = time.time()
    #             training_duration = end_train_time - start_train_time
    #
    #             metrics_df.loc[len(metrics_df)] = [pci, len(X_train), training_duration, None, mae, rmse, mbe, r2, std_dev]
    #         else:
    #             print(f"Not enough data to train model for PCI {pci}")
    #
    #     self.metrics = metrics_df.to_dict(orient='records')
    #     return self.metrics
    def cross_validate(self, variogram_model='gaussian'):
        metrics_df = pd.DataFrame(columns=[
            'PCI', 'Data_Points', 'Training_Time', 'Cross_Validated_RMSE',
            'MAE', 'RMSE', 'MBE', 'R2', 'Standard_Deviation'
        ])

        for pci, group in self.pci_groups:
            X_train, X_test, y_train, y_test = train_test_split(
                group[['latitude', 'longitude']],
                group['signalStrength'],
                test_size=0.2,
                random_state=42
            )

            start_train_time = time.time()

            OK = OrdinaryKriging(
                X_train['longitude'].values, X_train['latitude'].values,
                y_train.values, variogram_model=variogram_model
            )
            self.kriging_models[pci] = OK


            z_pred, _ = OK.execute('points', X_test['longitude'].values, X_test['latitude'].values)
            end_train_time = time.time()


            mae = mean_absolute_error(y_test.values, z_pred)
            rmse = sqrt(mean_squared_error(y_test.values, z_pred))
            mbe = np.mean(z_pred - y_test.values)
            r2 = r2_score(y_test.values, z_pred)
            std_dev = np.std(z_pred - y_test.values)


            training_duration = end_train_time - start_train_time
            metrics_df.loc[len(metrics_df)] = [pci, len(X_train), training_duration, None, mae, rmse, mbe, r2, std_dev]

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

        return closest_pci

    def predict(self, coordinates_list, variogram_model='exponential'):
        self._calculate_boundaries_and_point_zero()
        predictions = []
        model_prediction_times = {}
        for lat, lon in coordinates_list:
            pci = self._find_closest_pci(lat, lon)
            start_pci_prediction_time = time.time()
            if pci and pci in self.kriging_models:
                OK = self.kriging_models[pci]
                signal_strength, _ = OK.execute('points', [lon], [lat])
                predicted_value = signal_strength[0] if signal_strength.size > 0 else np.nan
                predictions.append((lat, lon, predicted_value))
            else:
                # Handle case where no Kriging model is available for the closest PCI
                predictions.append((lat, lon, np.nan))
            end_pci_prediction_time = time.time()

            pci_prediction_time = end_pci_prediction_time - start_pci_prediction_time
            model_prediction_times[pci] = model_prediction_times.get(pci, 0) + pci_prediction_time

        return predictions, model_prediction_times
