import time

import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
# Import other necessary libraries like numpy, etc.
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class IDWInterpolationByPCI:

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
        self.power = 2
        # self.pci_groups = self.df.groupby('PCI')
        self.pci_groups = self.df.groupby('PCI').filter(lambda x: len(x) >= min_samples_per_pci).groupby('PCI')
        self.boundaries = {}
        self.point_zero = {}
        self._calculate_boundaries_and_point_zero()

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

    def predict(self, coordinates_list):
        interpolated_values = []
        model_prediction_times = {}
        for coord in coordinates_list:
            closest_pci = self._find_closest_pci(coord[0], coord[1])
            pci_group = self.pci_groups.get_group(closest_pci) if closest_pci in self.pci_groups.groups else None
            if pci_group is not None:
                start_pci_prediction_time = time.time()
                interpolated_value = self.interpolate_single_point(coord, pci_group)
                interpolated_values.append((coord[0], coord[1], interpolated_value))
                end_pci_prediction_time = time.time()
                pci_prediction_time = end_pci_prediction_time - start_pci_prediction_time
                model_prediction_times[closest_pci] = model_prediction_times.get(closest_pci, 0) + pci_prediction_time
        return interpolated_values, model_prediction_times

    def calculate_distances(self, latitudes, longitudes, target_coord):
        # Radius of the Earth in km
        earth_radius = 6371.0

        target_lat, target_lon = target_coord
        target_lat = radians(target_lat)
        target_lon = radians(target_lon)

        latitudes = [radians(lat) for lat in latitudes]
        longitudes = [radians(lon) for lon in longitudes]

        dlat = [lat - target_lat for lat in latitudes]
        dlon = [lon - target_lon for lon in longitudes]

        a = [sin(dlat_i / 2) ** 2 + cos(radians(target_lat)) * cos(radians(latitudes[i])) * sin(dlon_i / 2) ** 2 for
             i, (dlat_i, dlon_i) in enumerate(zip(dlat, dlon))]
        distances = [2 * atan2(sqrt(a_i), sqrt(1 - a_i)) * earth_radius for a_i in a]

        return distances

    def interpolate_single_point(self, target_coord, pci_group):
        # Extract latitude, longitude, and signal strength from the PCI group
        latitudes = pci_group['latitude'].values
        longitudes = pci_group['longitude'].values
        signal_strengths = pci_group['signalStrength'].values

        # Calculate distances from the target coordinate to each point in the PCI group
        distances = self.calculate_distances(latitudes, longitudes, target_coord)

        # Handle zero distances to avoid division by zero
        distances = [1 if d == 0 else d for d in distances]

        # Calculate weights using the inverse distance weighting formula
        weights = [1 / (d ** self.power) for d in distances]

        # Calculate weighted signal strengths
        weighted_values = [signal_strengths[i] * weights[i] for i in range(len(weights))]

        # Sum the weighted values and divide by the sum of the weights to get the interpolated value
        if None not in weights and sum(weights) != 0:
            interpolated_value = sum(weighted_values) / sum(weights)
        else:
            interpolated_value = np.nan  # Handle cases where interpolation is not possible

        return interpolated_value

    def calculate_pci_metrics(self):
        metrics_df = pd.DataFrame(columns=['PCI', 'Data_Points', 'Training_Time',
                                           'Cross_Validated_RMSE', 'MAE', 'RMSE', 'MBE', 'R2', 'Standard_Deviation'])

        for pci, group in self.pci_groups:
            if len(group) < 5:  # Check if there are enough samples
                print(f"Not enough data for PCI {pci}. Skipping.")
                continue  # Skip this PCI group
            # Split data into training and testing sets
            train_set, test_set = train_test_split(group, test_size=0.2, random_state=42)

            # Update the data in the object to the training set
            self.df = train_set

            # Prepare actual and predicted values for the test set
            test_coords = test_set[['latitude', 'longitude']].values
            actual_values = test_set['signalStrength'].values
            start_time = time.time()
            predicted_values = [self.interpolate_single_point(coord, train_set) for coord in test_coords]
            end_time = time.time()

            # Calculate performance metrics
            mae = mean_absolute_error(actual_values, predicted_values)
            rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
            mbe = np.mean(np.array(predicted_values) - np.array(actual_values))
            r2 = r2_score(actual_values, predicted_values)
            std_dev = np.std(np.array(predicted_values) - np.array(actual_values))

            # Add to DataFrame
            metrics_df.loc[len(metrics_df)] = [pci, len(group), 0, None, mae, rmse, mbe, r2, std_dev]

        self.metrics = metrics_df.to_dict(orient='records')

        return self.metrics