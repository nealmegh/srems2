import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import time


class SignalStrengthInterpolatorIDW:

    def __init__(self, results):
        # Convert results to a DataFrame
        self.df = pd.DataFrame([{
            'latitude': r.latitude,
            'longitude': r.longitude,
            'signalStrength': r.signalStrength
        } for r in results])
        self.power = 2  # default power for IDW

    def train_model(self):
        print("IDW model does not require training.")

    def predict(self, coordinates_list):
        start_prediction_time = time.time()
        interpolated_values = []
        for coord in coordinates_list:
            interpolated_value = self.interpolate_single_point(coord)
            interpolated_values.append((coord[0], coord[1], interpolated_value))
        end_prediction_time = time.time()
        prediction_time = end_prediction_time - start_prediction_time
        return interpolated_values, prediction_time

    def interpolate_single_point(self, target_coord):
        distances = self.calculate_distances(self.df['latitude'].values, self.df['longitude'].values, target_coord)

        # Handle zero distances
        distances = [1 if d == 0 else d for d in distances]

        weights = [1 / (d ** self.power) for d in distances]
        weighted_values = [self.df['signalStrength'].iloc[i] * weights[i] for i in range(len(weights))]

        if None not in weights:
            interpolated_value = sum(weighted_values) / sum(weights)
        else:
            interpolated_value = np.nan

        return interpolated_value

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

    def calculate_performance_metrics(self, test_size=0.2):
        # Split data in self.df into reference and test sets
        reference_set, test_set = train_test_split(self.df, test_size=test_size)

        # Update the data in the object to reference set
        self.df = reference_set

        # Prepare actual and predicted values
        test_coords = test_set[['latitude', 'longitude']].values
        actual_values = test_set['signalStrength'].values
        # predicted_values = [val[2] for val in self.predict(test_coords)]
        interpolated_values, prediction_time = self.predict(test_coords)
        predicted_values = [val[2] for val in interpolated_values]
        # Calculate performance metrics
        mae = mean_absolute_error(actual_values, predicted_values)
        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        mbe = np.mean(np.array(predicted_values) - np.array(actual_values))
        r2 = r2_score(actual_values, predicted_values)
        std_dev = np.std(np.array(predicted_values) - np.array(actual_values))

        # Format the metrics as specified
        metrics = {
            'Total Data Points': len(self.df),
            'Training Duration (seconds)': 0,  # IDW doesn't require training
            'Cross-validated RMSE': None,  # Not applicable for IDW
            'MAE': mae,
            'RMSE': rmse,
            'MBE': mbe,
            'R2': r2,
            'Standard Deviation of Errors': std_dev
        }

        return metrics
