import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import time
from scipy.spatial import distance
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=UserWarning)

class InterpolationUsingRF:
    def __init__(self, results):
        # Convert results to a DataFrame
        self.df = pd.DataFrame([{
            'latitude': r.latitude,
            'longitude': r.longitude,
            'PCI': r.cellId_PCI,  # Assuming results have cellId_PCI attribute
            'signalStrength': r.signalStrength
        } for r in results])

    def train_model(self):
        if len(self.df) < 5:  # Threshold for minimum number of samples
            print("Not enough data to train the model.")
            return

        total_data_points = len(self.df)  # Total number of data points

        X = self.df[['latitude', 'longitude']]
        y = self.df['signalStrength']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)


        # Cross-validation
        num_folds = min(5, len(X_test))
        rmse_cv = None
        if num_folds > 1:
            scores = cross_val_score(self.model, X_train, y_train, cv=num_folds, scoring='neg_mean_squared_error')
            rmse_cv = np.sqrt(-np.mean(scores))
            print(f"Cross-validated RMSE: {rmse_cv}")
        start_time = time.time()
        # Training
        self.model.fit(X_train, y_train)
        end_time = time.time()
        # Prediction and performance metrics
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        mbe = np.mean(y_pred - y_test)
        r2 = r2_score(y_test, y_pred)
        std_dev = np.std(y_pred - y_test)


        training_duration = end_time - start_time  # Training duration
        print(f"Training took {training_duration:.2f} seconds")

        # Performance metrics
        metrics = {
            'Total Data Points': total_data_points,
            'Training Duration (seconds)': training_duration,
            'Cross-validated RMSE': rmse_cv,
            'MAE': mae,
            'RMSE': rmse,
            'MBE': mbe,
            'R2': r2,
            'Standard Deviation of Errors': std_dev
        }

        return metrics

    def predict(self, coordinates_list):
        predictions = []
        start_prediction_time = time.time()
        for lat, lon in coordinates_list:
            signal_strength = self.model.predict([[lat, lon]])[0]
            predictions.append((lat, lon, signal_strength))
        end_prediction_time = time.time()
        prediction_time = end_prediction_time - start_prediction_time
        return predictions, prediction_time
