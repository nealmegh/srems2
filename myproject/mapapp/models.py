from django.contrib.gis.db import models
from django.contrib.gis.geos import Point
from django.contrib.auth.models import User
from django.contrib.postgres.fields import JSONField


class NetworkData(models.Model):
    accuracy = models.FloatField()
    asuLevel = models.IntegerField()
    cellId = models.CharField(max_length=255)
    channelBands = models.CharField(max_length=255)
    channelBandwidth = models.BigIntegerField()
    channelQuality = models.IntegerField()
    gpsUpdateTime = models.BigIntegerField()
    latitude = models.FloatField()
    longitude = models.FloatField()
    networkId = models.BigIntegerField()
    networkType = models.CharField(max_length=100)
    networkUpdateTime = models.BigIntegerField()
    operatorName = models.CharField(max_length=100)
    signalStrength = models.IntegerField()
    cellId_PCI = models.CharField(max_length=255)
    cellId_TAC = models.CharField(max_length=255)
    location = models.PointField(null=True)  # Add this line

    def save(self, *args, **kwargs):
        self.location = Point(self.longitude, self.latitude)
        super().save(*args, **kwargs)

    def save(self, *args, **kwargs):
        # Convert latitude and longitude to float, if possible
        try:
            lat = float(self.latitude) if self.latitude is not None else None
            lon = float(self.longitude) if self.longitude is not None else None
        except (TypeError, ValueError) as e:
            print(f"Error converting coordinates to float: {e}")
            return

        # Check if latitude or longitude is None
        if lat is None or lon is None:
            print(f"Record not saved: Latitude or Longitude is None (Lat: {lat}, Lon: {lon})")
            return

        # Check if coordinates are within valid ranges
        try:
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                self.location = Point(lon, lat)
            else:
                print(f"Record not saved: Invalid coordinates (Lat: {lat}, Lon: {lon})")
                return
        except TypeError as e:
            print(f"Error in range checking for coordinates: {e}")
            return

        super().save(*args, **kwargs)


class InterpolatedNetwork(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    operator_name = models.CharField(max_length=100)
    coordinates = models.TextField()  # Storing as text, could be JSON too
    data_selection = models.CharField(max_length=50)
    interpolation_technique = models.CharField(max_length=50)
    interpolated_data = models.JSONField()  # Updated to use standard JSONField
    metrics = models.JSONField()  # As above
    summary_metrics = models.JSONField(blank=True, null=True)  # Optional field
    area = models.FloatField()
    creation_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Query by {self.user.username} on {self.creation_date}"



