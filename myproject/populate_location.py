# populate_location.py
import os
import django
from django.contrib.gis.geos import Point

# Setting up Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")
django.setup()

# Import your model
from mapapp.models import NetworkData


# Function to populate location field
def populate_location():
    for obj in NetworkData.objects.filter(location__isnull=True):
        obj.location = Point(obj.longitude, obj.latitude)
        obj.save()
    print("Location field populated for all NetworkData objects.")


# Run the function
populate_location()
