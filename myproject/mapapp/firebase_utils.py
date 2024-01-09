# firebase_utils.py
import pyrebase
from django.conf import settings

def get_firebase_database():
    firebase = pyrebase.initialize_app(settings.FIREBASE_CONFIG)
    return firebase.database()

def fetch_data_from_firebase(table_name):
    db = get_firebase_database()
    data = db.child(table_name).get()
    return data.val()  # Assuming this returns a list of dictionaries
