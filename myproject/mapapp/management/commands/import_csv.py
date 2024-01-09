import csv
from django.core.management.base import BaseCommand
from mapapp.models import NetworkData


class Command(BaseCommand):
    help = 'Import data from CSV file into the database'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str)

    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']
        print(f"CSV file path: {csv_file}")  # Debug print
        with open(csv_file, newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                NetworkData.objects.create(**row)
