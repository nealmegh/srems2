import geopandas as gpd
import pandas as pd
import requests
import json
import numpy as np
from shapely.geometry import Point as ShapelyPoint
from django.contrib.gis.geos import Point as DjangoPoint
from shapely.wkt import loads

class RoadCoverageChecker2:
    BUFFER_DISTANCE = 20  # in meters

    def __init__(self, boundary_coords, result_data, output_filename="covered_roads123.xlsx"):
        """Initialize the checker with boundary coordinates and result data."""
        self.boundary_coords = boundary_coords
        self.result_data = result_data
        self.output_filename = output_filename
        self.roads_gdf = self.fetch_roads_within_boundary()
        self.gdf_points = self.convert_results_to_gdf()
        self.covered_roads = self.check_road_coverage()


    def convert_results_to_gdf(self):
        """Convert result data into a GeoDataFrame."""
        gdf_points = gpd.GeoDataFrame(
            pd.DataFrame(self.result_data),
            geometry=gpd.points_from_xy(pd.DataFrame(self.result_data)['longitude'], pd.DataFrame(self.result_data)['latitude']),
            crs="EPSG:4326"
        )
        return gdf_points.to_crs(epsg=3395)

    def fetch_roads_within_boundary(self):
        """Fetch roads within the boundary using Overpass API."""
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
        way["highway"](poly:"{' '.join([f'{lat} {lon}' for lon, lat in self.boundary_coords])}");
        (._;>;);
        out body;
        """
        response = requests.get(overpass_url, params={'data': overpass_query})
        data = response.json()
        # Convert the data to a string
        data_string = json.dumps(data, indent=4)  # Pretty print the JSON

        # Write to a text file
        with open('data_output.txt', 'w') as file:
            file.write(data_string)

        ways = [feature for feature in data['elements'] if feature['type'] == 'way']
        nodes = {feature['id']: (feature['lon'], feature['lat']) for feature in data['elements'] if
                 feature['type'] == 'node'}

        road_geometries = []
        for way in ways:
            coordinates = [nodes[node] for node in way['nodes']]
            road_geometries.append({
                'type': 'Feature',
                'id': way['id'],
                'properties': way.get('tags', {}),
                'geometry': {
                    'type': 'LineString',
                    'coordinates': coordinates
                }
            })

        roads_gdf = gpd.GeoDataFrame.from_features(road_geometries)
        roads_gdf['id'] = [feature['id'] for feature in road_geometries]
        return roads_gdf.dropna(subset=['geometry']).set_geometry('geometry').set_crs("EPSG:4326")

    def check_road_coverage(self):
        """Check coverage of entire road, return unique covered roads by name, and export to Excel."""
        covered_roads_info = []

        roads_gdf_projected = self.roads_gdf.to_crs(epsg=3395)

        for _, road in roads_gdf_projected.iterrows():
            road_geom = road['geometry']
            road_buffered = road_geom.buffer(self.BUFFER_DISTANCE)
            road_covered = self.gdf_points.within(road_buffered).any()

            if road_covered and road.get('name') not in [None, 'nan', np.nan]:
                covered_roads_info.append((road.get('name'), road['id']))

        # Convert to DataFrame and export to Excel
        if covered_roads_info:
            df = pd.DataFrame(covered_roads_info, columns=['Road Name', 'OSM ID'])
            df.to_excel(self.output_filename, index=False)
            print(f"Covered roads data has been exported to {self.output_filename}.")
        else:
            print("No covered roads found.")

        return covered_roads_info