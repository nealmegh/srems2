import requests
import pandas as pd
import geopandas as gpd
import numpy as np
import json
from shapely.geometry import Point


class RoadCoverageChecker:
    BUFFER_DISTANCE = 10  # in meters

    def __init__(self, boundary_coords, result_data):
        """Initialize the checker with boundary coordinates and result data."""
        self.boundary_coords = boundary_coords
        self.result_data = result_data
        self.roads_gdf = self.fetch_roads_within_boundary()
        self.gdf_points = self.convert_results_to_gdf()
        self.uncovered_roads = self.check_road_coverage()

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
        nodes = {feature['id']: (feature['lon'], feature['lat']) for feature in data['elements'] if feature['type'] == 'node'}

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
        """Check coverage of entire road and return uncovered roads."""
        uncovered_roads = set()

        # Convert road geometries to same CRS as points for proper buffer operation
        roads_gdf_projected = self.roads_gdf.to_crs(epsg=3395)

        for _, road in roads_gdf_projected.iterrows():
            road_geom = road['geometry']
            road_buffered = road_geom.buffer(self.BUFFER_DISTANCE)
            road_covered = self.gdf_points.within(road_buffered).any()

            road_name = road.get('name', 'Unnamed')
            if not road_covered and road_name not in [None, 'nan', np.nan]:
                uncovered_roads.add((road_name, road['id']))
        # print(uncovered_roads)
        return uncovered_roads

    def get_uncovered_road_ids(self):
        """Return a set of uncovered road IDs."""
        return {road_id for _, road_id in self.uncovered_roads}