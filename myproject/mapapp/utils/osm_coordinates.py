# import osmapi
#
#
# class OSMCoordinates:
#     def __init__(self):
#         self.api = osmapi.OsmApi()
#
#     def get_coordinates_from_id(self, osm_id):
#         try:
#             way = self.api.WayGet(osm_id)
#         except osmapi.errors.ElementNotFoundApiError:
#             print(f"The OSM ID {osm_id} does not exist or is not a 'way'.")
#             return None
#
#         if not way:
#             return None
#
#         coordinates = []
#         for node_id in way['nd']:
#             node = self.api.NodeGet(node_id)
#             if node:
#                 lat, lon = node['lat'], node['lon']
#                 coordinates.append((lat, lon))
#
#         return coordinates
#
#     def get_coordinates_for_ids(self, osm_ids):
#         return {osm_id: self.get_coordinates_from_id(osm_id) for osm_id in osm_ids}

# import osmnx as ox
# from shapely.geometry import LineString, MultiLineString
#
# class OSMCoordinates:
#     def __init__(self, polygon, results):
#         """
#         Initializes the RoadDataAnalyzer with a polygon and query results.
#
#         :param polygon: Shapely Polygon object representing the area of interest.
#         :param results: Django QuerySet of NetworkData objects.
#         """
#         self.polygon = polygon
#         self.results = results
#         self.existing_data_coords = {(obj.longitude, obj.latitude) for obj in results}
# #
#     def get_road_coordinates_without_data(self):
#         """
#         Retrieves coordinates of roads within the polygon that are not in the results.
#
#         :return: Dictionary with OSM IDs and coordinates of roads lacking data.
#         """
#         # Retrieve the graph of all streets within the specified polygon
#         graph = ox.graph_from_polygon(self.polygon, network_type='drive')
#         edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
#
#         roads_without_data = {}
#
#         for osm_id, edge in edges.iterrows():
#             geometry = edge['geometry']
#             coords = self._get_coordinates_from_geometry(geometry)
#
#             # Format the coordinates as a list of tuples (lat, lon)
#             formatted_coords = [(coord[1], coord[0]) for coord in coords]
#
#             # Check if any coordinate of the road is not in existing data
#             if not any(coord in self.existing_data_coords for coord in formatted_coords):
#                 roads_without_data[osm_id] = formatted_coords
#
#         return roads_without_data
#
#
#     def _get_coordinates_from_geometry(self, geometry):
#         """
#         Extracts coordinates from a geometry object.
#
#         :param geometry: Shapely geometry object (LineString or MultiLineString).
#         :return: List of coordinate tuples.
#         """
#         coords = []
#         if isinstance(geometry, LineString):
#             coords = list(geometry.coords)
#         elif isinstance(geometry, MultiLineString):
#             for linestring in geometry:
#                 coords.extend(list(linestring.coords))
#         return coords

import osmnx as ox
from shapely.geometry import LineString, MultiLineString
import numpy as np
import time
from django.conf import settings

# ox.settings.cache_folder = settings.CACHES['default']['LOCATION']
# ox.settings.cahe_only_mode=False

class OSMCoordinates:
    def __init__(self, polygon, results, data_source):
        """
        Initializes the OSMCoordinates with a polygon and query results.

        :param polygon: Shapely Polygon object representing the area of interest.
        :param results: Django QuerySet of NetworkData objects.
        """
        self.polygon = polygon
        self.results = results
        if data_source != 'csv':
            self.existing_data_coords = {(obj.longitude, obj.latitude) for obj in results}
        else:
            self.existing_data_coords = {(obj['longitude'], obj['latitude']) for obj in results}

    def get_road_coordinates_without_data(self):

        start_time = time.time()
        """
        Retrieves coordinates of roads within the polygon that are not in the results,
        sorted from top (north) to bottom (south).

        :return: Dictionary with OSM IDs and coordinates of roads lacking data.
        """
        # Retrieve the graph of all streets within the specified polygon
        graph = ox.graph_from_polygon(self.polygon, network_type='drive')
        edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)

        roads_without_data = {}

        for osm_id, edge in edges.iterrows():
            geometry = edge['geometry']
            coords = self._get_coordinates_from_geometry(geometry)

            # Check if any coordinate of the road is not in existing data
            if not any(coord in self.existing_data_coords for coord in coords):
                # Interpolate and format the coordinates as a list of tuples (lat, lon)
                interpolated_coords = self._interpolate_road_coordinates(coords)
                formatted_interpolated_coords = [(coord[1], coord[0]) for coord in interpolated_coords]

                # Sort coordinates from top to bottom (north to south)
                sorted_coords = sorted(formatted_interpolated_coords, key=lambda x: x[0], reverse=True)

                roads_without_data[osm_id] = sorted_coords
        end_time = time.time()
        time_total = end_time - start_time
        print(time_total)
        return roads_without_data

    # def get_road_coordinates_without_data(self):
    #     graph = ox.graph_from_polygon(self.polygon, network_type='all')
    #     edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    #
    #     roads_without_data = {}
    #
    #     for osm_id, edge in edges.iterrows():
    #         geometry = edge['geometry']
    #         coords = self._get_coordinates_from_geometry(geometry)
    #
    #         # Interpolate and format the coordinates as a list of tuples (lat, lon)
    #         interpolated_coords = self._interpolate_road_coordinates(coords)
    #         filtered_coords = [(coord[1], coord[0]) for coord in interpolated_coords
    #                            if not self._has_nearby_existing_data(coord)]
    #
    #         if filtered_coords:
    #             # Sort coordinates from top to bottom (north to south)
    #             sorted_coords = sorted(filtered_coords, key=lambda x: x[0], reverse=True)
    #             roads_without_data[osm_id] = sorted_coords
    #
    #     return roads_without_data
    #
    # def _has_nearby_existing_data(self, coord, radius=10):
    #     for existing_coord in self.existing_data_coords:
    #         if self._haversine(coord[1], coord[0], existing_coord[1], existing_coord[0]) <= radius:
    #             return True
    #     return False

    # def get_road_coordinates_without_data(self):
    #     start_time = time.time()
    #     graph = ox.graph_from_polygon(self.polygon, network_type='drive')
    #     edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    #
    #     roads_without_data = {}
    #
    #     # Filter roads first
    #     filtered_osm_ids = [osm_id for osm_id, edge in edges.iterrows() if
    #                         not self._road_has_existing_data(edge['geometry'])]
    #
    #     for osm_id in filtered_osm_ids:
    #         edge = edges.loc[osm_id]
    #         geometry = edge['geometry']
    #         coords = self._get_coordinates_from_geometry(geometry)
    #
    #         interpolated_coords = self._interpolate_road_coordinates(coords)
    #         formatted_interpolated_coords = [(coord[1], coord[0]) for coord in interpolated_coords]
    #
    #         sorted_coords = sorted(formatted_interpolated_coords, key=lambda x: x[0], reverse=True)
    #         roads_without_data[osm_id] = sorted_coords
    #     end_time = time.time()
    #     time_total = end_time - start_time
    #     print(time_total)
    #     return roads_without_data
    #
    # def _road_has_existing_data(self, geometry, radius=5):
    #     coords = self._get_coordinates_from_geometry(geometry)
    #     for coord in coords:
    #         if any(self._haversine(coord[1], coord[0], existing_coord[1], existing_coord[0]) <= radius
    #                for existing_coord in self.existing_data_coords):
    #             return True
    #     return False

    def _get_coordinates_from_geometry(self, geometry):
        """
        Extracts coordinates from a geometry object.

        :param geometry: Shapely geometry object (LineString or MultiLineString).
        :return: List of coordinate tuples.
        """
        coords = []
        if isinstance(geometry, LineString):
            coords = list(geometry.coords)
        elif isinstance(geometry, MultiLineString):
            for linestring in geometry:
                coords.extend(list(linestring.coords))
        return coords

    def _interpolate_road_coordinates(self, coords, interval=20):
        """
        Interpolates coordinates along a road every 'interval' meters.

        :param coords: List of coordinate tuples (lat, lon).
        :param interval: Distance in meters between interpolated points.
        :return: List of interpolated coordinate tuples.
        """
        interpolated_coords = []
        for i in range(len(coords) - 1):
            start_lat, start_lon = coords[i]
            end_lat, end_lon = coords[i + 1]
            interpolated_segment = self._interpolate_coordinates(start_lat, start_lon, end_lat, end_lon, interval)
            interpolated_coords.extend(interpolated_segment)

        # Add the last point of the segment
        interpolated_coords.append(coords[-1])
        return interpolated_coords

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Radius of Earth in meters
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
        return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    @staticmethod
    def _interpolate_coordinates(start_lat, start_lon, end_lat, end_lon, interval):
        distance = OSMCoordinates._haversine(start_lat, start_lon, end_lat, end_lon)
        num_points = int(distance // interval)
        return [(start_lat + (end_lat - start_lat) * i / num_points, start_lon + (end_lon - start_lon) * i / num_points) for i in range(1, num_points)]
# 103 is overlapping

#165-197 not overlapping
