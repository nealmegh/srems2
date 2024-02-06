import osmapi


class OSMCoordinatesOSMAPI:
    def __init__(self):
        self.api = osmapi.OsmApi()

    def get_coordinates_from_id(self, osm_id):
        try:
            way = self.api.WayGet(osm_id)
        except osmapi.errors.ElementNotFoundApiError:
            print(f"The OSM ID {osm_id} does not exist or is not a 'way'.")
            return None

        if not way:
            return None

        coordinates = []
        for node_id in way['nd']:
            node = self.api.NodeGet(node_id)
            if node:
                lat, lon = node['lat'], node['lon']
                coordinates.append((lat, lon))

        return coordinates

    def get_coordinates_for_ids(self, osm_ids):
        return {osm_id: self.get_coordinates_from_id(osm_id) for osm_id in osm_ids}

