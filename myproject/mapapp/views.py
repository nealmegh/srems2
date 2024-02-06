import time
from .forms import CustomUserCreationForm
from django.contrib.auth.models import User
from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import render, redirect, get_object_or_404
from .models import InterpolatedNetwork, UserProfile
from django.views.decorators.http import require_POST
from django.http import HttpResponse
from io import StringIO
from django.http import JsonResponse
from django.contrib.auth.views import LogoutView
from django.http import HttpResponseRedirect
import folium
from folium.plugins import Draw, HeatMap
import json
from django.contrib.gis.geos import Polygon, Point, fromstr, LineString, LinearRing
from django.contrib.gis.geos import GEOSGeometry
from django.core.serializers.json import DjangoJSONEncoder
from .models import NetworkData
import math
import numpy as np
from django.db.models import Min, Max
import branca.colormap as cm
from django.shortcuts import render
from .models import NetworkData
from .utils.road_coverage_checker import RoadCoverageChecker
from .utils.osmapi_coordinates import OSMCoordinatesOSMAPI
from .utils.osm_coordinates import OSMCoordinates
from .utils.interpolationByPciUsingRF import InterpolationByPciUsingRF
from .utils.interpolationUsingRF import InterpolationUsingRF
from .utils.interpolationByIDW import SignalStrengthInterpolatorIDW
from .utils.InterpolationByPCIusingIDW import IDWInterpolationByPCI
from .utils.InterpolationUsingKriging import InterpolationUsingKriging
from .utils.InterpolationByPCIUsingKriging import InterpolationByPCIUsingKriging
from .utils.interpolationUsingDT import InterpolationUsingDT
from .utils.interpolationByPciUsingDT import InterpolationByPciUsingDT
from .utils.interpolationUsingRFV2 import InterpolationUsingRFV2
from .utils.InterpolationUsingCNN import InterpolationUsingCNN
from .utils.InterpolationUsingRNN import InterpolationUsingRNN
from .utils.InterpolationUsingGAN import InterpolationUsingGAN
import branca.colormap as cm
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.wkt import loads
import pyproj
from shapely.ops import transform
from functools import partial
# from shapely.geometry import Point
from branca.colormap import linear
from django.contrib.gis.measure import D
from django.contrib.gis.db.models.functions import Distance
from django.db.models.expressions import RawSQL
from django.db import connection
import pdb
from datetime import datetime
from branca.element import Template, MacroElement
from django.contrib.auth.forms import UserCreationForm
import csv
from django.contrib import messages


# from geopy.distance import geodesic

def define_boundary_view(request):
    if request.user.is_authenticated:
        # Assuming you have a one-to-one relationship to a UserProfile model
        has_access = getattr(request.user.userprofile, 'access_right', False)
    else:
        has_access = False

    context = {
        'has_access': has_access,
    }
    return render(request, 'mapapp/define_boundary.html', context)


def update_user(request):

    user = get_object_or_404(User, username='abrar')
    profile, created = UserProfile.objects.get_or_create(user=user)
    profile.access_right = True  # or False, based on your logic
    profile.save()



    return HttpResponse('User updated successfully.')


def calculate_distance(point1, point2):
    R = 6371.0  # Radius of the Earth in km

    lat1 = math.radians(float(point1['latitude']))
    lon1 = math.radians(float(point1['longitude']))
    lat2 = math.radians(float(point2['latitude']))
    lon2 = math.radians(float(point2['longitude']))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in meters
    distance = R * c * 1000
    return distance


def process_network_data(data):
    # Initialize metrics
    numberOfCellId = len(set(item['cellId'] for item in data))
    LTECoverage = NSACoverage = totalDistance = 0
    LTEIntra = LTEInter = NSAInter = NSAIntra = 0
    signalStrengths = []
    channelBands = set()
    operator = 'NULL'

    prev_item = None
    for item in sorted(data, key=lambda x: x['gpsUpdateTime']):
        signalStrengths.append(int(item['signalStrength']))
        # signalStrengths.append(float(item['signalStrength']))
        channelBands.add(item['channelBands'])

        if prev_item:
            distance = calculate_distance(prev_item, item)  # Define this function as per your logic
            totalDistance += distance

            if prev_item['networkType'] == '1':
                LTECoverage += distance
                if prev_item['networkType'] != item['networkType']:
                    LTEInter += 1
                elif prev_item['cellId'] != item['cellId']:
                    LTEIntra += 1

            if prev_item['networkType'] == '2':
                NSACoverage += distance
                if prev_item['networkType'] != item['networkType']:
                    NSAInter += 1
                elif prev_item['cellId'] != item['cellId']:
                    NSAIntra += 1
        else:
            operator = item['operator_name']

        prev_item = item

    # Convert distances to kilometers
    LTECoverage = round(LTECoverage / 1000, 2)
    NSACoverage = round(NSACoverage / 1000, 2)
    totalDistance = round(totalDistance / 1000, 2)

    # NSAPercent = round((NSACoverage / totalDistance) * 100, 2) if totalDistance else 0
    NSAPercent = round((float(NSACoverage) / float(totalDistance)) * 100, 2) if totalDistance else 0
    SD = round(np.std(signalStrengths), 2)
    mean = round(np.mean(signalStrengths), 2)
    median = round(np.median(signalStrengths), 2)

    startDate = min(data, key=lambda x: x['gpsUpdateTime'])['gpsUpdateTime']
    endDate = max(data, key=lambda x: x['gpsUpdateTime'])['gpsUpdateTime']
    # startDate = datetime.fromtimestamp(startDate / 1000)  # Assuming Unix timestamp in milliseconds
    # endDate = datetime.fromtimestamp(endDate / 1000)

    startDate = datetime.fromtimestamp(float(startDate) / 1000).isoformat() # Convert to ISO format string
    endDate = datetime.fromtimestamp(float(endDate) / 1000).isoformat()


    return {
        'Operator Name': operator,
        'Start Date': startDate,
        'End Date': endDate,
        'Number Of Unique PCI': numberOfCellId,
        'LTE Coverage': LTECoverage,  # in kilometers
        'NSA Coverage': NSACoverage,  # in kilometers
        'Total Distance': totalDistance,  # in kilometers
        'NSA Percentage': NSAPercent,
        'LTE Intra': LTEIntra,
        'LTE Inter': LTEInter,
        'NSA Inter': NSAInter,
        'NSA Intra': NSAIntra,
        'Channel Bands': [int(band.strip('[]')) for band in channelBands],
        'SD': SD,
        'Mean': mean,
        'Median': median
    }


def area_calculate(geo_polygon):
    wgs84 = pyproj.CRS('EPSG:4326')  # WGS84 coordinate system
    utm = pyproj.CRS('EPSG:32630')  # Example UTM zone (you should choose the correct UTM zone for your area)
    project = partial(pyproj.transform, pyproj.Proj(wgs84), pyproj.Proj(utm))

    projected_polygon = transform(project, geo_polygon)

    area_meters = projected_polygon.area

    return area_meters / 1_000_000


def calculate_area_sq_km(coordinates):
    """
    Calculate the area of a polygon in square kilometers given its vertices' coordinates.
    """
    cartesian_coords = np.array([latlong_to_cartesian(lat, lon) for lat, lon in coordinates])

    x = cartesian_coords[:, 0]
    y = cartesian_coords[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    return area / 1e6


def latlong_to_cartesian(lon, lat):
    """
    Convert longitude and latitude to Cartesian coordinates.
    """
    # Constants for the WGS84 ellipsoid model of the Earth
    WGS84_RADIUS = 6378137.0  # Earth's radius in meters
    WGS84_ECCENTRICITY = 0.081819190842622

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    N = WGS84_RADIUS / np.sqrt(1 - WGS84_ECCENTRICITY ** 2 * np.sin(lat_rad) ** 2)
    X = N * np.cos(lat_rad) * np.cos(lon_rad)
    Y = N * np.cos(lat_rad) * np.sin(lon_rad)

    return X, Y


def map_view(request):
    return render(request, 'mapapp/map.html')


def display_interpolated_network(request, network_id):
    interpolated_network = get_object_or_404(InterpolatedNetwork, id=network_id, user=request.user)
    operator_name = interpolated_network.operator_name
    dataSelection = interpolated_network.data_selection
    coordinates_string = interpolated_network.coordinates
    interpolation_technique = interpolated_network.interpolation_technique
    coordinates_tuples = process_coordinates(coordinates_string)
    polygon = Polygon(coordinates_tuples)
    results = fetch_network_data(polygon, operator_name, dataSelection, coordinates_tuples)
    result_data = process_results(results, data_source='database')
    OD_matrics = process_network_data(result_data)
    shapely_polygon = ShapelyPolygon(coordinates_tuples)
    # area = area_calculate(shapely_polygon)
    # area = calculate_area_sq_km(coordinates_tuples)
    # print(coordinates_tuples)
    # print('area')
    # print(area_2)
    interpolated_results = interpolated_network.interpolated_data
    metrics = interpolated_network.metrics
    combined_data = combine_data(result_data, interpolated_results)
    folium_map = create_folium_map(shapely_polygon, combined_data, result_data)
    map_html = folium_map._repr_html_()
    chart_context = prepare_chart_context(interpolation_technique, metrics)
    summary_metrics = interpolated_network.summary_metrics

    return render(request, 'mapapp/display_heatmap3.html', {
        'map_html': map_html,
        'metrics': metrics,
        'chart_context': chart_context,
        'interpolation_technique': interpolation_technique,
        'area': interpolated_network.area,
        'summary_metrics': summary_metrics,
        'OD_matrics': OD_matrics
    })


def display_heatmap_view(request):
    if request.method != 'POST':
        return redirect('define_boundary')

    # operator_name, coordinates_string, dataSelection, interpolation_technique = get_post_data(request)
    try:
        # Attempt to extract data from POST request
        operator_name = request.POST['operatorName']
        coordinates_string = request.POST['coordinates']
        dataSelection = request.POST['dataSelection']
        interpolation_technique = request.POST['InterpolationTechnique']
        print(operator_name, coordinates_string, interpolation_technique)
    except KeyError:
        messages.error(request, 'Missing required data. Please Make sure to draw polygon and select the proper options')
        return redirect('define_boundary')

    coordinates_tuples = process_coordinates(coordinates_string)

    polygon = Polygon(coordinates_tuples)
    # results = fetch_network_data(polygon, operator_name, dataSelection)
    data_source = request.POST.get('data_source', 'database')

    if not request.FILES.get('data_file'):
        messages.warning(request, 'No Files uploaded')
        return redirect('define_boundary')
    if data_source == 'csv' and request.FILES.get('data_file'):
        csv_file = request.FILES['data_file']
        csv_data = csv.DictReader(StringIO(csv_file.read().decode('utf-8')))
        results = process_csv_data(list(csv_data), polygon, operator_name, dataSelection)
        print('done')
    else:
        results = fetch_network_data(polygon, operator_name, dataSelection, coordinates_tuples)
    if not results:
        messages.warning(request, 'No data found for the specified area and/or criteria.')
        return redirect('define_boundary')

    result_data = process_results(results, data_source)
    OD_matrics = process_network_data(result_data)
    shapely_polygon = ShapelyPolygon(coordinates_tuples)
    # area = area_calculate(shapely_polygon)
    area = calculate_area_sq_km(coordinates_tuples)
    interpolated_results = []
    metrics = []
    if interpolation_technique != 'OD':
        coords_map, all_coords = get_osm_coordinates(shapely_polygon, results, interpolation_technique, coordinates_tuples,
                                                 result_data, data_source)
        interpolated_results, metrics = perform_interpolation(interpolation_technique, results, all_coords, data_source)
    combined_data = combine_data(result_data, interpolated_results)
    folium_map = create_folium_map(shapely_polygon, combined_data, result_data)
    map_html = folium_map._repr_html_()
    chart_context = prepare_chart_context(interpolation_technique, metrics)
    summary_metrics = {}
    print(chart_context)
    if interpolation_technique != 'OD':
        summary_metrics = calculate_summary_metrics(metrics)
        if request.user.is_authenticated and data_source == 'database':
            InterpolatedNetwork.objects.create(
                user=request.user,
                operator_name=operator_name,
                coordinates=coordinates_string,
                data_selection=dataSelection,
                interpolation_technique=interpolation_technique,
                interpolated_data=interpolated_results,
                metrics=metrics,
                summary_metrics=summary_metrics,
                area=area
            )
    return render(request, 'mapapp/display_heatmap3.html', {
        'map_html': map_html,
        'metrics': metrics,
        'chart_context': chart_context,
        'interpolation_technique': interpolation_technique,
        'area': area,
        'summary_metrics': summary_metrics,
        'OD_matrics': OD_matrics
    })


# def display_heatmap_view(request):
#     if request.method == 'POST':
#         operator_name = request.POST.get('operatorName')
#         coordinates_string = request.POST.get('coordinates')
#         dataSelection = request.POST.get('dataSelection')
#         interpolation_technique = request.POST.get('InterpolationTechnique')
#         print(interpolation_technique)
#         print(dataSelection)
#         # print(coordinates_string)
#         coordinates_dicts = json.loads(coordinates_string)
#         coordinates_tuples = [(coord['lng'], coord['lat']) for coord in coordinates_dicts[0]]
#
#         if coordinates_tuples[0] != coordinates_tuples[-1]:
#             coordinates_tuples.append(coordinates_tuples[0])
#         # print(coordinates_tuples)
#
#         polygon = Polygon(coordinates_tuples)
#         results = []
#         if dataSelection == 'CD':
#             print('I am CD')
#             results = NetworkData.objects.filter(location__within=polygon, operatorName=operator_name)
#             print(len(results))
#         elif dataSelection == 'BD':
#             print('I am BD')
#             linear_ring = LinearRing(coordinates_tuples)
#             polygon_initiate = Polygon(linear_ring)
#             polygon_initiate.srid = 4326
#
#             # Transform the polygon to UTM and apply the buffer
#             polygon_initiate.transform(32630, clone=False)
#
#             buffer_distance_meters = 20
#             buffered_polygon = polygon_initiate.buffer(buffer_distance_meters)
#
#             # Transform the buffered polygon back to WGS84
#             ring_area = buffered_polygon.difference(polygon_initiate)
#             ring_area.transform(4326)
#
#             print(GEOSGeometry(buffered_polygon).wkt)
#
#             # Filtering NetworkData
#             results = NetworkData.objects.filter(location__intersects=ring_area, operatorName=operator_name)
#             print(len(results))
#         # pdb.set_trace()
#         result_data = [
#             {
#                 'latitude': obj.latitude,
#                 'longitude': obj.longitude,
#                 'cellId': obj.cellId,
#                 'cellId_TAC': obj.cellId_TAC,
#                 'cellId_PCI': obj.cellId_PCI,
#                 'signalStrength': obj.signalStrength,
#                 'gpsUpdateTime': obj.gpsUpdateTime,
#                 'channelBands': obj.channelBands,
#                 'networkType': obj.networkType,
#                 'operator_name': obj.operatorName
#
#             }
#             for obj in results
#         ]
#         OD_matrics = process_network_data(result_data)
#         print(OD_matrics)
#         print('Data Fetched from DB')
#         shapely_polygon = ShapelyPolygon(coordinates_tuples)
#         centroid = shapely_polygon.centroid
#         bounds = shapely_polygon.bounds
#
#         # checker = RoadCoverageChecker(coordinates_tuples, result_data)
#         # uncovered_road_ids = checker.get_uncovered_road_ids()
#         # print('fetched Road IDs')
#         # coords_obj = OSMCoordinates()
#         # coords_map = coords_obj.get_coordinates_for_ids(uncovered_road_ids)
#         # all_coords = [coord for sublist in coords_map.values() for coord in sublist if sublist]
#         wkt_string = polygon.wkt
#         shapelyPolygon = loads(wkt_string)
#         area = area_calculate(shapelyPolygon)
#
#         coords_obj = OSMCoordinates(shapelyPolygon, results)
#         coords_map = coords_obj.get_road_coordinates_without_data()
#         all_coords = [coord for sublist in coords_map.values() for coord in sublist if sublist]
#         # print(all_coords)
#
#         # print('got Uncovered Coordinates')
#         if interpolation_technique == 'RF_PCI':
#             print('RF_PCI')
#             print(len(all_coords))
#             CRF = InterpolationByPciUsingRF(results)
#             metrics = CRF.train_pci_models()
#             interpolated_results = CRF.predict(all_coords)
#         elif interpolation_technique == 'RF':
#             print('RF')
#             print(len(all_coords))
#             RF = InterpolationUsingRF(results)
#             metrics = RF.train_model()
#             interpolated_results = RF.predict(all_coords)
#         elif interpolation_technique == 'RF_2':
#             print('RF_2')
#             print(len(all_coords))
#             RF = InterpolationUsingRFV2(results)
#             metrics = RF.train_model()
#             interpolated_results = RF.predict(all_coords)
#         elif interpolation_technique == 'IDW':
#             print('IDW')
#             IDW = SignalStrengthInterpolatorIDW(results)
#             metrics = IDW.calculate_performance_metrics()
#             interpolated_results = IDW.predict(all_coords)
#             # print(interpolated_results)
#         elif interpolation_technique == 'IDW_PCI':
#             print('IDW_PCI')
#             IDW_PCI = IDWInterpolationByPCI(results)
#             metrics = IDW_PCI.calculate_pci_metrics()
#             interpolated_results = IDW_PCI.predict(all_coords)
#         elif interpolation_technique == 'OK':
#             print('OK')
#             OK = InterpolationUsingKriging(results)
#             metrics = OK.cross_validate()
#             interpolated_results = OK.predict(all_coords)
#         elif interpolation_technique == 'OK_PCI':
#             print('OK_PCI')
#             OK_PCI = InterpolationByPCIUsingKriging(results)
#             metrics = OK_PCI.cross_validate()
#             interpolated_results = OK_PCI.predict(all_coords)
#             # print(interpolated_results)
#         elif interpolation_technique == 'DT_PCI':
#             print('DT_PCI')
#             CDT = InterpolationByPciUsingDT(results)
#             metrics = CDT.train_pci_models()  # Get RMSE, R2, etc.
#             interpolated_results = CDT.predict(all_coords)
#         elif interpolation_technique == 'DT':  # Change this to your actual other technique name
#             print('DT')
#             DT = InterpolationUsingDT(results)
#             metrics = DT.train_model()
#             interpolated_results = DT.predict(all_coords)
#         else:
#             print('No interpolation')
#             interpolated_results = []
#             metrics = []
#             chart_context = []
#         # Assuming predict method returns interpolated data
#         print('Got Interpolated Values')
#         # combined_data = list(result_data) + interpolated_results
#         combined_data = []
#
#         combined_data.extend(result_data)
#
#         for lat, lon, signal in interpolated_results:
#             combined_data.append({'latitude': lat, 'longitude': lon, 'signalStrength': signal})
#         # combined_data
#         # polygon = Polygon(coordinates_tuples)
#         # filtered_data = [data for data in combined_data
#         #                  if polygon.contains(Point(data['longitude'], data['latitude']))]
#         # filtered_data = [data for data in combined_data
#         #                  if -140 <= data['signalStrength'] <= -60 and
#         #                  polygon.contains(Point(data['longitude'], data['latitude']))]
#
#         # print(filtered_data)
#         # breakpoint()
#         filtered_data = combined_data
#         folium_map = folium.Map(location=[centroid.y, centroid.x])
#         # Find the max and min signal strength in filtered_data
#         max_signal = max(item['signalStrength'] for item in filtered_data)
#         min_signal = min(item['signalStrength'] for item in filtered_data)
#         print(max_signal, min_signal)
#         # Update vmin and vmax based on the signal strengths
#         vmin = min_signal if min_signal > -140 else -140
#         vmax = max_signal if max_signal < -60 else -60
#         color_scale = cm.LinearColormap(colors=['red', 'yellow', 'green'], vmin=vmin, vmax=vmax)
#         print(vmin, vmax)
#         hotlines = []
#         current_line = []
#
#         for key, ref in enumerate(filtered_data):
#             if key > 0:
#                 prev_ref = filtered_data[key - 1]
#                 distance = calculate_distance(prev_ref, ref)
#
#                 if distance > 200:
#                     if current_line:
#                         hotlines.append(current_line)
#                     current_line = [ref]
#
#                 current_line.append(ref)
#             else:
#                 current_line.append(ref)
#
#         if current_line:
#             hotlines.append(current_line)
#         print('hotlines created')
#         print(len(hotlines))
#         for line in hotlines:
#             # print('hotlines painting ....')
#             for i in range(len(line) - 1):
#                 start_point = line[i]
#                 end_point = line[i + 1]
#                 start_color = color_scale(start_point['signalStrength'])
#                 end_color = color_scale(end_point['signalStrength'])
#                 gradient = {
#                     0: start_color,
#                     1: end_color
#                 }
#                 folium.PolyLine(
#                     locations=[[start_point['latitude'], start_point['longitude']],
#                                [end_point['latitude'], end_point['longitude']]],
#                     color=start_color,
#                     gradient=gradient,
#                     opacity=0.4,
#                     weight=8
#                 ).add_to(folium_map)
#
#             for point in line:
#                 is_original = point in result_data
#                 point_color = color_scale(point['signalStrength'])
#                 marker_offset = 0.00001
#
#                 if is_original:
#                     folium.CircleMarker(
#                         location=[point['latitude'], point['longitude']],
#                         radius=3,
#                         color=point_color,
#                         fill=True,
#                         fill_color=point_color
#                     ).add_to(folium_map)
#                 else:
#                     folium.Rectangle(
#                         bounds=[[point['latitude'] - marker_offset, point['longitude'] - marker_offset],
#                                 [point['latitude'] + marker_offset, point['longitude'] + marker_offset]],
#                         color=point_color,
#                         fill=True,
#                         fill_color=point_color
#                     ).add_to(folium_map)
#         legend_html = f"""
#                 <div style="position: fixed;
#                             top: 50px; left: 50px; width: 200px; height: 120px;
#                             border:2px solid grey; z-index:9999; font-size:14px;
#                             background-color: white;
#                             opacity: 0.8;">
#                     <p style="text-align:center;"><b>Signal Strength (dBm)</b></p>
#                     <p style="margin-left:10px;">{vmin} dBm <span style="margin-left:10px;background-color:red;width:10px;height:10px;display:inline-block;"></span></p>
#                     <p style="margin-left:10px;">{(vmin + vmax) / 2} dBm <span style="margin-left:10px;background-color:yellow;width:10px;height:10px;display:inline-block;"></span></p>
#                     <p style="margin-left:10px;">{vmax} dBm <span style="margin-left:10px;background-color:green;width:10px;height:10px;display:inline-block;"></span></p>
#                 </div>
#                 """
#
#         folium_map.get_root().html.add_child(folium.Element(legend_html))
#
#         folium_map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
#
#         map_html = folium_map._repr_html_()
#         summary_metrics = []
#         print('Can see on the map.')
#         if interpolation_technique == 'RF_PCI' or interpolation_technique == 'IDW_PCI' or interpolation_technique == 'OK_PCI' or interpolation_technique == 'DT_PCI':
#             pci_labels = [metric['PCI'] for metric in metrics]
#             data_points = [metric['Data_Points'] for metric in metrics]
#             rmse_values = [metric['RMSE'] for metric in metrics]
#             r2_values = [metric['R2'] for metric in metrics]
#             # Assuming data_points, rmse_values, r2_values are lists of equal length
#             combined_chart_data = list(zip(data_points, rmse_values, r2_values))
#             sorted_combined_chart_data = sorted(combined_chart_data,
#                                                 key=lambda x: x[0])  # Sort by the number of data points
#
#             # Unzip the sorted data
#             sorted_data_points, sorted_rmse_values, sorted_r2_values = zip(*sorted_combined_chart_data)
#
#             chart_context = {
#                 'data_points': json.dumps(sorted_data_points),
#                 'rmse_values': json.dumps(sorted_rmse_values),
#                 'r2_values': json.dumps(sorted_r2_values),
#             }
#             # Calculating the sum and average of the metrics
#             # total_data_points = sum(metric['Data_Points'] for metric in metrics)
#             # total_duration = sum(metric['Training_Time'] for metric in metrics)
#             # avg_rmse_cv = sum(metric['Cross_Validated_RMSE'] for metric in metrics) / len(metrics)
#             # avg_mae = sum(metric['MAE'] for metric in metrics) / len(metrics)
#             # avg_rmse = sum(metric['RMSE'] for metric in metrics) / len(metrics)
#             # avg_mbe = sum(metric['MBE'] for metric in metrics) / len(metrics)
#             # avg_r2 = sum(metric['R2'] for metric in metrics) / len(metrics)
#             # avg_std_dev = sum(metric['Standard_Deviation'] for metric in metrics) / len(metrics)
#
#             # Update the context with the new calculated values
#             # summary_metrics = {
#             #     'Total Data Points': total_data_points,
#             #     'Total Training Time (seconds)': total_duration,
#             #     'Average Cross-Validated RMSE': avg_rmse_cv,
#             #     'Average MAE': avg_mae,
#             #     'Average RMSE': avg_rmse,
#             #     'Average MBE': avg_mbe,
#             #     'Average R2': avg_r2,
#             #     'Average Standard Deviation of Errors': avg_std_dev
#             # }
#         elif interpolation_technique == 'RF' or interpolation_technique == 'RF_2' or interpolation_technique == 'IDW' or interpolation_technique == 'OK' or interpolation_technique == 'DT':
#             chart_context = {
#                 'data_points': json.dumps(metrics['Total Data Points']),
#                 'duration': json.dumps(metrics['Training Duration (seconds)']),
#                 'cross_validated_rmse': json.dumps(metrics['Cross-validated RMSE']),
#                 'mae_values': json.dumps(metrics['MAE']),
#                 'rmse_values': json.dumps(metrics['RMSE']),
#                 'mbe_values': json.dumps(metrics['MBE']),
#                 'r2_values': json.dumps(metrics['R2']),
#                 'std_dev_values': json.dumps(metrics['Standard Deviation of Errors'])
#             }
#             # print(chart_context)
#
#         return render(request, 'mapapp/display_heatmap3.html', {
#             'map_html': map_html,
#             'metrics': metrics,
#             'chart_context': chart_context,
#             'interpolation_technique': interpolation_technique,
#             'area': area,
#             'summary_metrics': summary_metrics,
#             'OD_matrics': OD_matrics
#         })
#
#     return redirect('define_boundary')


def get_post_data(request):
    operator_name = request.POST.get('operatorName')
    coordinates_string = request.POST.get('coordinates')
    dataSelection = request.POST.get('dataSelection')
    interpolation_technique = request.POST.get('InterpolationTechnique')
    return operator_name, coordinates_string, dataSelection, interpolation_technique


def process_coordinates(coordinates_string):
    coordinates_dicts = json.loads(coordinates_string)
    coordinates_tuples = [(coord['lng'], coord['lat']) for coord in coordinates_dicts[0]]
    if coordinates_tuples[0] != coordinates_tuples[-1]:
        coordinates_tuples.append(coordinates_tuples[0])
    return coordinates_tuples


def get_osm_coordinates(shapely_polygon, results, interpolation_technique, coordinates_tuples, result_data, data_source):
    start_time = time.time()
    coords_obj = OSMCoordinates(shapely_polygon, results, data_source)
    coords_map = coords_obj.get_road_coordinates_without_data()
    all_coords = [coord for sublist in coords_map.values() for coord in sublist if sublist]

    # different
    # checker = RoadCoverageChecker(coordinates_tuples, result_data)
    # uncovered_road_ids = checker.get_uncovered_road_ids()
    # coords_obj = OSMCoordinatesOSMAPI()
    # coords_map = coords_obj.get_coordinates_for_ids(uncovered_road_ids)
    # all_coords = [coord for sublist in coords_map.values() for coord in sublist if sublist]

    end_time = time.time()
    exec_time = end_time - start_time
    print('execution time')
    print(exec_time)
    return coords_map, all_coords


def combine_data(result_data, interpolated_results):
    combined_data = result_data.copy()
    for lat, lon, signal in interpolated_results:
        combined_data.append({'latitude': lat, 'longitude': lon, 'signalStrength': signal})
    return combined_data


# def fetch_network_data(polygon, operator_name, dataSelection, coordinates_tuples):
#     if dataSelection == 'CD':
#         return NetworkData.objects.filter(location__within=polygon, operatorName=operator_name)
#     elif dataSelection == 'BD':
#         linear_ring = LinearRing(coordinates_tuples)
#         polygon_initiate = Polygon(linear_ring)
#         polygon_initiate.srid = 4326
#         polygon_initiate.transform(32630, clone=False)
#         buffer_distance_meters = 50
#         buffered_polygon = polygon_initiate.buffer(buffer_distance_meters)
#         ring_area = buffered_polygon.difference(polygon_initiate)
#         ring_area.transform(4326)
#         return NetworkData.objects.filter(location__intersects=ring_area, operatorName=operator_name)
#     return []
def fetch_network_data(polygon, operator_name, dataSelection, coordinates_tuples):
    results = []

    if dataSelection in ['CD', 'BD']:
        if dataSelection == 'CD':
            results.extend(NetworkData.objects.filter(location__within=polygon, operatorName=operator_name))

        linear_ring = LinearRing(coordinates_tuples)
        polygon_initiate = Polygon(linear_ring)
        polygon_initiate.srid = 4326
        polygon_initiate.transform(32630, clone=False)
        buffer_distance_meters = 50
        buffered_polygon = polygon_initiate.buffer(buffer_distance_meters)
        ring_area = buffered_polygon.difference(polygon_initiate)
        ring_area.transform(4326)
        results.extend(NetworkData.objects.filter(location__intersects=ring_area, operatorName=operator_name))

    return results
# def buffer_and_transform_polygon(coordinates_tuples, buffer_distance_meters=50):
#     """Buffer the polygon and transform its coordinate system."""
#     linear_ring = LinearRing(coordinates_tuples)
#     polygon = Polygon(linear_ring)
#     polygon.srid = 4326
#     polygon.transform(32630)
#     buffered_polygon = polygon.buffer(buffer_distance_meters)
#     buffered_polygon.transform(4326)
#     return buffered_polygon
#
#
# def fetch_network_data(request, polygon, operator_name, dataSelection, coordinates_tuples):
#     results = []
#
#     try:
#         if dataSelection == 'CD':
#             query_results = NetworkData.objects.filter(location__within=polygon, operatorName=operator_name)
#             if query_results.exists():
#                 results.extend(query_results)
#
#         if dataSelection in ['CD', 'BD']:
#             buffered_polygon = buffer_and_transform_polygon(coordinates_tuples)
#             ring_area = buffered_polygon.difference(polygon)
#             query_results = NetworkData.objects.filter(location__intersects=ring_area, operatorName=operator_name)
#             if query_results.exists():
#                 results.extend(query_results)
#
#         # Check if results list is empty after fetching data
#         if not results:
#             messages.warning(request, 'No data found for the specified area and criteria.')
#             return redirect('define_boundary')
#
#     except Exception as e:
#         # General error handling (e.g., issues with the database, spatial operations, etc.)
#         messages.error(request, f'An error occurred while fetching data: {e}')
#         return redirect('define_boundary')
#
#     return results


def perform_interpolation(interpolation_technique, results, all_coords, data_source):
    interpolated_results = []
    metrics = []

    if interpolation_technique == 'RF_PCI':
        print('RF_PCI')
        print(len(all_coords))
        CRF = InterpolationByPciUsingRF(results, data_source)
        metrics = CRF.train_pci_models()
        interpolated_results, model_prediction_times = CRF.predict(all_coords)
        for metric in metrics:
            pci = metric['PCI']
            training_time = metric.get('Training_Time', 0)
            prediction_time = model_prediction_times.get(pci, 0)
            # Calculate total Model Execution Time
            metric['Model_Execution_Time'] = training_time + prediction_time
            metric['Model_Prediction_Time'] = prediction_time

    elif interpolation_technique == 'RF':
        print('RF')
        # print(len(all_coords))
        RF = InterpolationUsingRF(results, data_source)
        metrics = RF.train_model()
        interpolated_results, prediction_time = RF.predict(all_coords)
        metrics['Prediction Time'] = prediction_time
        metrics['Total Execution time'] = metrics['Training Duration (seconds)'] + prediction_time
    elif interpolation_technique == 'RF_2':
        print('RF_2')
        # print(len(all_coords))
        RF = InterpolationUsingRFV2(results, data_source)
        metrics = RF.train_model()
        interpolated_results, prediction_time = RF.predict(all_coords)
        metrics['Prediction Time'] = prediction_time
        metrics['Total Execution time'] = metrics['Training Duration (seconds)'] + prediction_time
    elif interpolation_technique == 'CNN':
        print('CNN')
        # print(len(all_coords))
        CNN = InterpolationUsingCNN(results)
        metrics = CNN.train_model()
        interpolated_results = CNN.predict(all_coords)
    elif interpolation_technique == 'RNN':
        print('RNN')
        # print(len(all_coords))
        RNN = InterpolationUsingRNN(results)
        metrics = RNN.train_model()
        interpolated_results = RNN.predict(all_coords)
    elif interpolation_technique == 'GAN':
        print('GAN')
        # print(len(all_coords))
        GAN = InterpolationUsingGAN(results, data_source)
        metrics = GAN.train_model()
        interpolated_results, prediction_time = GAN.predict(all_coords)
        metrics['Prediction Time'] = prediction_time
        metrics['Total Execution time'] = metrics['Training Duration (seconds)'] + prediction_time
    elif interpolation_technique == 'IDW':
        print('IDW')
        IDW = SignalStrengthInterpolatorIDW(results, data_source)
        metrics = IDW.calculate_performance_metrics()
        interpolated_results, prediction_time = IDW.predict(all_coords)
        metrics['Prediction Time'] = prediction_time
        metrics['Total Execution time'] = metrics['Training Duration (seconds)'] + prediction_time
        # print(interpolated_results)
    elif interpolation_technique == 'IDW_PCI':
        print('IDW_PCI')
        IDW_PCI = IDWInterpolationByPCI(results, data_source)
        metrics = IDW_PCI.calculate_pci_metrics()
        interpolated_results, model_prediction_times = IDW_PCI.predict(all_coords)
        for metric in metrics:
            pci = metric['PCI']
            training_time = metric.get('Training_Time', 0)
            prediction_time = model_prediction_times.get(pci, 0)
            # Calculate total Model Execution Time
            metric['Model_Execution_Time'] = training_time + prediction_time
            metric['Model_Prediction_Time'] = prediction_time
    elif interpolation_technique == 'OK':
        print('OK')
        OK = InterpolationUsingKriging(results, data_source)
        metrics = OK.cross_validate()
        interpolated_results, prediction_time = OK.predict(all_coords)
        metrics['Prediction Time'] = prediction_time
        metrics['Total Execution time'] = metrics['Training Duration (seconds)'] + prediction_time
    elif interpolation_technique == 'OK_PCI':
        print('OK_PCI')
        OK_PCI = InterpolationByPCIUsingKriging(results, data_source)
        metrics = OK_PCI.cross_validate()
        interpolated_results, model_prediction_times = OK_PCI.predict(all_coords)
        for metric in metrics:
            pci = metric['PCI']
            training_time = metric.get('Training_Time', 0)
            prediction_time = model_prediction_times.get(pci, 0)
            # Calculate total Model Execution Time
            metric['Model_Execution_Time'] = training_time + prediction_time
            metric['Model_Prediction_Time'] = prediction_time
        # print(interpolated_results)
    elif interpolation_technique == 'DT_PCI':
        print('DT_PCI')
        CDT = InterpolationByPciUsingDT(results, data_source)
        metrics = CDT.train_pci_models()
        interpolated_results, model_prediction_times = CDT.predict(all_coords)
        for metric in metrics:
            pci = metric['PCI']
            training_time = metric.get('Training_Time', 0)
            prediction_time = model_prediction_times.get(pci, 0)
            # Calculate total Model Execution Time
            metric['Model_Execution_Time'] = training_time + prediction_time
            metric['Model_Prediction_Time'] = prediction_time
    elif interpolation_technique == 'DT':
        print('DT')
        DT = InterpolationUsingDT(results, data_source)
        metrics = DT.train_model()
        interpolated_results, prediction_time = DT.predict(all_coords)
        metrics['Prediction Time'] = prediction_time
        metrics['Total Execution time'] = metrics['Training Duration (seconds)'] + prediction_time
    else:
        print('No interpolation')
        interpolated_results = []
        metrics = []
        chart_context = []

    return interpolated_results, metrics


def calculate_summary_metrics(metrics):
    summary_metrics = {}

    if isinstance(metrics, list):
        total_data_points = sum(metric.get('Data_Points', 0) for metric in metrics)
        total_duration = sum(metric.get('Training_Time', 0) for metric in metrics)
        avg_rmse = sum(metric.get('RMSE', 0) for metric in metrics) / len(metrics)
        avg_r2 = sum(metric.get('R2', 0) for metric in metrics) / len(metrics)

        summary_metrics = {
            'Total Data Points': total_data_points,
            'Total Training Time (seconds)': total_duration,
            'Average RMSE': avg_rmse,
            'Average R2': avg_r2
        }
    elif isinstance(metrics, dict):  # Assuming metrics is a dictionary
        # Example calculations for a single dictionary
        summary_metrics = {
            'Total Data Points': metrics.get('Total Data Points', 0),
            'Training Duration (seconds)': metrics.get('Training Duration (seconds)', 0),
            'Cross-validated RMSE': metrics.get('Cross-validated RMSE', 0),
            'MAE': metrics.get('MAE', 0),
            'RMSE': metrics.get('RMSE', 0),
            'MBE': metrics.get('MBE', 0),
            'R2': metrics.get('R2', 0),
            'Standard Deviation of Errors': metrics.get('Standard Deviation of Errors', 0)
        }

    return summary_metrics


def process_results(results, data_source):
    """
    Process the results fetched from the database.

    Args:
    results (QuerySet or list): The results fetched from the database.

    Returns:
    list: A list of dictionaries where each dictionary represents processed data.
    """
    processed_data = []
    if data_source == 'csv':
        for obj in results:
            data_item = {
                'latitude': obj['latitude'],
                'longitude': obj['longitude'],
                'cellId': obj['cellId'],
                'cellId_TAC': obj['cellId_TAC'],
                'cellId_PCI': obj['cellId_PCI'],
                'signalStrength': obj['signalStrength'],
                'gpsUpdateTime': obj['gpsUpdateTime'],
                'channelBands': obj['channelBands'],
                'networkType': obj['networkType'],
                'operator_name': obj['operatorName']
            }
            processed_data.append(data_item)
    if data_source == 'database':
        for obj in results:
            data_item = {
                'latitude': obj.latitude,
                'longitude': obj.longitude,
                'cellId': obj.cellId,
                'cellId_TAC': obj.cellId_TAC,
                'cellId_PCI': obj.cellId_PCI,
                'signalStrength': obj.signalStrength,
                'gpsUpdateTime': obj.gpsUpdateTime,
                'channelBands': obj.channelBands,
                'networkType': obj.networkType,
                'operator_name': obj.operatorName
            }
            processed_data.append(data_item)

    return processed_data


def prepare_chart_context(interpolation_technique, metrics):
    chart_context = {}

    if interpolation_technique in ['RF_PCI', 'IDW_PCI', 'OK_PCI', 'DT_PCI']:
        # Assuming metrics is a list of dictionaries for these techniques
        pci_labels = [metric['PCI'] for metric in metrics]
        data_points = [metric['Data_Points'] for metric in metrics]
        rmse_values = [metric['RMSE'] for metric in metrics]
        r2_values = [metric['R2'] for metric in metrics]

        chart_context = {
            'pci_labels': json.dumps(pci_labels),
            'data_points': json.dumps(data_points),
            'rmse_values': json.dumps(rmse_values),
            'r2_values': json.dumps(r2_values)
        }
    elif interpolation_technique in ['RF', 'RF_2', 'IDW', 'OK', 'DT', 'CNN', 'GAN', 'RNN']:
        # Assuming metrics is a single dictionary for these techniques
        chart_context = {
            'data_points': json.dumps(metrics.get('Total Data Points', [])),
            'duration': json.dumps(metrics.get('Training Duration (seconds)', [])),
            'prediction_duration': json.dumps(metrics.get('Prediction Time', [])),
            'execution_duration': json.dumps(metrics.get('Total Execution time', [])),
            'cross_validated_rmse': json.dumps(metrics.get('Cross-validated RMSE', [])),
            'mae_values': json.dumps(metrics.get('MAE', [])),
            'rmse_values': json.dumps(metrics.get('RMSE', [])),
            'mbe_values': json.dumps(metrics.get('MBE', [])),
            'r2_values': json.dumps(metrics.get('R2', [])),
            'std_dev_values': json.dumps(metrics.get('Standard Deviation of Errors', []))
        }

    return chart_context


def create_folium_map(shapely_polygon, combined_data, result_data):
    centroid = shapely_polygon.centroid
    bounds = shapely_polygon.bounds
    folium_map = folium.Map(location=[centroid.y, centroid.x])
    signalStrengths = [int(item['signalStrength']) for item in combined_data]
    max_signal = max(signalStrengths)
    min_signal = min(signalStrengths)
    # Define color scale
    # max_signal = max(int(item['signalStrength']) for item in combined_data)
    # min_signal = min(int(item['signalStrength']) for item in combined_data)
    # vmin = min_signal if min_signal > -140 else -140
    # vmax = max_signal if max_signal < -60 else -60
    vmax = -80
    vmin = -120
    color_scale = cm.LinearColormap(colors=['red', 'yellow', 'green'], vmin=vmin, vmax=vmax)

    # Create hotlines and markers
    hotlines = []
    current_line = []
    marker_offset = 0.00001  # Adjust as needed for visual clarity

    for key, ref in enumerate(combined_data):
        is_original = ref in result_data

        # Add markers
        point_color = color_scale(int(ref['signalStrength']))
        if is_original:
            folium.CircleMarker(
                location=[float(ref['latitude']), float(ref['longitude'])],
                radius=3,
                color=point_color,
                fill=True,
                fill_color=point_color
            ).add_to(folium_map)
        else:
            folium.Rectangle(
                bounds=[[float(ref['latitude']) - marker_offset, float(ref['longitude']) - marker_offset],
                        [float(ref['latitude']) + marker_offset, float(ref['longitude']) + marker_offset]],
                color=point_color,
                fill=True,
                fill_color=point_color
            ).add_to(folium_map)

        # Add to hotlines
        if key > 0:
            prev_ref = combined_data[key - 1]
            distance = calculate_distance(prev_ref, ref)
            if distance > 200:
                if current_line:
                    hotlines.append(current_line)
                current_line = [ref]
        current_line.append(ref)
    if current_line:
        hotlines.append(current_line)

    # Add hotlines to the map
    for line in hotlines:
        for i in range(len(line) - 1):
            start_point = line[i]
            end_point = line[i + 1]
            start_signal_strength = int(start_point['signalStrength'])
            end_signal_strength = int(end_point['signalStrength'])

            start_color = color_scale(start_signal_strength)
            end_color = color_scale(end_signal_strength)
            # start_color = color_scale(start_point['signalStrength'])
            # end_color = color_scale(end_point['signalStrength'])
            start_latitude = float(start_point['latitude'])
            start_longitude = float(start_point['longitude'])
            end_latitude = float(end_point['latitude'])
            end_longitude = float(end_point['longitude'])

            gradient = {0: start_color, 1: end_color}
            folium.PolyLine(
                locations=[
                    [start_latitude, start_longitude],
                    [end_latitude, end_longitude]
                ],
                color=start_color,
                gradient=gradient,
                opacity=0.4,
                weight=8
            ).add_to(folium_map)
    legend_html = f"""
        <div style="position: fixed; 
                    top: 50px; left: 50px; width: 150px; height: 150px; 
                    border:2px solid grey; z-index:9999; font-size:18px;
                    background-color: white;
                    opacity: 0.8;">
            <p style="text-align:center;"><b>Signal Strength</b></p>
            <p style="margin-left:10px;">{vmin} dBm <span style="margin-left:10px;background-color:red;width:10px;height:10px;display:inline-block;"></span></p>
            <p style="margin-left:10px;">{(vmin + vmax) / 2} dBm <span style="margin-left:10px;background-color:yellow;width:10px;height:10px;display:inline-block;"></span></p>
            <p style="margin-left:10px;">{vmax} dBm <span style="margin-left:10px;background-color:green;width:10px;height:10px;display:inline-block;"></span></p>
        </div>
        """

    folium_map.get_root().html.add_child(folium.Element(legend_html))

    folium_map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    return folium_map


def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)  # Save the user form data to the user object
            user.first_name = form.cleaned_data['first_name']
            user.last_name = form.cleaned_data['last_name']
            user.email = form.cleaned_data['email']
            user.save()  # Now, save the user object with the additional details to the database
            return redirect('login')
    else:
        form = CustomUserCreationForm()
    return render(request, 'register.html', {'form': form})


def user_specific_interpolated_network(request):
    if not request.user.is_authenticated:
        # Redirect to login page or handle it as you wish
        return redirect('login')

    # Fetch data specific to the logged-in user
    # user_data = InterpolatedNetwork.objects.filter(user=request.user)
    user_data = InterpolatedNetwork.objects.filter(user=request.user).order_by('id')

    return render(request, 'mapapp/interpolated_network.html', {'user_data': user_data})


@require_POST
def delete_interpolated_network(request, record_id):
    record = get_object_or_404(InterpolatedNetwork, id=record_id, user=request.user)
    record.delete()
    return redirect('interpolated_network')


def download_csv_view(request):
    if request.method == 'POST':
        operator_name = request.POST.get('operatorName')
        dataSelection = request.POST.get('dataSelection')
        coordinates_string = request.POST.get('coordinates')
        coordinates_tuples = process_coordinates(coordinates_string)
        polygon = Polygon(coordinates_tuples)
        results = fetch_network_data(polygon, operator_name, dataSelection, coordinates_tuples)
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="network_data.csv"'

        writer = csv.writer(response)
        # Write CSV headers based on the NetworkData model fields
        writer.writerow(['Accuracy', 'ASU Level', 'Cell ID', 'Channel Bands', 'Channel Bandwidth', 'Channel Quality',
                         'GPS Update Time', 'Latitude', 'Longitude', 'Network ID', 'Network Type',
                         'Network Update Time', 'Operator Name', 'Signal Strength', 'Cell ID PCI', 'Cell ID TAC'])

        # Write data rows
        for item in results:
            writer.writerow([item.accuracy, item.asuLevel, item.cellId, item.channelBands, item.channelBandwidth,
                             item.channelQuality, item.gpsUpdateTime, item.latitude, item.longitude, item.networkId,
                             item.networkType, item.networkUpdateTime, item.operatorName, item.signalStrength,
                             item.cellId_PCI, item.cellId_TAC])

        return response
    else:
        # Redirect or handle GET request differently
        return redirect('define_boundary')  # Adjust the redirect as needed


def process_csv_data(csv_data, polygon, operator_name, data_selection):
    processed_data = []

    # Function to transform coordinates
    project = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:4326'),  # source coordinate system
        pyproj.Proj(init='epsg:32630')  # destination coordinate system
    )


    if data_selection == 'CD':
        for row in csv_data:
            point = Point(float(row['longitude']), float(row['latitude']))
            if point.within(polygon) and row['operatorName'] == operator_name:
                processed_data.append(row)

    elif data_selection == 'BD':
        linear_ring = LinearRing(polygon.exterior.coords)
        polygon_initiate = Polygon(linear_ring)
        polygon_initiate = transform(project, polygon_initiate)
        buffer_distance_meters = 20
        buffered_polygon = polygon_initiate.buffer(buffer_distance_meters)
        ring_area = buffered_polygon.difference(polygon_initiate)
        ring_area = transform(lambda x, y: (y, x), ring_area)  # Inverting coordinates to match lat/lon

        for row in csv_data:
            point = Point(float(row['longitude']), float(row['latitude']))
            if point.intersects(ring_area) and row['operatorName'] == operator_name:
                processed_data.append(row)

    return processed_data


def update_interpolated_network_areas(request):
    try:
        networks = InterpolatedNetwork.objects.all()
        for network in networks:
            # Parse the JSON string in the coordinates field
            coordinates_string = network.coordinates
            coordinates_tuples = process_coordinates(coordinates_string)
            # Flatten the list if it's nested and convert it to the required format
            # coordinates = [(coord['lng'], coord['lat']) for sublist in coordinates_json for coord in sublist]

            # Calculate the area using the coordinates
            area_sq_km = calculate_area_sq_km(coordinates_tuples)
            print(str(network.id) + ' write ' + str(area_sq_km))
            # Update the area field of the network
            network.area = area_sq_km
            network.save()

        return HttpResponse("Area updated for all interpolated networks.", status=200)

    except ObjectDoesNotExist:
        return HttpResponse("No records found.", status=404)
    except Exception as e:
        return HttpResponse(f"An error occurred: {e}", status=500)
