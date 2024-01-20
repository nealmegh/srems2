from django.urls import path
from . import views
from .views import display_interpolated_network, user_specific_interpolated_network, delete_interpolated_network, download_csv_view

urlpatterns = [
    path('', views.map_view, name='map_view'),
    path('define-boundary/', views.define_boundary_view, name='define_boundary'),
    path('display-heatmap/', views.display_heatmap_view, name='display_heatmap'),
    path('register/', views.register, name='register'),
    path('update_area/', views.update_interpolated_network_areas, name='up_area'),
    path('display-network/<int:network_id>/', display_interpolated_network, name='display_interpolated_network'),
    path('interpolated-network/', user_specific_interpolated_network, name='interpolated_network'),
    path('delete-record/<int:record_id>/', delete_interpolated_network, name='delete_record'),
    path('download-csv/', download_csv_view, name='download_csv'),

]
