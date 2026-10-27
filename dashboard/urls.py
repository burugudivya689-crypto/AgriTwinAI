from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/status/", views.api_status, name="api_status"),
    path("api/options/", views.api_options, name="api_options"),
    path("api/predict/", views.api_predict, name="api_predict"),
    path("api/simulate/", views.api_simulate, name="api_simulate"),
    path("api/fields/", views.api_fields, name="api_fields"),
    path("api/metrics/", views.api_metrics, name="api_metrics"),
]
