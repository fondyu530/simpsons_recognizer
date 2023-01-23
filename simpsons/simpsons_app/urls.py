from django.urls import path

from .views import *

urlpatterns = [
    path('', index),
    path('name', name),
    path('upload', upload)
]