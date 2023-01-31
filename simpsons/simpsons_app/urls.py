from django.urls import path

from .views import *

urlpatterns = [
    path('', index),
    path('name', name, name='name'),
    path('upload', upload, name='upload')
]