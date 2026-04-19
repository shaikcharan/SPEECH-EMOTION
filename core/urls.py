from django.urls import path
from . import views

urlpatterns = [
    path("", views.predict_emotion, name="home"),
    path("predict/", views.predict_emotion, name="predict"),
]