from django.urls import path

from optimiser import views

app_name = "optimiser"

urlpatterns = [
    path('calculate/',views.ModelOptimize.as_view()),
]
