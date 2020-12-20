from django.urls import path, include
from rest_framework.routers import DefaultRouter

from scenario_planner import views


router = DefaultRouter()
router.register('savedscenario', views.ScenarioViewSet)
app_name = "scenario"

urlpatterns = [
    path('', include(router.urls)),
    path('download/', views.down)
]
