from django.urls import path, include
from rest_framework.routers import DefaultRouter

from scenario_planner import views


router = DefaultRouter()
router.register('savedscenario', views.ScenarioViewSet)
router.register('scenario-metrics', views.ScenarioPlannerMetricsViewSet)
router.register('scenario-metrics-obj', views.ScenarioPlannerMetricsViewSetObject)
app_name = "scenario"

urlpatterns = [
    path('', include(router.urls)),
    path('download/', views.down),
    path('downloads/', views.ExampleViewSet.as_view({'post' : 'download'})),
    path('getExcel/', views.ExampleViewSet.as_view({'post' : 'getData'})),
    path('optimize/',views.ModelOptimize.as_view()),
    path('promo-simulate/',views.PromoSimulatorView.as_view({'get': 'get'})),
    path('promo-simulate-test/',views.PromoSimulatorViewTest.as_view())
]
