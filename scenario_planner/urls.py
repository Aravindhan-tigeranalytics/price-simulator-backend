from django.urls import path, include
from rest_framework.routers import DefaultRouter

from scenario_planner import views


router = DefaultRouter()
# router.register('savedscenario', views.ScenarioViewSet)
# savescenario
router.register('savedscenario', views.SaveScenarioViewSet)
router.register('scenario-metrics', views.ScenarioPlannerMetricsViewSet)
router.register('scenario-metrics-obj', views.ScenarioPlannerMetricsViewSetObject)
router.register('promo-test', views.PromoSimulatorTestViewSet)
app_name = "scenario"

urlpatterns = [
    path('', include(router.urls)),
    path('download/', views.down),
    path('list-saved-promo/', views.LoadScenario.as_view({'get' : 'list',})),
    path('list-saved-promo/<int:id>/', views.LoadScenario.as_view({'get' : 'retrieve'})),
    path('list-saved-promo/<int:id>/<int:_id>/', views.LoadScenario.as_view({'get' : 'retrieve_pricing_promo'})),
    path('downloads/', views.ExampleViewSet.as_view({'post' : 'download'})),
    path('getExcel/', views.ExampleViewSet.as_view({'post' : 'getData'})),
    path('optimize/',views.ModelOptimize.as_view()),
    path('promo-simulate/',views.PromoSimulatorView.as_view({'get': 'get'})),
    path('save-promo/',views.savePromo),
    path('list-saved-promo-test/', views.LoadScenarioTest.as_view({'get' : 'list',})),
    path('list-saved-promo-test/<int:id>/', views.LoadScenarioTest.as_view({'get' : 'list',})),
    path('map-promo-pricing/', views.MapPricingPromo.as_view() ),
    # path('promo-simulate-test/',views.PromoSimulatorViewTest.as_view({'get': 'list','post' : 'post'}))
]
