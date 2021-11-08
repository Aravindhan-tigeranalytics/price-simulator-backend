from django.urls import path, include
from rest_framework.routers import DefaultRouter

from scenario_planner import views


router = DefaultRouter()
# router.register('savedscenario', views.ScenarioViewSet)
# savescenario
router.register('savescenario', views.SaveScenarioViewSet)
router.register('scenario-metrics', views.ScenarioPlannerMetricsViewSet)
router.register('scenario-metrics-obj', views.ScenarioPlannerMetricsViewSetObject)
router.register('promo-test', views.PromoSimulatorTestViewSet)
app_name = "scenario"

urlpatterns = [
    path('', include(router.urls)),
    path('download/', views.down),
    path('save/',views.SavePromo.as_view({'get':'get','post' : 'post'})),
    path('update/',views.UpdatePromo.as_view({'get':'get','post' : 'post'})),
    path('list-saved-promo/', views.LoadScenario.as_view({'get' : 'list',})),
    path('list-saved-promo/<int:id>/', views.LoadScenario.as_view({'get' : 'retrieve'})),
    path('list-saved-promo/<int:id>/<int:_id>/', views.LoadScenario.as_view({'get' : 'retrieve_pricing_promo'})),
    path('downloads/', views.ExampleViewSet.as_view({'post' : 'download'})),
    path('getExcel/', views.ExampleViewSet.as_view({'post' : 'getData'})),
    path('optimize/',views.ModelOptimize.as_view()),
    path('promo-simulate/',views.PromoSimulatorView.as_view({'get': 'get'})),
    path('promo-simulate-file-upload/',views.PromoSimulatorUploadView.as_view({'get': 'get'})),
    path('pricing-weekly-upload/',views.pricing_weely_upload),
    path('promo-download/',views.PromoSimulatorView.as_view({'get': 'get'})),
    path('pricing-download/',views.pricing_download),
    path('compare-scenario-download/',views.CompareScenarioExcelDownloadView.as_view()),
    path('compare-scenario-download-pricing/',views.CompareScenarioExcelDownloadView.as_view()),
    path('weekly-input-template-download/',views.WeeklyInputTemplateDownload.as_view()),
    path('save-promo/',views.savePromo),
    path('list-saved-promo-test/', views.LoadScenarioTest.as_view({'get' : 'list',})),
    path('list-saved-promo-test/<int:id>/', views.LoadScenarioTest.as_view({'get' : 'list',})),
    path('map-promo-pricing/', views.MapPricingPromo.as_view() ),
    path('promo-simulate-test/',views.PromoSimulatorViewTest.as_view({'get': 'get'})),
    path('promo-simulate-test/<int:id>/',views.PromoSimulatorViewTest.as_view({'get': 'retrieve'})),
    path('upload/',views.MyUploadView.as_view({'get': 'get'}))
    # path('price-simulate' , views.)
    
]
