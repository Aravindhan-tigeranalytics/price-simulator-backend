from django.urls import path

from optimiser import views

app_name = "optimiser"

urlpatterns = [
    # path('calculate/',views.ModelOptimize.as_view()),
    path('calculate/',views.ModelOptimize.as_view({'get':'get'})),
    path('calculate/<int:id>/',views.ModelOptimize.as_view({'get':'retrieve'})),
    path('save/',views.SaveOptimier.as_view({'get':'get','post' : 'post'})),
    path('list-saved-optimizer/', views.LoadScenarioOptimizer.as_view({'get' : 'list',})),
    path('list-saved-optimizer/<int:id>/', views.LoadScenarioOptimizer.as_view({'get' : 'retrieve'})),
    path('list-saved-optimizer/<int:id>/<int:_id>/', views.LoadScenarioOptimizer.as_view({'get' : 'retrieve_pricing_optimizer'})),
    path('map-promo-optimizer/', views.MapOptimizerPromo.as_view() ),
    path('map-pricing-optimizer/', views.MapOptimizerPricing.as_view() ),
    path('optimizer-download/', views.DownloadOptimizer.as_view() ),
]
