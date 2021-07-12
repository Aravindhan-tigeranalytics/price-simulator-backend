from rest_framework.permissions import BasePermission
from django.db.models import Q
class PricingPermission(BasePermission):
    def has_permission(self, request, view):
        # print("permission check ")
        return request.user.groups.filter(Q(name = 'admin')|Q(name = 'pricing')).exists()
    
class PromoPermission(BasePermission):
    def has_permission(self, request, view):
        # print("permission check ")
        return request.user.groups.filter(Q(name = 'admin')|Q(name = 'promo')).exists()
    
class OptimizerPermission(BasePermission):
    def has_permission(self, request, view):
        # print("permission check ")
        return request.user.groups.filter(Q(name = 'admin')|Q(name = 'optimizer')).exists()