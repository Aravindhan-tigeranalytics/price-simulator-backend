from django.contrib import admin
# from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
# from django.utils.translation import ugettext_lazy as _
from core import models
# Register your models here.

admin.site.register(models.User)
admin.site.register(models.Scenario)

# class UserAdmin(BaseUserAdmin):
#     ordering = ['id']
#     list_display = ['email', 'name']
#     fieldsets = (
#         (None, {'fields': ('email', 'password')}),
#         (_('Personal info'), {'fields': ('name',)}),
#         (_('Permissions'), {
#             'fields': ('is_active', 'is_staff', 'is_superuser',),
#         }),
#         (_('Important dates'), {'fields': ('last_login', 'date_joined')}),
#     )
#     add_fieldsets = (
#         (None, {
#             'classes': ('wide',),
#             'fields': ('username', 'password1', 'password2'),
#         }),
#     )


# admin.site.register(models.User, BaseUserAdmin)
