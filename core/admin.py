from django.contrib import admin
# from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
# from django.utils.translation import ugettext_lazy as _
from core import models
from django import forms
from django.urls import path
from django.shortcuts import render,redirect
from utils import excel as excel
import openpyxl
# Register your models here.
class CsvImportForm(forms.Form):
    csv_file = forms.FileField()
    # roi_file  = forms.FileField()
# @admin.register(models.Scenario)

class ModelMetaAdmin(admin.ModelAdmin):
    
    list_display = [field.name for field in models.ModelMeta._meta.fields]
    list_filter = ('account_name','corporate_segment','product_group')
    change_list_template = "admin/promo_upload.html"
    
    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            path('import-promo/', self.import_csv),
        ]
        return my_urls + urls
    def import_csv(self, request):
        if request.method == "POST":
            csv_file = request.FILES["csv_file"]
            # roi_file = request.FILES["roi_file"]
            excel.read_promo_coeff(csv_file)
            excel.read_promo_data(csv_file)
            excel.read_roi_data(csv_file)   
            # excel.lift(csv_file , roi_file)
            excel.lift_test()
            # excel.read_coeff_map(csv_file)
            self.message_user(request, "Your csv file has been imported")
            return redirect("..")
        form = CsvImportForm()
        # form = {}
        payload = {"form": form}
        return render(
            request, "admin/excel_form.html", payload
        )
    
    
class ScenarioPlannerMetricsAdmin(admin.ModelAdmin):
    list_display = [field.name for field in models.ScenarioPlannerMetrics._meta.get_fields()]
    list_filter = ('year','category','product_group','retailer','brand_filter',
    'brand_format_filter','strategic_cell_filter')
    actions = ['delete_all']
    change_list_template = "admin/upload_excel.html"
    def delete_all(self, request, queryset):
        for q in queryset:
            q.delete()
        # models.ScenarioPlannerMetrics.objects.all().delete()
    delete_all.short_description = "delete all"
    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            path('import-csv/', self.import_csv),
        ]
        return my_urls + urls
    def import_csv(self, request):
        if request.method == "POST":
            csv_file = request.FILES["csv_file"]
            excel.read_excel(csv_file)
            self.message_user(request, "Your csv file has been imported")
            return redirect("..")
        form = CsvImportForm()
        payload = {"form": form}
        return render(
            request, "admin/excel_form.html", payload
        )

    # def get_urls(self):
    #     urls = super().get_urls()
    #     my_urls = [
    #         ...
    #         path('import-csv/', self.import_csv),
    #     ]
    #     return my_urls + urls

    # def import_csv(self, request):
    #     if request.method == "POST":
    #         csv_file = request.FILES["csv_file"]
    #         reader = csv.reader(csv_file)
    #         # Create Hero objects from passed in data
    #         # ...
    #         self.message_user(request, "Your csv file has been imported")
    #         return redirect("..")
    #     form = CsvImportForm()
    #     payload = {"form": form}
    #     return render(
    #         request, "admin/csv_form.html", payload
    #     )

class ScenarioAdmin(admin.ModelAdmin):
    list_display = ['name' ,'user' , 'comments' , 'is_yearly']
    
class ModelCoefficientAdmin(admin.ModelAdmin):
    list_display = [field.name for field in models.ModelCoefficient._meta.fields]

class ModelDataAdmin(admin.ModelAdmin):
    search_fields = ['model_meta__id','model_meta__slug','model_meta__account_name']
    list_display = [field.name for field in models.ModelData._meta.fields]

class ModelROIAdmin(admin.ModelAdmin):
    search_fields = ['model_meta__id','model_meta__slug','model_meta__account_name']
    list_display = [field.name for field in models.ModelROI._meta.fields]
    
class CoeffMapAdmin(admin.ModelAdmin):
    search_fields = ['model_meta__id','model_meta__slug','model_meta__account_name']
    list_display = [field.name for field in models.CoeffMap._meta.fields]


admin.site.register(models.User)
admin.site.register(models.ScenarioPlannerMetrics,ScenarioPlannerMetricsAdmin)
admin.site.register(models.Scenario,ScenarioAdmin)
admin.site.register(models.ModelMeta,ModelMetaAdmin)
admin.site.register(models.ModelCoefficient,ModelCoefficientAdmin)
admin.site.register(models.ModelData,ModelDataAdmin)
admin.site.register(models.ModelROI,ModelROIAdmin)
admin.site.register(models.CoeffMap,CoeffMapAdmin)

# admin.site.register(models.Scenario,ScenarioAdmin)

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
