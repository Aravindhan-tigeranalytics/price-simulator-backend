from django.contrib import admin
from . import models as models

# Register your models here.

class TestAdmin(admin.ModelAdmin):
    
    list_display = [field.name for field in models.TestDb._meta.fields]
admin.site.register(models.TestDb,TestAdmin)
