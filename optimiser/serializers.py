from django.contrib.auth import get_user_model, authenticate
from django.utils.translation import ugettext_lazy as _
from rest_framework.exceptions import server_error
from core import models as model
from rest_framework import serializers
from scenario_planner import fields as field
from optimiser import optimizer


class OptimizerSerializer(serializers.Serializer):
    OBJ_CHOICES = (
        ("MAC", "MAC"), 
        ("RP", "RP"), 
        ("Trade_Expense", "Trade Expense"), 
        ("Units", "Units"), 
    )

    OBJ_CHOICES1 = (
        ("cell", "Choose Strategic Cell"), 
    )
    OBJ_CHOICES2 = (
        ("cell", "Choose Account Name"), 
    )
    OBJ_CHOICES3 = (
        ("cell", "Choose Corporate Segment"), 
    )
    query = model.ModelMeta.objects.prefetch_related('data').all()
    account_name = field.ChoiceField(choices=[i + i for i in list(query.values_list('account_name').distinct())])
    corporate_segment = field.ChoiceField(choices=[i + i for i in list(query.values_list('corporate_segment').distinct())])
    strategic_cell = serializers.ChoiceField(choices=[i + i for i in list(query.filter(strategic_cell_filter__isnull=False).values_list('strategic_cell_filter').distinct())])
    brand = serializers.ChoiceField(choices=[i + i for i in list(query.filter(brand_filter__isnull=False).values_list('brand_filter').distinct())])
    brand_format = serializers.ChoiceField(choices=[i + i for i in list(query.filter(brand_format_filter__isnull=False).values_list('brand_format_filter').distinct())])

    product_group = field.ChoiceField(choices=[i + i for i in list(query.values_list('product_group').distinct())])

    objective_function = serializers.ChoiceField(
                        choices = OBJ_CHOICES)
    mars_tpr = serializers.CharField(allow_blank = True)
    co_investment = serializers.IntegerField(max_value = 52 ,min_value=0,default=0,initial=0)
    # max_promotion = serializers.IntegerField(max_value = 53 ,min_value=0,default=23,initial=23)
    # min_promotion = serializers.IntegerField(max_value = 53 ,min_value=0,default=16,initial=16)
    
    config_mac = serializers.BooleanField(default=True,initial=True)
    config_rp = serializers.BooleanField(default=True,initial=True)
    config_trade_expense = serializers.BooleanField(default=False,initial=False)
    config_units = serializers.BooleanField(default=False,initial=False)
    config_nsv = serializers.BooleanField(default=False,initial=False)
    config_gsv = serializers.BooleanField(default=False,initial=False)
    config_sales = serializers.BooleanField(default=False,initial=False)
    config_mac_perc = serializers.BooleanField(default=False,initial=False)
    config_rp_perc = serializers.BooleanField(default=True,initial=True)
    config_min_consecutive_promo = serializers.BooleanField(default=True,initial=True)
    config_max_consecutive_promo = serializers.BooleanField(default=True,initial=True)
    config_promo_gap = serializers.BooleanField(default=True,initial=True)
    
    param_mac = serializers.FloatField(max_value = 10 ,min_value=0,default=1.0,initial=1.0)
    param_rp = serializers.FloatField(max_value = 10 ,min_value=0,default=1.0,initial=1.0)
    param_trade_expense = serializers.FloatField(max_value = 10 ,min_value=0,default=1.0,initial=1.0)
    param_units = serializers.FloatField(max_value = 10 ,min_value=0,default=1.0,initial=1.0)
    param_nsv = serializers.FloatField(max_value = 10 ,min_value=0,default=1.0,initial=1.0)
    param_gsv = serializers.FloatField(max_value = 10 ,min_value=0,default=1.0,initial=1.0)
    param_sales =serializers.FloatField(max_value = 10 ,min_value=0,default=1.0,initial=1.0)
    param_mac_perc = serializers.FloatField(max_value = 10 ,min_value=0,default=1.0,initial=1.0)
    param_rp_perc = serializers.FloatField(max_value = 10 ,min_value=0,default=1.0,initial=1.0)
    param_min_consecutive_promo = serializers.IntegerField(max_value = 10 ,min_value=0,default=6,initial=6)
    param_max_consecutive_promo = serializers.IntegerField(max_value = 10 ,min_value=0,default=6,initial=6)
    param_promo_gap = serializers.IntegerField(max_value = 10 ,min_value=0,default=2,initial=2)
    param_total_promo_min = serializers.IntegerField(max_value = 52 ,min_value=0,default=10,initial=10)
    param_total_promo_max  = serializers.IntegerField(max_value = 52 ,min_value=0,default=26,initial=26)
    param_compulsory_no_promo_weeks =serializers.CharField(allow_blank = True)
    param_compulsory_promo_weeks = serializers.CharField(allow_blank = True)
    
    
    result = serializers.SerializerMethodField('obj')
    def obj(self,ob):
        return optimizer.process(dict(ob))
