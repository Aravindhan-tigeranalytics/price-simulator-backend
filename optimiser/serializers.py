from django.contrib.auth import get_user_model, authenticate
from django.utils.translation import ugettext_lazy as _
from core import models as model
from rest_framework import serializers
from scenario_planner import fields as field
from optimiser import optimizer


class CommentSerializer(serializers.Serializer):
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

    # query = model.ModelMeta.objects.prefetch_related('data').all()
    # account_name = field.ChoiceField(choices=[i + i for i in list(query.values_list('account_name').distinct())])
    # corporate_segment = field.ChoiceField(choices=[i + i for i in list(query.values_list('corporate_segment').distinct())])
    account_name = serializers.ChoiceField(choices=
                                       OBJ_CHOICES1)
    corporate_segment = serializers.ChoiceField(choices=
                                       OBJ_CHOICES2)
    strategic_cell = serializers.ChoiceField(choices=
                                       OBJ_CHOICES3)

    objective_function = serializers.ChoiceField(
                        choices = OBJ_CHOICES)
    max_promotion = serializers.IntegerField(max_value = 53 ,min_value=0,default=23,initial=23)
    min_promotion = serializers.IntegerField(max_value = 53 ,min_value=0,default=16,initial=16)
    
    config_mac = serializers.BooleanField(default=True,initial=True)
    config_rp = serializers.BooleanField(default=True,initial=True)
    config_trade_expense = serializers.BooleanField(default=False,initial=False)
    config_units = serializers.BooleanField(default=False,initial=False)
    config_mac_perc = serializers.BooleanField(default=False,initial=False)
    config_min_length = serializers.BooleanField(default=True,initial=True)
    config_max_length = serializers.BooleanField(default=True,initial=True)
    config_promo_gap = serializers.BooleanField(default=True,initial=True)
    
    param_mac = serializers.FloatField(max_value = 10 ,min_value=0,default=1.0,initial=1.0)
    param_rp = serializers.FloatField(max_value = 10 ,min_value=0,default=1.0,initial=1.0)
    param_trade_expense = serializers.FloatField(max_value = 10 ,min_value=0,default=1.0,initial=1.0)
    param_units = serializers.FloatField(max_value = 10 ,min_value=0,default=1.0,initial=1.0)
    param_mac_perc = serializers.FloatField(max_value = 10 ,min_value=0,default=1.0,initial=1.0)
    param_min_length = serializers.IntegerField(max_value = 10 ,min_value=0,default=2,initial=2)
    param_max_length = serializers.IntegerField(max_value = 10 ,min_value=0,default=5,initial=5)
    param_promo_gap = serializers.IntegerField(max_value = 10 ,min_value=0,default=4,initial=4)
    
    result = serializers.SerializerMethodField('obj')
    def obj(self,ob):
        return optimizer.process(dict(ob))
