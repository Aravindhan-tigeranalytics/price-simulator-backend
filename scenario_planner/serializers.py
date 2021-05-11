from rest_framework import serializers
from utils import exceptions as exception
from core.models import Scenario,ScenarioPlannerMetrics
from utils import optimizer as optimizer


class ScenarioSerializer(serializers.ModelSerializer):
    class Meta:
        model = Scenario
        fields = ('id', 'name', 'comments', 'savedump','is_yearly')
        read_only_fields = ('id',)

    def validate(self,data):
        print(data , "DATA")
        if not data['name']:
            raise exception.EmptyException
        return data

class ScenarioPlannerMetricsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ScenarioPlannerMetrics
        fields = '__all__'

class ScenarioPlannerMetricsSerializerObject(serializers.ModelSerializer):
    my_field = serializers.SerializerMethodField('obj')
    def obj(self,metric):
        print(metric , "metics additionsl")
        return "hola"
    class Meta:
        model = ScenarioPlannerMetrics
        fields = '__all__'
GEEKS_CHOICES =( 
    ("1", "One"), 
    ("2", "Two"), 
    ("3", "Three"), 
    ("4", "Four"), 
    ("5", "Five"), 
)

class CommentSerializer(serializers.Serializer):
    OBJ_CHOICES = (
        ("MAC", "MAC"), 
        ("RP", "RP"), 
        ("Trade_Expense", "Trade Expense"), 
        ("Units", "Units"), 
    )
#     Max p[romotion 23
# min promotion 16
# maximizing option(objective function)  MAC RP TE etc
    # content = serializers.CharField(max_length=200)
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
        # import pdb
        # pdb.set_trace()
        # print(self.data , "self data ")
        # print(dict(ob) , "metics additionsl")
        # optimizer.process(serializer.data)
        return optimizer.process(dict(ob))