from rest_framework import serializers
from utils import exceptions as exception
from core.models import Scenario,ScenarioPlannerMetrics


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

    
    