from rest_framework import serializers
from utils import exceptions as exception
from core.models import Scenario


class ScenarioSerializer(serializers.ModelSerializer):
    class Meta:
        model = Scenario
        fields = ('id', 'name', 'comments', 'savedump')
        read_only_fields = ('id',)

    def validate(self,data):
        if not data['name']:
            raise exception.EmptyException
        return data
    # def save():
    #     pass
