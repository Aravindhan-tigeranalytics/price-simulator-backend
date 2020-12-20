from rest_framework import serializers

from core.models import Scenario


class ScenarioSerializer(serializers.ModelSerializer):
    class Meta:
        model = Scenario
        fields = ('id', 'name', 'comments', 'savedump')
        read_only_fields = ('id',)
