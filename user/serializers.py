from django.contrib.auth import get_user_model, authenticate 
from django.contrib.auth.models import Group
from django.utils.translation import ugettext_lazy as _


from rest_framework import serializers

class GroupSerializer(serializers.ModelSerializer):    
    class Meta:
        model = Group
        fields = ('name',)
        
class UserSerializer(serializers.ModelSerializer):
    # groups = serializers.SerializerMethodField()
    groups = GroupSerializer(many=True)
    class Meta:
        model = get_user_model()
        fields = ('email', 'password', 'name' , 'id' , 'groups')
        extra_kwargs = {
            'password': {'write_only': True, 'min_length': 5}
        }
    # def get_groups(self, obj):
    #     return [group.name for group in obj.groups]

    def create(self, validated_data):
        return get_user_model().objects.create_user(**validated_data)


class AuthTokenSerializer(serializers.Serializer):
    email = serializers.CharField()
    password = serializers.CharField(
        style={'input_type': 'password'},
        trim_whitespace=False
    )
    # user = UserSerializer()

    def validate(self, attrs):
        # print(attrs, "ATTRS")
        email = attrs.get('email')
        password = attrs.get('password')
        user = authenticate(
            request=self.context.get('request'),
            username=email,
            password=password
        )
        if not user:
            msg = _('Unable to authenticate')
            raise serializers.ValidationError(msg, code='authentication')
        attrs['user'] = user
        # import pdb
        # pdb.set_trace()
        # print(user , "TOKEN USER")
        # print(attrs , "ATTRS ")
        return attrs
