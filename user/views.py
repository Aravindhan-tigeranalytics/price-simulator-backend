from rest_framework import generics
from rest_framework.views import APIView
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.authtoken.models import Token
from rest_framework.authentication import TokenAuthentication
from rest_framework.settings import api_settings    
from user.serializers import UserSerializer, AuthTokenSerializer
from rest_framework.response import Response
from rest_framework import status


class CreateUserView(generics.CreateAPIView):
    serializer_class = UserSerializer


class Logout(APIView):
    authentication_classes = (TokenAuthentication,)
    def get(self, request, format=None):
        # simply delete the token to force a login
        request.user.auth_token.delete()
        return Response(status=status.HTTP_200_OK)

class CreateTokenView(ObtainAuthToken):
    serializer_class = AuthTokenSerializer
    renderer_classes = api_settings.DEFAULT_RENDERER_CLASSES
    def post(self, request, *args, **kwargs):
        # import pdb
        # pdb.set_trace()
        serializer = self.serializer_class(data=request.data,
                                        context={'request': request})
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        token, created = Token.objects.get_or_create(user=user)
        print(user, " USER INDO")
       
        # import pdb
        # pdb.set_trace()
        user_ser = UserSerializer(token.user)
        

        return Response({
            'token': token.key,
            'user' : user_ser.data
        })
