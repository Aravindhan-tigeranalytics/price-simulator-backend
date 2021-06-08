from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from optimiser import serializers as sc
from optimiser import optimizer

class ModelOptimize(APIView):
    serializer_class = sc.CommentSerializer
    def get(self, request, format=None):
        content = optimizer.process()
        return Response(content)
    
    def post(self, request, format=None):
        content = None
        
        serializer = sc.CommentSerializer(data=request.data)
        
        if serializer.is_valid():
            
            return Response(serializer.data, status=status.HTTP_201_CREATED)
            
        return Response(content, status=status.HTTP_201_CREATED)