from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from optimiser import serializers as sc
from optimiser import optimizer

class ModelOptimize(APIView):
    serializer_class = sc.OptimizerSerializer
    def get(self, request, format=None):
        serializer = sc.OptimizerSerializer()
        return Response(serializer.data)
    
    def post(self, request, format=None):
        content = None
        
        serializer = sc.OptimizerSerializer(data=request.data)
        
        # import pdb
        # pdb.set_trace()
        
        if serializer.is_valid():
            
            return Response(serializer.data, status=status.HTTP_201_CREATED)
            
        return Response(content, status=status.HTTP_201_CREATED)