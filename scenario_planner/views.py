# from xlwt import Workbook
from utils import util
from rest_framework import fields, viewsets, mixins
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import action
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.renderers import JSONRenderer
from rest_framework import status
from django.db.models.query import Prefetch
from django.shortcuts import get_object_or_404
from core.models import ModelMeta, ModelROI, Scenario , ScenarioPlannerMetrics,ModelData,ModelCoefficient
from scenario_planner import serializers as sc
from rest_framework import serializers
from . import query as pd_query
from utils import units_calculation as uc

import utils
from . import calculations as cal
from utils import exceptions as exception
from utils import excel as excel
from utils import optimizer as optimizer
from json import loads, dumps

# import xlwt
# from xlrd import ope
from xlwt import Workbook
from django.http import HttpResponse
# import StringIO
import xlsxwriter
import json
import io


class ScenarioViewSet(viewsets.GenericViewSet, mixins.ListModelMixin, mixins.CreateModelMixin,
mixins.UpdateModelMixin,mixins.DestroyModelMixin):
    authentication_classes = (TokenAuthentication,)
    permission_classes = (IsAuthenticated,)
    queryset = Scenario.objects.all()
    serializer_class = sc.ScenarioSerializer

    def get_queryset(self):
        # queryset = super(ScenarioViewSet, self).get_queryset()
        return self.queryset.filter(user=self.request.user).order_by('-name')
    def filter_queryset(self,queryset):
        # print(self.request.GET.get('yearly' , '') , "yearly value")
        # print(type(self.request.GET.get('yearly' , '')) , "yearly type")
        if(self.request.GET.get('yearly' , '')!="ALL"):
            return queryset.filter(is_yearly = self.request.GET.get('yearly' , '') == 'true')
        return queryset
    
    def create(self, request , *args , **kwargs):
        if not request.data['name']:
            raise exception.EmptyException
        query = self.queryset.filter(user=self.request.user)
        if(query.count() > 20):
            raise exception.CountExceedException
        # if(query.filter(name = serializer.validated_data['name']).exists()):
        #     raise exception.AlredyExistsException
        super().create(request,args,kwargs)
        

   
    # def perform_create(self, serializer):
    #     query = self.queryset.filter(user=self.request.user)
    #     if(query.count() > 2):
    #         raise exception.CountExceedException
    #     if(query.filter(name = serializer.validated_data['name']).exists()):
    #         raise exception.AlredyExistsException
    #     serializer.save(user=self.request.user)

    def put(self,**kwargs):
        print(self.request , "request")
        print(kwargs , "kwargs")
        self.partial_update(self.request,*args,**kwargs)
    
    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)

    # def
class ScenarioPlannerMetricsViewSet(viewsets.GenericViewSet, mixins.ListModelMixin):
    queryset = ScenarioPlannerMetrics.objects.all()
    serializer_class = sc.ScenarioPlannerMetricsSerializer

class ScenarioPlannerMetricsViewSetObject(viewsets.GenericViewSet, mixins.ListModelMixin):
    queryset = ScenarioPlannerMetrics.objects.all()
    serializer_class = sc.ScenarioPlannerMetricsSerializerObject

class ExampleViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Scenario.objects.all()
    serializer_class = sc.ScenarioSerializer


    @action(methods=['post'], detail=True)
    def download(self, *args, **kwargs):
        # print(args , "ARGS")
        # print(kwargs , "KWARGS")
        # print(self , "self")
        formdata = self.request.data['data']
        typ = self.request.data['type']
       
        output = io.BytesIO()
        if typ == 'comp':
            excel.excel_summary(json.loads(formdata) , output)
        else:
            excel.excel(json.loads(formdata) , output)
        output.seek(0)
        filename = 'django_simple.xlsx'
        response = HttpResponse(
            output,
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        response['Content-Disposition'] = 'attachment; filename=%s' % filename
        return response

    @action(methods=['post'], detail=True)
    def getData(self, *args, **kwargs):
        # print(args , "ARGS")
        # print(kwargs , "KWARGS")
        formdata = self.request.data['file']
        # print(formdata)
        # import pdb
        # pdb.set_trace()
        # print(self , "self")
        return 1
       
        # instance = self.get_object()

        # # get an open file handle (I'm just using a file attached to the model for this example):
        # file_handle = instance.file.open()

        # # send file
        # response = FileResponse(file_handle, content_type='whatever')
        # response['Content-Length'] = instance.file.size
        # response['Content-Disposition'] = 'attachment; filename="%s"' % instance.file.name

        # return response

def down(request):
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output)
    worksheet = workbook.add_worksheet()

    expenses = (
        ['Rent', 1000],
        ['Gas',   100],
        ['Food',  300],
        ['Gym',    50],
    )

    row = 3
    col = 3

    # Iterate over the data and write it out row by row.
    for item, cost in (expenses):
        worksheet.write(row, col,     item)
        worksheet.write(row, col + 1, cost)
        row += 1

    worksheet.write(row, 3, 'Total')
    worksheet.write(row, 4, '=SUM(B1:B4)')

    workbook.close()
    output.seek(0)
    filename = 'django_simple.xlsx'
    response = HttpResponse(
        output,
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = 'attachment; filename=%s' % filename
    return response


def WriteToExcel(weather_data, town=None):
    # output = StringIO.StringIO()
    workbook = xlsxwriter.Workbook("output")

    # Here we will adding the code to add data

    workbook.close()
    xlsx_data = " output.getvalue()"
    # xlsx_data contains the Excel file
    return xlsx_data

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

# class PromoSimulatorView(viewsets.GenericViewSet,mixins.ListModelMixin):
#     queryset = ModelData.objects.select_related('model_meta').prefetch_related(
#         Prefetch('model_meta__coefficient',
#                  queryset=ModelCoefficient.objects.all(),
#                  to_attr='prefetched_coeff')
#         ).order_by('id')[:10]
     
#     serializer_class = sc.ModelDataSerializer
#     # serializer_class = sc.PromoSimulatorSerializer
#     def get(self, request, format=None):
#         serializer = sc.ModelDataSerializer()
#         return Response(serializer.data, status=status.HTTP_201_CREATED)
#     def post(self, request, format=None):
#         serializer = sc.ModelDataSerializer()
#         return Response(serializer.data, status=status.HTTP_201_CREATED)

class PromoSimulatorView(viewsets.GenericViewSet):
    queryset = ModelMeta.objects.prefetch_related(
        Prefetch(
            'data',
        queryset = ModelData.objects.all().order_by('week'),
        to_attr='prefetched_data'
        ),
        # 'data',
        Prefetch(
            'coefficient',
            queryset=ModelCoefficient.objects.all(),
            to_attr='prefetched_coeff'
        ),
        Prefetch(
            'roi',
            queryset=ModelROI.objects.all().order_by('week'),
            to_attr='prefetched_roi'
        )
    ).order_by('id')
     
    def get(self, request, format=None):
        
        serializer = sc.ModelMetaGetSerializer()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    def get_serializer_class(self):
        return sc.ModelMetaGetSerializer
    
    def get_queryset(self):
        return super().get_queryset()
    
    def post(self, request, format=None):
        query = self.queryset
        get_serializer = sc.ModelMetaGetSerializer(request.data)
        value_dict = loads(dumps((get_serializer.to_internal_value(request.data))))
        meta = {
            'account_name' : value_dict['account_name'],
            'corporate_segment' : value_dict['corporate_segment'],
            'product_group' : value_dict['product_group']
        }
        
        print(value_dict , "value dict")
        coeff_list , data_list ,roi_list = pd_query.get_list_value_from_query(ModelCoefficient,ModelData,ModelROI,value_dict['account_name'],
                                           value_dict['product_group'] )
        # import pdb
        # pdb.set_trace()
        cloned_data_list = cal.update_from_request(data_list, value_dict)
        # import pdb
        # pdb.set_trace()
       
        parsed = json.loads(uc.list_to_frame(coeff_list , data_list).to_json(orient="records"))
        parsed_new = json.loads(uc.list_to_frame(coeff_list , cloned_data_list,flag=True).to_json(orient="records"))
        res = cal.calculate_financial_mertrics(coeff_list , data_list ,roi_list,
                                               parsed , 'base')
        res_new = cal.calculate_financial_mertrics(coeff_list , cloned_data_list ,roi_list,
                                               parsed_new , 'simulated',value_dict['promo_elasticity'])
       
        # import pdb
        # pdb.set_trace()
        
        # query = query.get(account_name = value_dict['account_name'],
        #           corporate_segment=value_dict['corporate_segment'],
        #           product_group = value_dict['product_group'])
        
        
         
        # import copy
        
        # simulated = copy.deepcopy(query)
        
        # serializer = sc.ModelMetaSerializer(query)
        
        # serializer2 = sc.ModelMetaSerializer(simulated)
         
        # sales = 0
        # units = 0
        # te= 0
        # lsv = 0
        # nsv = 0
        # mac = 0
        # rp = 0
        # asp = 0
        # avg_promo_selling_price = 0
        # roi = 0
        # rp_percent = 0
        # mac_percent = 0
        # volume = 0
        # te_per_unit = 0
        # te_percent_of_lsv = 0
        # base_units = 0
        # increment_units = 0
        # lift = 0
       
        
        # for i in range(0,52):
            
        #     sales = sales + serializer.data['prefetched_data'][i]['base']['total_rsv_w_o_vat']
        #     units = units + serializer.data['prefetched_data'][i]['base']['predicted_units']
        #     # import pdb
        #     # pdb.set_trace()
        #     base_units = base_units + serializer.data['prefetched_data'][i]['base']['base_unit']
        #     volume = volume + serializer.data['prefetched_data'][i]['base']['total_weight_in_tons']
        #     te_per_unit = te_per_unit + serializer.data['prefetched_data'][i]['base']['te_per_units']
        #     increment_units = increment_units + serializer.data['prefetched_data'][i]['base']['incremental_unit']
        #     # increment_units = increment_units + serializer2.data['prefetched_data'][i]['base']['predicted_units']
        #     te = te + serializer.data['prefetched_data'][i]['base']['trade_expense']
        #     nsv = nsv + serializer.data['prefetched_data'][i]['base']['total_nsv']
        #     mac = mac + serializer.data['prefetched_data'][i]['base']['mars_mac']
        #     lsv = lsv + serializer.data['prefetched_data'][i]['base']['total_lsv']
        #     rp = rp + serializer.data['prefetched_data'][i]['base']['retailer_margin']
        #     roi = roi + serializer.data['prefetched_data'][i]['base']['roi']
        #     asp = util.average(asp,query.prefetched_data[i].wk_sold_avg_price_byppg)
        #     avg_promo_selling_price = util.average(avg_promo_selling_price,serializer.data['prefetched_data'][i]['base']['promo_asp'])
        #     rp_percent = util.average(rp_percent,serializer.data['prefetched_data'][i]['base']['retailer_margin_percent_of_rsp'])
        #     mac_percent = util.average(mac_percent,serializer.data['prefetched_data'][i]['base']['mars_mac_percent_of_nsv'])
        #     te_percent_of_lsv = util.average(te_percent_of_lsv,serializer.data['prefetched_data'][i]['base']['te_percent_of_lsv'])
        #     lift = lift + (serializer.data['prefetched_data'][i]['base']['incremental_unit']/serializer.data['prefetched_data'][i]['base']['base_unit'])
        #     serializer.data['prefetched_data'][i]['simulated'] = serializer2.data['prefetched_data'][i]['base']
        # json_data = serializer.data
        # # import pdb
        # # pdb.set_trace()
        # json_data['units'] = units
        # json_data['incremental_units'] = increment_units
        # json_data['base_units'] = base_units
        # json_data['total_rsv_w_o_vat'] = sales
        # json_data['volumes_in_tonnes'] = volume
        # json_data['te'] = te
        # json_data['roi'] = roi
        # json_data['nsv'] = nsv
        # json_data['mac'] = mac
        # json_data['lsv'] = lsv
        # json_data['rp'] = rp
        # json_data['average_selling_price'] = asp
        # json_data['avg_promo_selling_price'] = avg_promo_selling_price
        # json_data['te_per_unit'] = te_per_unit
        # json_data['te_percent_of_lsv'] = te_percent_of_lsv
        # json_data['rp_percent_of_rsp'] = rp_percent
        # json_data['mac_percent_of_nsv'] = mac_percent
        # json_data['lift'] = lift

        return Response( {**meta,**res , **res_new}, status=status.HTTP_201_CREATED)
    
class PromoSimulatorViewTest(APIView):
    queryset = ModelMeta.objects.prefetch_related(
        'data',
        Prefetch(
            'coefficient',
            queryset=ModelCoefficient.objects.all(),
            to_attr='prefetched_coeff'
        )
    ).order_by('id')
     
    def get(self, request, format=None):
        query=ModelMeta.objects.all()
        
        serializer = sc.ModelMetaGetSerializer()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    # def get_serializer_class(self):
    #     return sc.ModelMetaGetSerializer
    
    # def get_queryset(self):
    #     return super().get_queryset()
    
    def post(self, request, format=None):
        query = self.queryset
        get_serializer = sc.ModelMetaGetSerializer(request.data)
        value_dict = loads(dumps((get_serializer.to_internal_value(request.data))))
        print(value_dict , "value dict")
        query = query.get(account_name = value_dict['account_name'],
                  corporate_segment=value_dict['corporate_segment'],
                  product_group = value_dict['product_group'])
        serializer = sc.ModelMetaSerializer(query)
        return Response(serializer.data, status=status.HTTP_201_CREATED)