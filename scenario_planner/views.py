# from xlwt import Workbook
from rest_framework import viewsets, mixins
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import action

from core.models import Scenario , ScenarioPlannerMetrics
from scenario_planner import serializers as sc
from rest_framework import serializers
from utils import exceptions as exception
from utils import excel as excel
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
        if(query.filter(name = serializer.validated_data['name']).exists()):
            raise exception.AlredyExistsException
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
