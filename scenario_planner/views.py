# from xlwt import Workbook
from rest_framework import viewsets, mixins
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated

from core.models import Scenario
from scenario_planner import serializers
from utils import exceptions as exception
# import xlwt
from xlwt import Workbook
from django.http import HttpResponse
# import StringIO
import xlsxwriter


class ScenarioViewSet(viewsets.GenericViewSet, mixins.ListModelMixin, mixins.CreateModelMixin,
mixins.UpdateModelMixin,mixins.DestroyModelMixin):
    authentication_classes = (TokenAuthentication,)
    permission_classes = (IsAuthenticated,)
    queryset = Scenario.objects.all()
    serializer_class = serializers.ScenarioSerializer

    def get_queryset(self):
       
        return self.queryset.filter(user=self.request.user).order_by('-name')

    def perform_create(self, serializer):
        query = self.queryset.filter(user=self.request.user)
        if(query.count() > 1):
            raise exception.CountExceedException
        if(query.filter(name = serializer.validated_data['name']).exists()):
            raise exception.AlredyExistsException
        serializer.save(user=self.request.user)

    def put(self,**kwargs):
        self.partial_update(self.request,*args,**kwargs)
    
    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)

    # def


def down(request):
    # import xlwt
    wb = Workbook()

# add_sheet is used to create sheet.
    sheet1 = wb.add_sheet('Sheet 1')

    sheet1.write(1, 0, 'ISBT DEHRADUN')
    sheet1.write(2, 0, 'SHASTRADHARA')
    sheet1.write(3, 0, 'CLEMEN TOWN')
    sheet1.write(4, 0, 'RAJPUR ROAD')
    sheet1.write(5, 0, 'CLOCK TOWER')
    sheet1.write(0, 1, 'ISBT DEHRADUN')
    sheet1.write(0, 2, 'SHASTRADHARA')
    sheet1.write(0, 3, 'CLEMEN TOWN')
    sheet1.write(0, 4, 'RAJPUR ROAD')
    sheet1.write(0, 5, 'CLOCK TOWER')

    wb.save('xlwt example.xls')
    # if 'excel' in request.POST:
    response = HttpResponse(content_type='application/xls')
    response['Content-Disposition'] = 'attachment; filename=Report.xls'
    # xlsx_data = WriteToExcel("weather_period", "town")
    response.write(wb)
    return response


def WriteToExcel(weather_data, town=None):
    # output = StringIO.StringIO()
    workbook = xlsxwriter.Workbook("output")

    # Here we will adding the code to add data

    workbook.close()
    xlsx_data = " output.getvalue()"
    # xlsx_data contains the Excel file
    return xlsx_data
