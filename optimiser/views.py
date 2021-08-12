import pdb
from numpy import product
from scenario_planner import serializers
from django.http import HttpResponse
from django.shortcuts import render
from django.db.models import Q
from django.db.models import Prefetch
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import viewsets
from rest_framework import status
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from optimiser import serializers as sc
from optimiser import permissions as perm
from scenario_planner import serializers as ser
from optimiser import optimizer
from optimiser import process as process
from core import models as model
from utils import excel as excel
from utils import exceptions as exception
from optimiser import utils as opt_util
import ast



class DownloadOptimizer(APIView):
    serializer_class = sc.DownloadOptimizerSerializer
    authentication_classes = (TokenAuthentication,)
    permission_classes = (IsAuthenticated,)
    
    def post(self, request, format=None):
        
        filename = 'promo_optimizer.xlsx'
        response = HttpResponse(
            excel.download_excel_optimizer(
                request.data['account_name'],
                request.data['product_group'],
                ast.literal_eval(request.data['optimizer_data'])),
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        response['Content-Disposition'] = 'attachment; filename=%s' % filename
        return response

class LoadScenarioOptimizer(viewsets.ReadOnlyModelViewSet):
    # queryset = Scenario.objects.filter(scenario_type = 'promo')
    queryset = model.SavedScenario.objects.all()
    # .filter(
        # Q(scenario_type='promo') | Q(scenario_type='optimizer')
    # )
    # .prefetch_related(
    #     Prefetch('promo_saved',queryset=model.PromoSave.objects.all() , to_attr= 'saved_promotion'),
    #     Prefetch('optimizer_saved',queryset=model.OptimizerSave.objects.all() , to_attr='saved_optimizer')
    # )
    # serializer_class = sc.PromoScenarioSavedList
    serializer_class = ser.ScenarioSavedList
    lookup_field = "id"
    
    def list(self, request, *args, **kwargs):
        sp_val = super().list(request, *args, **kwargs)
        val =  []
        for sp in sp_val.data:
            if sp['scenario_type'] == 'pricing':
                del sp['has_price']
            val.append(sp)
        sp_val.data = val
        return sp_val
    
    def retrieve_pricing_optimizer(self, request, *args, **kwargs):
        pricing_save_id = request.parser_context['kwargs']['_id']
        pricing_week = model.PricingWeek.objects.select_related(
                'pricing_save'
                ).filter(pricing_save__id = int(pricing_save_id))
    
        return Response( optimizer.process(pricing_week = pricing_week), 
                    status=status.HTTP_201_CREATED)
    
    
    def retrieve_pricing_promo(self, request, *args, **kwargs):
        pass
        
      
   
        
    def retrieve(self, request, *args, **kwargs):
        obj = self.get_object()
        if obj.scenario_type == 'pricing':
            pricing_save = model.PricingSave.objects.filter(saved_scenario = obj)
            serializer = ser.PricingSaveSerializer(pricing_save,many=True)
            return Response(serializer.data , status=200)
           
        if obj.scenario_type == 'optimizer':
            
            opt_save = model.OptimizerSave.objects.filter(saved_scenario = obj)
            # opt_list = [list(i) for i in opt_save]
            account_name = opt_save[0].model_meta.account_name
            product_group = opt_save[0].model_meta.product_group
            corporate_segment = opt_save[0].model_meta.corporate_segment
            tpr = model.ModelData.objects.select_related('model_meta').values(
            'tpr_discount' , 'year','quater' , 'week' , 'date','promo_depth' , 'co_investment',
            'flag_promotype_motivation' , 'flag_promotype_n_pls_1' , 'flag_promotype_traffic'
            ).filter(
            model_meta__account_name = account_name, model_meta__product_group = product_group
            ).order_by('week')
            tpr_list = list(tpr)
            
           
          
            for i in tpr_list:
                # import pdb
                # pdb.set_trace()
                i['tpr_discount'] = opt_save[i['week']-1].optimum_promo
                i['co_investment'] = opt_save[i['week']-1].optimum_co_investment
                i['promo_depth'] = opt_save[i['week']-1].optimum_promo
                if (opt_save[i['week']-1].mechanic == 'N+1'):
                    i['flag_promotype_n_pls_1'] = 1
                if (opt_save[i['week']-1].mechanic == 'Motivation'):
                    i['flag_promotype_motivation'] = 1
                    
                    
                    
           
            min_consecutive_promo,max_consecutive_promo,min_length_gap,tot_promo_min,tot_promo_max,no_of_promo, no_of_waves = optimizer.get_promo_wave_values([i['tpr_discount'] for i in tpr])

        
            serializer = sc.OptimizerSerializer({'param_total_promo_min' : tot_promo_min,
                                                'param_total_promo_max' : tot_promo_max,
                                                'param_promo_gap' : min_length_gap,
                                                'param_max_consecutive_promo' : max_consecutive_promo,
                                                'param_min_consecutive_promo' : min_consecutive_promo,
                                            'param_mac' : 1.0, 'param_rp' : 1.0,'param_trade_expense' : 1.0,'param_units' : 1.0,'param_nsv' :1.0,'param_gsv' : 1.0,'param_sales' : 1.0,'param_mac_perc' : 1.0,'param_rp_perc' : 1.0,
                                            'config_mac' : True, 'config_rp' : True,'config_trade_expense' : False,'config_units' : False,'config_nsv' :False,'config_gsv' : False,'config_sales' : False,'config_mac_perc' : False,
                                            'config_rp_perc' : True, 'config_min_consecutive_promo' : True, 'config_max_consecutive_promo' : True, 'config_promo_gap' : True,
                                                'account_name' :account_name,'corporate_segment' : corporate_segment,'brand':'','brand_format':'',
                                                'product_group' : product_group,'strategic_cell':'','result' : ''} )
            if serializer.is_valid():
                pass
            res = {"data" : serializer.data , "weekly" : tpr_list}
            res["data"]["param_no_of_waves"] = no_of_waves
            res["data"]["param_no_of_promo"] = no_of_promo
            return Response(res,200)
            
        if obj.scenario_type == 'promo':
           
            promo_save = model.PromoSave.objects.filter(saved_scenario = obj)
            account_name = promo_save[0].account_name
            product_group = promo_save[0].product_group
            corporate_segment = promo_save[0].corporate_segment
            
            promo_week = model.PromoWeek.objects.select_related(
                'pricing_save','pricing_save__saved_scenario','pricing_save__saved_pricing'
                ).filter(pricing_save__saved_scenario = obj).order_by('week').values()
            promo_week_list = list(promo_week)
            tpr = model.ModelData.objects.select_related('model_meta').values(
            'tpr_discount' , 'year','quater' , 'week' , 'date','promo_depth' , 'co_investment',
            'flag_promotype_motivation' , 'flag_promotype_n_pls_1' , 'flag_promotype_traffic'
            ).filter(
            model_meta__account_name = account_name, model_meta__product_group = product_group
            ).order_by('week')
            tpr_list = list(tpr)
           
            for pwl in promo_week_list:
                
                index = pwl['week'] - 1
                tpr_list[index]['tpr_discount'] =  pwl['promo_depth']
            # import pdb
            # pdb.set_trace()
           
            min_consecutive_promo,max_consecutive_promo,min_length_gap,tot_promo_min,tot_promo_max,no_of_promo, no_of_waves = optimizer.get_promo_wave_values([i['tpr_discount'] for i in tpr_list])

        
            serializer = sc.OptimizerSerializer({'param_total_promo_min' : tot_promo_min,
                                                'param_total_promo_max' : tot_promo_max,
                                                'param_promo_gap' : min_length_gap,
                                                'param_max_consecutive_promo' : max_consecutive_promo,
                                                'param_min_consecutive_promo' : min_consecutive_promo,
                                            'param_mac' : 1.0, 'param_rp' : 1.0,'param_trade_expense' : 1.0,'param_units' : 1.0,'param_nsv' :1.0,'param_gsv' : 1.0,'param_sales' : 1.0,'param_mac_perc' : 1.0,'param_rp_perc' : 1.0,
                                            'config_mac' : True, 'config_rp' : True,'config_trade_expense' : False,'config_units' : False,'config_nsv' :False,'config_gsv' : False,'config_sales' : False,'config_mac_perc' : False,
                                            'config_rp_perc' : True, 'config_min_consecutive_promo' : True, 'config_max_consecutive_promo' : True, 'config_promo_gap' : True,
                                                'account_name' :account_name,'corporate_segment' : corporate_segment,'brand':'','brand_format':'',
                                                'product_group' : product_group,'strategic_cell':'','result' : ''} )
            if serializer.is_valid():
                pass
            res = {"data" : serializer.data , "weekly" : tpr}
            res["data"]["param_no_of_waves"] = no_of_waves
            res["data"]["param_no_of_promo"] = no_of_promo
            return Response(res,200)
        
        
       
       
        
        



class ModelOptimizeBKP(viewsets.GenericViewSet):
    authentication_classes = (TokenAuthentication,)
    permission_classes = (perm.OptimizerPermission,)
    serializer_class = sc.OptimizerMeta
    queryset =model.ModelMeta.objects.all()
    def get(self, request, format=None):
        # serializer = sc.OptimizerSerializer()
        serializer = sc.OptimizerMeta(self.queryset.filter(
            id__in = request.user.allowed_retailers.all()
            ) , many=True)
        return Response(serializer.data)
    def post(self,request,format=None):
        # import pdb
        # pdb.set_trace()
        # print(request.data , "request data")
        account_name = request.data['account_name']
        product_group = request.data['product_group']
        corporate_segment = request.data['corporate_segment']
        if 'objective_function' in request.data:
            # import pdb
            # pdb.set_trace()
            ser = sc.OptimizerSerializer(request.data)
            if ser.is_valid():
                # print(ser.validated_data , ":: of serializer")
                # print(dict(ser.validated_data) , "dictionary of serializer")
                
                return Response(optimizer.process(dict(ser.validated_data)) , 200)
           
       
        tpr = model.ModelData.objects.values_list('tpr_discount',flat=True).filter(model_meta__account_name = account_name, model_meta__product_group = product_group)
        
      
        min_consecutive_promo,max_consecutive_promo,min_length_gap,tot_promo_min,tot_promo_max,no_of_promo, no_of_waves  = optimizer.get_promo_wave_values(list(tpr))

    
        serializer = sc.OptimizerSerializer({'param_total_promo_min' : tot_promo_min,
                                             'param_total_promo_max' : tot_promo_max,
                                             'param_promo_gap' : min_length_gap,
                                             'param_max_consecutive_promo' : max_consecutive_promo,
                                             'param_min_consecutive_promo' : min_consecutive_promo,
                                        'param_mac' : 1.0, 'param_rp' : 1.0,'param_trade_expense' : 1.0,'param_units' : 1.0,'param_nsv' :1.0,'param_gsv' : 1.0,'param_sales' : 1.0,'param_mac_perc' : 1.0,'param_rp_perc' : 1.0,
                                        'config_mac' : True, 'config_rp' : True,'config_trade_expense' : False,'config_units' : False,'config_nsv' :False,'config_gsv' : False,'config_sales' : False,'config_mac_perc' : False,
                                        'config_rp_perc' : True, 'config_min_consecutive_promo' : True, 'config_max_consecutive_promo' : True, 'config_promo_gap' : True,
                                             'account_name' :account_name,'corporate_segment' : corporate_segment,'brand':'','brand_format':'',
                                             'product_group' : product_group,'strategic_cell':'','result' : ''} )
        
        # import pdb
        # pdb.set_trace()
        if serializer.is_valid():
            
            optimizer.process(dict(serializer))
            return Response(serializer.data,200)
        return Response(serializer.data,200)


class SaveOptimier(viewsets.GenericViewSet):
    authentication_classes = (TokenAuthentication,)
    serializer_class = sc.SaveOptimizer
    def get(self, request, format=None):
        serializer = sc.SaveOptimizer()
        return Response(serializer.data)
    
    def post(self, request, format=None):
        # import pdb
        # pdb.set_trace()
        
        if model.SavedScenario.objects.filter(name =  request.data['name']).exists():
            # return Response({"error" : scenario.id} , status=status.HTTP_409_CONFLICT)
            raise exception.AlredyExistsException("{} already exists".format(request.data['name']))
        scenario = model.SavedScenario(
            scenario_type = request.data['type'],
            name =  request.data['name'],
            comments = request.data['comments'],
            user = request.user
            
        )   
        scenario.save()
        meta = model.ModelMeta.objects.get(
        id = request.data["meta_id"]
    )
       
       
        val = request.data['optimizer_data']
        weekly = val
        bulk_obj = []
        for week in weekly:
             bulk_obj.append(model.OptimizerSave(
                 
                  model_meta=  meta,
                  saved_scenario = scenario,
                  week = week['week'],
                  optimum_promo=week['Optimum_Promo'],
                  optimum_co_investment = week['Coinvestment'],
                  mechanic = week["Mechanic"]
             ))
        model.OptimizerSave.objects.bulk_create(bulk_obj)   
        return Response({"message" : scenario.id} ,status=status.HTTP_201_CREATED)
    
    
class MapOptimizerPromo(APIView):
    serializer_class = sc.MapOptimizerPromoSerializer
    authentication_classes = (TokenAuthentication,)
    permission_classes = (IsAuthenticated,)
  
    def create_optimizer(self,request):
        saved_scenario = model.SavedScenario(
            scenario_type = request.data['save_scenario.scenario_type'],
            name = request.data['save_scenario.name'],
            comments = request.data['save_scenario.comments'],
            user = request.user
        )
        saved_scenario.save()
        promo_save = model.PromoSave.objects.get(saved_scenario__id = int(request.data['promo_id']))
        meta = model.ModelMeta.objects.get(
            account_name__iexact = promo_save.account_name,
            product_group__iexact = promo_save.product_group
        )
      
       
        val = request.data['optimizer_data']
        weekly = ast.literal_eval(val)
        bulk_obj = []
        for week in weekly:
             bulk_obj.append(model.OptimizerSave(
                 
                  model_meta =  meta,
                  saved_scenario = saved_scenario,
                  promo_save = promo_save,
                  week = week['week'],
                  optimum_promo=week['Optimum_Promo'],
                  optimum_co_investment = week['Coinvestment']
             ))
        model.OptimizerSave.objects.bulk_create(bulk_obj)  
    def post(self, request, format=None):
        self.create_optimizer(request)
        return Response({}, status=status.HTTP_201_CREATED)


class MapOptimizerPricing(APIView):
    serializer_class = sc.MapOptimizerPricingSerializer
    authentication_classes = (TokenAuthentication,)
    permission_classes = (IsAuthenticated,)
  
    def create_optimizer(self,request):
        saved_scenario = model.SavedScenario(
            scenario_type = request.data['save_scenario.scenario_type'],
            name = request.data['save_scenario.name'],
            comments = request.data['save_scenario.comments'],
            user = request.user
        )
        saved_scenario.save()
        pricing_save = model.PricingSave.objects.get(id = int(request.data['pricing_id']))
        meta = model.ModelMeta.objects.get(
            account_name__iexact = pricing_save.account_name,
            product_group__iexact = pricing_save.product_group
        )
      
       
        val = request.data['optimizer_data']
        weekly = ast.literal_eval(val)
        bulk_obj = []
        for week in weekly:
             bulk_obj.append(model.OptimizerSave(
                 
                  model_meta =  meta,
                  saved_scenario = saved_scenario,
                  pricing_save = pricing_save,
                  week = week['week'],
                  optimum_promo=week['Optimum_Promo'],
                  optimum_co_investment = week['Coinvestment']
             ))
        model.OptimizerSave.objects.bulk_create(bulk_obj)  
    def post(self, request, format=None):
        self.create_optimizer(request)
        return Response({}, status=status.HTTP_201_CREATED)
    
    

class ModelOptimize(viewsets.GenericViewSet):
    authentication_classes = (TokenAuthentication,)
    permission_classes = (perm.OptimizerPermission,)
    serializer_class = sc.OptimizerMeta
    queryset =model.ModelMeta.objects.all()
    
    def retrieve(self):
        return Response(status=200)
    def get(self, request, format=None):
        serializer = sc.OptimizerMeta(self.queryset , many=True)
        # serializer = sc.OptimizerMeta(self.queryset , many=True)
        return Response(serializer.data)
    def get_serializer_class(self):
        return sc.OptimizerMeta
    
    def get_queryset(self):
        return super().get_queryset()
    def post(self,request,format=None):
        # import pdb
        # pdb.set_trace()
        # # print(request.data , "request data")
        account_name = request.data['account_name']
        product_group = request.data['product_group']
        corporate_segment = request.data['corporate_segment']
        if 'objective_function' in request.data:
            # import pdb
            # pdb.set_trace()
            ser = sc.OptimizerSerializer(request.data)
            print(ser.is_valid() , "is valid check")
            print(ser.errors , "errors valid check")
            if ser.is_valid():
                # print(ser.validated_data , ":: of serializer")
                # print(dict(ser.validated_data) , "dictionary of serializer")
                
                return Response(optimizer.process(dict(ser.validated_data)) , 200)
           
       
        tpr = model.ModelData.objects.values(
            'tpr_discount' , 'year','quater' , 'week' , 'date','promo_depth' , 'co_investment',
            'flag_promotype_motivation' , 'flag_promotype_n_pls_1' , 'flag_promotype_traffic'
            ).filter(
            model_meta__account_name = account_name, model_meta__product_group = product_group
            ).order_by('week')
            
            
        
        min_consecutive_promo,max_consecutive_promo,min_length_gap,tot_promo_min,tot_promo_max,no_of_promo, no_of_waves = optimizer.get_promo_wave_values([i['tpr_discount'] for i in tpr])

    
        serializer = sc.OptimizerSerializer({'param_total_promo_min' : tot_promo_min,
                                             'param_total_promo_max' : tot_promo_max,
                                             'param_promo_gap' : min_length_gap,
                                             'param_max_consecutive_promo' : max_consecutive_promo,
                                             'param_min_consecutive_promo' : min_consecutive_promo,
                                        'param_mac' : 1.0, 'param_rp' : 1.0,'param_trade_expense' : 1.0,'param_units' : 1.0,'param_nsv' :1.0,'param_gsv' : 1.0,'param_sales' : 1.0,'param_mac_perc' : 1.0,'param_rp_perc' : 1.0,
                                        'config_mac' : True, 'config_rp' : True,'config_trade_expense' : False,'config_units' : False,'config_nsv' :False,'config_gsv' : False,'config_sales' : False,'config_mac_perc' : False,
                                        'config_rp_perc' : True, 'config_min_consecutive_promo' : True, 'config_max_consecutive_promo' : True, 'config_promo_gap' : True,
                                             'account_name' :account_name,'corporate_segment' : corporate_segment,'brand':'','brand_format':'',
                                             'product_group' : product_group,'strategic_cell':'','result' : ''} )
        

        if serializer.is_valid():
            
            optimizer.process(dict(serializer))
            return Response(serializer.data,200)
        res = {"data" : serializer.data , "weekly" : tpr}
        res["data"]["param_no_of_waves"] = no_of_waves
        res["data"]["param_no_of_promo"] = no_of_promo
        return Response(res,200)